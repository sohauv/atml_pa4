"""Task 3 teacher-rollout preparation and matched on/off-policy LoRA distillation."""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("USE_TF", "0")

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from task1.baseline import read_jsonl
from task3.speculative import ModelState, markov_check, speculative


TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
QUOTAS = {"math": 30_000, "short": 10_000}
CAPS = {"math": 192, "short": 32}
UPDATE_TOKENS = 256


def write_json(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def save_checkpoint(path, payload):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def synchronize():
    torch.cuda.synchronize()


def load_model(name, revision):
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype=torch.float16, attn_implementation="sdpa",
    ).cuda().eval()
    return model.requires_grad_(False)


class LoRA(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.base = layer
        self.a = nn.Parameter(torch.empty(8, layer.in_features, device=layer.weight.device,
                                          dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(layer.out_features, 8, device=layer.weight.device,
                                          dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        update = F.linear(F.linear(x.float(), self.a), self.b)
        return self.base(x) + (2 * update).to(x.dtype)


def adapter_layers(model):
    for i, block in enumerate(model.model.layers):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            yield f"{i}.{name}", block.self_attn, name


def install_adapters(model):
    for _, attention, name in adapter_layers(model):
        setattr(attention, name, LoRA(getattr(attention, name)))


def adapter_state(model):
    return {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}


def load_adapters(model, state):
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(params) != set(state):
        raise ValueError("Adapter checkpoint parameters differ")
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)


@torch.no_grad()
def merge_adapters(model, state):
    modules = dict(model.named_modules())
    for name, attention, projection in adapter_layers(model):
        layer = getattr(attention, projection)
        prefix = f"model.layers.{name.split('.')[0]}.self_attn.{projection}"
        if modules[prefix] is not layer:
            raise ValueError("Unexpected adapter projection path")
        a, b = (state[f"{prefix}.{key}"].to(layer.weight.device) for key in ("a", "b"))
        layer.weight.copy_((layer.weight.float() + 2 * (b @ a)).half())


def prompt_ids(tokenizer, text):
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                         add_generation_prompt=True)


def tensor(ids):
    return torch.tensor([ids], device="cuda", dtype=torch.long)


@torch.inference_mode()
def rollout(model, ids, seed, cap, eos):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    state = ModelState.prefill(model, tensor(ids))
    response = []
    for step in range(cap):
        token = torch.multinomial(state.probabilities, 1, generator=rng).item()
        response.append(token)
        if token in eos or step + 1 == cap:
            break
        state.advance([token])
    return response, ("eos" if response[-1] in eos else "cap")


def training_prompts(course, lock):
    from datasets import load_dataset

    manifest = read_jsonl(course / "data/public_manifests/gsm8k_train_ids.jsonl")
    dataset = load_dataset("openai/gsm8k", "main", split="train", revision=lock["gsm8k"])
    math_rows = [dict(id=f"GSM8K-train-{row['order']:04d}", text=dataset[row["row_index"]]["question"],
                      source_row_index=row["row_index"])
                 for row in sorted(manifest, key=lambda row: row["order"])]
    short_rows = [dict(id=row["id"], text=row["input"], family=row["family"])
                  for row in read_jsonl(course / "data/short_form/short_train.jsonl")]
    if len(math_rows) != 500 or len(short_rows) != 300:
        raise ValueError("Unexpected training prompt count")
    return {"math": math_rows, "short": short_rows}


def teacher_pool(target, prompts, output, eos, deadline):
    path = output / "teacher_rollouts.json"
    pool = json.loads(path.read_text()) if path.exists() else {}
    for index, (category, row) in enumerate((c, r) for c, rows in prompts.items() for r in rows):
        if row["id"] in pool:
            continue
        if perf_counter() >= deadline:
            return None
        seed = 6304 + 10007 * index
        synchronize()
        start = perf_counter()
        ids, termination = rollout(target, row["tokens"], seed, CAPS[category], eos)
        synchronize()
        pool[row["id"]] = dict(category=category, response_ids=ids, seed=seed,
                               termination=termination, generation_seconds=perf_counter()-start)
        write_json(path, pool)
        if (index + 1) % 25 == 0:
            print(f"Teacher rollouts: {index + 1}/800", flush=True)
    return pool


def scored_logits(model, prompt, response, start, stop):
    tokens = tensor(prompt + response[:stop - 1])
    positions = torch.arange(len(prompt) + start - 1, len(prompt) + stop - 1, device="cuda")
    return model(tokens, use_cache=False, logits_to_keep=positions).logits[0].float()


def initial_state():
    return dict(counts=dict.fromkeys(QUOTAS, 0), indices=dict.fromkeys(QUOTAS, 0),
                presentations=[], updates=[], pending_tokens=0, pending_kl=0.,
                times=dict(teacher_scoring=0., student_rollout=0., backward=0., optimizer=0.),
                peak_allocated_bytes=0)


def train(target, student, prompts, pool, eos, output, method, seed, deadline):
    run = output / f"{method}-{seed}"
    run.mkdir(exist_ok=True)
    checkpoint = run / "resume.pt"
    report = run / "training.json"
    adapter = run / "adapter.pt"
    if report.exists() and json.loads(report.read_text()).get("finished"):
        return True
    torch.manual_seed(seed)
    install_adapters(student)
    student.train()
    parameters = {name: p for name, p in student.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(parameters.values(), lr=3e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda")
    state = initial_state()
    if checkpoint.exists():
        previous = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = previous["state"]
        load_adapters(student, previous["adapter"])
        optimizer.load_state_dict(previous["optimizer"])
        scaler.load_state_dict(previous["scaler"])
        for name, gradient in previous["grads"].items():
            parameters[name].grad = gradient.to("cuda")
        del previous

    schedule = {}
    rng = random.Random(seed)
    for category, rows in prompts.items():
        schedule[category] = rows.copy()
        rng.shuffle(schedule[category])

    def save_progress():
        save_checkpoint(checkpoint, dict(
            state=state, adapter=adapter_state(student), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),
            grads={name: p.grad.detach().cpu() for name, p in parameters.items() if p.grad is not None},
        ))
        write_json(report, state)

    def update():
        synchronize()
        start = perf_counter()
        scaler.unscale_(optimizer)
        pending = state["pending_tokens"]
        # Accumulate losses divided by the nominal quota, then correct the final short update.
        if pending != UPDATE_TOKENS:
            for p in parameters.values():
                if p.grad is not None:
                    p.grad.mul_(UPDATE_TOKENS / pending)
        norm = nn.utils.clip_grad_norm_(parameters.values(), 1.)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        optimizer.zero_grad(set_to_none=True)
        synchronize()
        state["times"]["optimizer"] += perf_counter() - start
        grad_norm = norm.item()
        state["updates"].append(dict(
            attempted_update=len(state["updates"]) + 1, scored_tokens=sum(state["counts"].values()),
            tokens=pending, mean_kl=state["pending_kl"]/pending, lr=3e-4,
            skipped=skipped, grad_norm=grad_norm if math.isfinite(grad_norm) else None,
        ))
        state["pending_tokens"], state["pending_kl"] = 0, 0.

    torch.cuda.reset_peak_memory_stats()
    while any(state["counts"][c] < QUOTAS[c] for c in QUOTAS):
        if perf_counter() >= deadline:
            save_progress()
            return False
        candidates = [c for c in QUOTAS if state["counts"][c] < QUOTAS[c]]
        category = min(candidates, key=lambda c: state["indices"][c])
        index = state["indices"][category]
        row = schedule[category][index % len(schedule[category])]
        prompt = row["tokens"]
        rollout_seed = seed * 1_000_003 + index * 2 + int(category == "short")
        if method == "off_policy":
            frozen = pool[row["id"]]
            response, termination, rollout_seed = frozen["response_ids"], frozen["termination"], frozen["seed"]
        else:
            synchronize()
            start = perf_counter()
            student.eval()
            with torch.autocast("cuda", dtype=torch.float16):
                response, termination = rollout(student, prompt, rollout_seed, CAPS[category], eos)
            student.train()
            synchronize()
            state["times"]["student_rollout"] += perf_counter() - start
        score_count = min(len(response), QUOTAS[category] - state["counts"][category])
        start_token = 0
        while start_token < score_count:
            length = min(score_count - start_token, UPDATE_TOKENS - state["pending_tokens"], 32)
            end = start_token + length
            synchronize()
            started = perf_counter()
            with torch.no_grad():
                target_logp = scored_logits(target, prompt, response, start_token, end).log_softmax(-1)
            synchronize()
            state["times"]["teacher_scoring"] += perf_counter() - started
            started = perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = scored_logits(student, prompt, response, start_token, end)
            kl = F.kl_div(logits.log_softmax(-1), target_logp, reduction="sum", log_target=True)
            value = kl.item()
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite distillation KL in {method}/{seed}")
            scaler.scale(kl / UPDATE_TOKENS).backward()
            synchronize()
            state["times"]["backward"] += perf_counter() - started
            del target_logp, logits, kl
            start_token = end
            state["counts"][category] += length
            state["pending_tokens"] += length
            state["pending_kl"] += value
            if state["pending_tokens"] == UPDATE_TOKENS:
                update()
        state["presentations"].append(dict(
            category=category, source_id=row["id"], rollout_seed=rollout_seed,
            response_token_ids=response, termination=termination, scored_tokens=score_count,
            quota_truncated=score_count < len(response),
        ))
        state["indices"][category] += 1
        state["peak_allocated_bytes"] = max(state["peak_allocated_bytes"], torch.cuda.max_memory_allocated())
        if len(state["presentations"]) % 25 == 0:
            save_progress()
            print(method, seed, state["counts"], flush=True)
    if state["pending_tokens"]:
        update()
    assert state["counts"] == QUOTAS
    assert len(state["updates"]) == math.ceil(sum(QUOTAS.values()) / UPDATE_TOKENS)
    state.update(finished=True, method=method, seed=seed,
                 adapter_parameters=sum(p.numel() for p in parameters.values()),
                 backbone_parameters=sum(p.numel() for p in student.parameters() if not p.requires_grad),
                 shared_teacher_rollout_seconds=sum(row["generation_seconds"] for row in pool.values())
                 if method == "off_policy" else 0.,
                 overflowed_updates=sum(row["skipped"] for row in state["updates"]))
    save_checkpoint(adapter, adapter_state(student))
    save_progress()
    return True


@torch.inference_mode()
def validate_models(target, draft, tokens, eos):
    prompt = tensor(tokens)
    checks = {}
    for name, model in (("target", target), ("draft", draft)):
        cut = prompt.shape[1] - 2
        full = model(prompt, use_cache=False, logits_to_keep=2).logits[0].float()
        prior = model(prompt[:, :cut], use_cache=True, logits_to_keep=1)
        state = ModelState(model, prior.past_key_values, prior.logits[0, -1].float().softmax(-1))
        split = state.advance(prompt[:, cut:])
        theoretical = full.softmax(-1)
        delta = (split - theoretical).abs().max().item()
        if not math.isfinite(delta) or delta > .025 or state.length != prompt.shape[1]:
            raise ValueError(f"{name} split-cache validation failed: {delta}")
        # Crop away the last token and replay it to verify rollback semantics.
        state.cache.crop(prompt.shape[1] - 1)
        replay = state.advance(prompt[:, -1:])
        rollback = (replay[-1] - theoretical[-1]).abs().max().item()
        if not math.isfinite(rollback) or rollback > .025:
            raise ValueError(f"{name} rollback validation failed: {rollback}")
        checks[name] = dict(split_max_probability_error=delta, rollback_max_probability_error=rollback)
    for gamma in (1, 2, 4):
        sample = speculative(target, draft, prompt, gamma, 11, eos, 6304 + gamma)
        if len(sample["token_ids"]) > 11 or not sample["rounds"]:
            raise ValueError("Invalid real-model generation length")
        checks[f"gamma_{gamma}"] = sample
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    deadline = perf_counter() + args.hours * 3600
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.course_data / "scripts"))
    from setup_data import revisions

    lock = revisions()
    target = load_model(TARGET, lock["model_target"])
    tokenizer = AutoTokenizer.from_pretrained(TARGET, revision=lock["model_target"])
    draft_tokenizer = AutoTokenizer.from_pretrained(DRAFT, revision=lock["model_draft"])
    eos = target.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    prompts = training_prompts(args.course_data, lock)
    for group in prompts.values():
        for row in group:
            row["tokens"] = prompt_ids(tokenizer, row["text"])
            if row["tokens"] != prompt_ids(draft_tokenizer, row["text"]):
                raise ValueError(f"Chat template mismatch for {row['id']}")
    metadata = dict(revisions=lock, seed=6304, gpu=torch.cuda.get_device_name(),
                    torch_version=torch.__version__, eos=sorted(eos),
                    quotas=QUOTAS, caps=CAPS, temperature=1, top_p=1,
                    lora=dict(rank=8, alpha=16, dropout=0, projections=["q", "k", "v", "o"]))
    path = output / "metadata.json"
    if path.exists() and json.loads(path.read_text()) != metadata:
        raise ValueError("Existing training checkpoint has incompatible metadata")
    write_json(path, metadata)
    write_json(output / "prompt_ids.json", {c: [r["id"] for r in rows] for c, rows in prompts.items()})
    checks = output / "sampler_checks.json"
    if not checks.exists():
        draft = load_model(DRAFT, lock["model_draft"])
        if draft.config.vocab_size != target.config.vocab_size or tokenizer.get_vocab() != draft_tokenizer.get_vocab():
            raise ValueError("Target and draft vocabulary differ")
        write_json(checks, dict(markov=markov_check(), qwen=validate_models(
            target, draft, prompts["math"][0]["tokens"], eos
        )))
        del draft
        torch.cuda.empty_cache()
    pool = teacher_pool(target, prompts, output, eos, deadline)
    if pool is None:
        print("Teacher pool checkpoint saved. Resume with the exported ZIP.", flush=True)
        return
    for method in ("off_policy", "on_policy"):
        for seed in (6304, 6305):
            if perf_counter() >= deadline:
                return
            print("Starting", method, seed, flush=True)
            student = load_model(DRAFT, lock["model_draft"])
            finished = train(target, student, prompts, pool, eos, output, method, seed, deadline)
            del student
            torch.cuda.empty_cache()
            if not finished:
                return


if __name__ == "__main__":
    main()

