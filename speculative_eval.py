"""Held-out Task 3 prefix banks, draft quality, and paired speculative benchmarks."""
import argparse
import json
import math
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import perf_counter

import torch

from baseline import read_jsonl
from distillation import (DRAFT, TARGET, install_adapters, load_adapters, load_model,
                          merge_adapters, prompt_ids, rollout, synchronize, tensor, write_json)
from speculative import ModelState, speculative
from transformers import AutoTokenizer


DRAFTS = ("original", "off_policy-6304", "off_policy-6305", "on_policy-6304", "on_policy-6305")
BANKS = ("teacher", "original_draft", "exact_spec_original_gamma4", *DRAFTS[1:])
POSITION_BINS = ((1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_prompts(course, revision, tokenizer):
    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main", revision=revision, split="test")
    manifest = sorted(read_jsonl(course / "data/public_manifests/gsm8k_test_ids.jsonl"),
                      key=lambda row: row["order"])
    rows = [dict(id=f"GSM8K-{entry['order']:04d}", text=gsm[entry["row_index"]]["question"],
                 domain="math", answer=gsm[entry["row_index"]]["answer"].rsplit("####", 1)[-1].strip())
            for entry in manifest]
    rows.extend(dict(id=x["id"], text=x["input"], domain=x["split"], answer=x["answer"], family=x["family"])
                for x in read_jsonl(course / "data/short_form/short_test.jsonl"))
    assert len(rows) == len({row["id"] for row in rows}) == 120
    assert {domain: sum(row["domain"] == domain for row in rows)
            for domain in ("math", "familiar", "shifted")} == {"math": 60, "familiar": 30, "shifted": 30}
    for row in rows:
        row["tokens"] = prompt_ids(tokenizer, row["text"])
    return rows


def draft_model(name, rev, adapters, tokenizer, probe, info):
    if name == "original":
        return load_model(DRAFT, rev)
    state = torch.load(adapters / name / "adapter.pt", map_location="cpu", weights_only=True)
    check = info / f"{name}.json"
    compare = not check.exists()
    if compare:
        unmerged = load_model(DRAFT, rev)
        install_adapters(unmerged)
        load_adapters(unmerged, state)
        with torch.autocast("cuda", dtype=torch.float16):
            before = unmerged(probe, use_cache=False, logits_to_keep=2).logits.float()
        del unmerged
        torch.cuda.empty_cache()
    model = load_model(DRAFT, rev)
    synchronize()
    start = perf_counter()
    merge_adapters(model, state)
    synchronize()
    milliseconds = 1000 * (perf_counter() - start)
    if compare:
        after = model(probe, use_cache=False, logits_to_keep=2).logits.float()
        error = (after - before).abs()
        if not torch.isfinite(error).all():
            raise FloatingPointError(f"Invalid merge logits: {name}")
        write_json(check, dict(name=name, merge_ms=milliseconds, max_abs_error=error.max().item(),
                               mean_abs_error=error.mean().item(), argmax_agreement=
                               (after.argmax(-1) == before.argmax(-1)).float().mean().item()))
    return model


def positions_for(row, bank, length):
    rng = random.Random(f"PA4-banks-v1:6304:{row['id']}:{bank}")
    positions = []
    for lo, hi, count in POSITION_BINS:
        available = list(range(lo, min(hi, length) + 1))
        positions.extend(sorted(rng.sample(available, min(count, len(available)))))
    return positions


def make_bank(folder, name, rows, model, target, eos, deadline):
    folder.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        if perf_counter() >= deadline:
            return False
        path = folder / f"{row['id']}.json"
        if path.exists():
            continue
        seed = 6304 + 10007 * index + (0 if name in ("teacher", "exact_spec_original_gamma4") else 123)
        if name == "exact_spec_original_gamma4":
            draw = speculative(target, model, tensor(row["tokens"]), 4, 192, eos, seed)
            ids = draw["token_ids"]
            termination = "eos" if draw["eos"] else "cap"
        else:
            ids, termination = rollout(model, row["tokens"], seed, 192, eos)
        write_json(path, dict(prompt_id=row["id"], domain=row["domain"], bank=name,
                              response_ids=ids, positions=positions_for(row, name, len(ids)),
                              termination=termination, seed=seed))
    return True


def alignment_logits(model, row, rollout_record, positions):
    tokens = row["tokens"]
    sequence = tokens + rollout_record["response_ids"][:positions[-1] - 1]
    indices = torch.tensor([len(tokens) + pos - 2 for pos in positions], device="cuda")
    out = model(tensor(sequence), use_cache=False, logits_to_keep=indices)
    return out.logits[0].float().log_softmax(-1)


def prefix_alignment(model, target, row, banks):
    comparisons = {}
    for bank in BANKS:
        saved = read(banks / bank / f"{row['id']}.json")
        points = saved["positions"]
        metrics = []
        for start in range(0, len(points), 24):
            positions = points[start:start + 24]
            a = alignment_logits(target, row, saved, positions)
            b = alignment_logits(model, row, saved, positions)
            p, q = a.exp(), b.exp()
            tv = .5 * (p - q).abs().sum(-1)
            values = torch.stack(((p * (a - b)).sum(-1), (q * (b - a)).sum(-1), tv,
                                  1 - tv, (a.argmax(-1) == b.argmax(-1)).float()), dim=-1)
            if not torch.isfinite(values).all():
                raise FloatingPointError(f"Invalid prefix metrics on {row['id']} {bank}")
            metrics.extend(dict(position=i, forward_kl=v[0], reverse_kl=v[1], tv=v[2],
                                expected_acceptance=v[3], greedy_agreement=bool(v[4]))
                           for i, v in zip(positions, values.tolist(), strict=True))
        comparisons[bank] = metrics
    return comparisons


def target_only(model, prompt, cap, eos, seed):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    synchronize()
    start = perf_counter()
    state = ModelState.prefill(model, prompt)
    synchronize()
    prefilled = perf_counter()
    ids = []
    for i in range(cap):
        token = torch.multinomial(state.probabilities, 1, generator=rng).item()
        ids.append(token)
        if token in eos or i + 1 == cap:
            break
        state.advance([token])
    synchronize()
    end = perf_counter()
    return dict(token_ids=ids, prefill_ms=(prefilled - start) * 1000,
                decode_ms=(end - prefilled) * 1000, total_ms=(end - start) * 1000,
                eos=ids[-1] in eos, truncated=len(ids) == cap and ids[-1] not in eos,
                target_decode_passes=len(ids) - 1)


def measured(call):
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.memory_allocated()
    result = call()
    result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), resident_before_bytes=start)
    return result


def benchmark_prompt(model, target, prompt, eos, index, fixed, draft_index):
    pairs = []
    configurations = [(g, rep, False) for g in (2, 4) for rep in range(2)]
    if fixed:
        configurations.extend((g, 0, True) for g in (2, 4))
    if (index + draft_index) % 2:
        configurations.reverse()
    for gamma, rep, ignore_eos in configurations:
        stop = set() if ignore_eos else eos
        seed = 6304 + index * 10007 + rep * 701 + gamma * 37
        baseline = lambda: measured(lambda: target_only(target, prompt, 64, stop, seed))
        proposed = lambda: measured(lambda: speculative(target, model, prompt, gamma, 64, stop, seed, measure=True))
        order = "target_first" if (index + rep + gamma // 2 + draft_index) % 2 == 0 else "speculative_first"
        if order == "target_first":
            t, s = baseline(), proposed()
        else:
            s, t = proposed(), baseline()
        pairs.append(dict(gamma=gamma, repetition=rep, fixed_64_ignores_eos=ignore_eos, seed=seed,
                          order=order, target=t, speculative=s))
    return pairs


def assess_answer(row, text):
    if row["domain"] != "math":
        clean = text.strip()
        match = clean.casefold() == row["answer"].strip().casefold()
        # Score answer-only compliance separately from detecting the answer.
        contained = re.search(r"(?<!\w)" + re.escape(row["answer"].strip()) + r"(?!\w)",
                              clean, flags=re.IGNORECASE) is not None
        return dict(correct=contained, format_compliant=match, extracted=clean if match else None)
    tail = text.rsplit("####", 1)[-1] if "####" in text else text
    candidates = re.findall(r"-?\d[\d,]*(?:\.\d+)?", tail)
    answer = candidates[-1].replace(",", "") if candidates else None
    try:
        correct = answer is not None and Decimal(answer) == Decimal(row["answer"].replace(",", ""))
    except InvalidOperation:
        correct = False
    return dict(correct=correct, format_compliant=None, extracted=answer)


def warmup(model, target, prompt, eos):
    for seed in (6304, 6305):
        target_only(target, prompt, 8, eos, seed)
        speculative(target, model, prompt, 2, 8, eos, seed, measure=True)


def isolated_cost(model, target):
    reports = []
    for length in (128, 512, 1024):
        prompt = tensor([970] * length)
        for label, active in (("draft", model), ("target", target)):
            for width in ((1,) if label == "draft" else (1, 2, 4)):
                times = []
                for i in range(7):
                    state = ModelState.prefill(active, prompt)
                    synchronize()
                    start = perf_counter()
                    state.advance([970] * width)
                    synchronize()
                    if i >= 2:
                        times.append((perf_counter() - start) * 1000)
                reports.append(dict(prompt_length=length, model=label, width=width, samples_ms=times))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-data", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    deadline = perf_counter() + args.hours * 3600
    adapters, out = args.adapters, args.output_dir
    lock = read(adapters / "metadata.json")["revisions"]
    for name in DRAFTS[1:]:
        assert read(adapters / name / "training.json")["finished"] and (adapters / name / "adapter.pt").exists()
    banks, info, per_prompt = (out / name for name in ("banks", "model_checks", "per_prompt"))
    for folder in (banks, info, per_prompt):
        folder.mkdir(parents=True, exist_ok=True)
    target = load_model(TARGET, lock["model_target"])
    tokenizer = AutoTokenizer.from_pretrained(TARGET, revision=lock["model_target"])
    rows = load_prompts(args.course_data, lock["gsm8k"], tokenizer)
    stop = target.generation_config.eos_token_id
    eos = set(stop if isinstance(stop, list) else [stop])
    fixed = {row["id"] for row in rows if row["domain"] == "math"}
    fixed = set(sorted(fixed)[:24])
    manifest = dict(revisions=lock, gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                    precision="float16", backend="sdpa", cap=64, gammas=[2,4], repetitions=2,
                    fixed64_math_ids=sorted(fixed), prompt_ids=[r["id"] for r in rows],
                    bank_recipe="Course seeds and position strata; sequential temperature-1 multinomial rollout",
                    timing="CUDA-synchronized paired runs; includes Python and sampling overhead")
    provenance = out / "metadata.json"
    if provenance.exists() and read(provenance) != manifest:
        raise ValueError("Existing evaluation has different provenance")
    write_json(provenance, manifest)
    probe = tensor(rows[0]["tokens"])
    if not make_bank(banks / "teacher", "teacher", rows, target, target, eos, deadline):
        return
    for name in DRAFTS:
        pending = ("original_draft", "exact_spec_original_gamma4") if name == "original" else (name,)
        if all((banks / bank).exists() and len(list((banks / bank).glob("*.json"))) == 120 for bank in pending):
            continue
        if perf_counter() >= deadline:
            return
        print("Generating frozen banks:", name, flush=True)
        model = draft_model(name, lock["model_draft"], adapters, tokenizer, probe, info)
        for bank in pending:
            if not make_bank(banks / bank, bank, rows, model, target, eos, deadline):
                return
        del model
        torch.cuda.empty_cache()
    for draft_index, name in enumerate(DRAFTS):
        folder = per_prompt / name
        folder.mkdir(exist_ok=True)
        if len(list(folder.glob("*.json"))) == 120:
            continue
        print("Evaluating", name, flush=True)
        model = draft_model(name, lock["model_draft"], adapters, tokenizer, probe, info)
        started = perf_counter()
        warmup(model, target, probe, eos)
        warmup_ms = 1000 * (perf_counter() - started)
        isolated = info / f"{name}-cost.json"
        if not isolated.exists():
            write_json(isolated, dict(warm_setup_ms=warmup_ms, samples=isolated_cost(model, target)))
        for index, row in enumerate(rows):
            if perf_counter() >= deadline:
                return
            path = folder / f"{row['id']}.json"
            if path.exists():
                continue
            prompt = tensor(row["tokens"])
            banks_result = prefix_alignment(model, target, row, banks)
            pairs = benchmark_prompt(model, target, prompt, eos, index, row["id"] in fixed, draft_index)
            cap = 512 if row["domain"] == "math" else 32
            quality = target_only(model, prompt, cap, eos, 6304 + 10007 * index)
            text = tokenizer.decode(quality["token_ids"], skip_special_tokens=True)
            write_json(path, dict(id=row["id"], domain=row["domain"], prompt_tokens=len(row["tokens"]),
                                  answer=row["answer"], prefix_metrics=banks_result, paired_runs=pairs,
                                  standalone=dict(**quality, text=text, **assess_answer(row, text))))
            if (index + 1) % 10 == 0:
                print(name, index + 1, "/120", flush=True)
        del model
        torch.cuda.empty_cache()
    optional = info / "unmerged-overhead.json"
    if not optional.exists() and perf_counter() < deadline:
        name = "off_policy-6304"
        model = draft_model(name, lock["model_draft"], adapters, tokenizer, probe, info)
        merged = [target_only(model, tensor(row["tokens"]), 64, set(), 6304)["decode_ms"]
                  for row in rows[:8]]
        del model
        model = load_model(DRAFT, lock["model_draft"])
        install_adapters(model)
        load_adapters(model, torch.load(adapters / name / "adapter.pt", map_location="cpu", weights_only=True))
        with torch.autocast("cuda", dtype=torch.float16):
            target_only(model, probe, 8, set(), 6304)
            unmerged = [target_only(model, tensor(row["tokens"]), 64, set(), 6304)["decode_ms"]
                        for row in rows[:8]]
        write_json(optional, dict(draft=name, prompt_ids=[row["id"] for row in rows[:8]],
                                 merged_decode_ms=merged, unmerged_decode_ms=unmerged))


if __name__ == "__main__":
    with torch.inference_mode():
        main()

