import argparse
import gzip
import json
import math
import re
from time import perf_counter

import transformers
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-1.5B"
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj")


def read_jsonl(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def score_tokens(model, token_ids, first_target):
    if not 1 <= first_target < len(token_ids):
        raise ValueError("The scoring span must have a context and a target.")
    tokens = torch.tensor([token_ids], device=model.device)
    logits = model(tokens[:, :-1], use_cache=False).logits[0, first_target - 1:]
    targets = tokens[0, first_target:]
    nll = torch.nn.functional.cross_entropy(logits.float(), targets, reduction="sum")
    value = nll.item()
    if not math.isfinite(value):
        raise FloatingPointError(
            f"Non-finite NLL for {len(token_ids)} tokens starting at target {first_target}. "
            "Check the FP16 model and attention backend."
        )
    return value, targets.numel()


def perplexity(model, rows):
    windows = []
    for row in rows:
        tokens = row["token_ids"]
        first, stop = row["score_target_offsets"]
        if not 1 <= first < stop <= len(tokens) <= 512:
            raise ValueError(f"Invalid score offsets for source row {row['source_row']}")
        nll, count = score_tokens(model, tokens[:stop], first)
        if count != row["scored_tokens"]:
            raise ValueError("Window scored-token count disagrees with offsets")
        windows.append(dict(source_row=row["source_row"], token_offset=row["token_offset"],
                            nll_sum=nll, scored_tokens=count))
    count = sum(row["scored_tokens"] for row in windows)
    if count != 50_000:
        raise ValueError(f"Expected 50,000 scored tokens, got {count}")
    mean_nll = math.fsum(row["nll_sum"] for row in windows) / count
    return dict(scored_tokens=count, mean_nll=mean_nll,
                perplexity=math.exp(mean_nll), windows=windows)


def hellaswag_text(text):
    # Formatting from lm-evaluation-harness v0.4.9 HellaSwag.
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def hellaswag(model, tokenizer, rows, limit=2000):
    rows = sorted(rows, key=lambda row: row["order"])
    if len(rows) != 2000 or len({row["source_row_index"] for row in rows}) != 2000:
        raise ValueError("Expected 2,000 distinct HellaSwag examples")
    examples = []
    for row in rows[:limit]:
        doc = row["example"]
        context = hellaswag_text(f"{doc['activity_label']}: {doc['ctx_a']} {doc['ctx_b'].capitalize()}")
        # Match the harness's trailing whitespace transfer and joint encoding.
        prefix = context.rstrip()
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        label = int(doc["label"])
        raw = []
        normalized = []
        for ending in map(hellaswag_text, doc["endings"]):
            tokens = tokenizer.encode(context + " " + ending, add_special_tokens=False)
            nll, _ = score_tokens(model, tokens, len(prefix_ids))
            raw.append(-nll)
            normalized.append(-nll / len(ending))
        prediction = max(range(4), key=raw.__getitem__)
        prediction_norm = max(range(4), key=normalized.__getitem__)
        examples.append(dict(id=row["source_row_index"], order=row["order"], label=label,
                             log_likelihood=raw, normalized_log_likelihood=normalized,
                             correct=prediction == label, correct_norm=prediction_norm == label))
    return dict(accuracy=sum(row["correct"] for row in examples) / len(examples),
                acc_norm=sum(row["correct_norm"] for row in examples) / len(examples),
                examples=examples)


def parameter_inventory(model):
    eligible = {}
    for layer_index, layer in enumerate(model.model.layers):
        for name in PROJECTIONS:
            module = getattr(layer.self_attn if name in PROJECTIONS[:4] else layer.mlp, name)
            eligible[f"{layer_index}.{name}"] = module.weight.numel()
    unique = {}
    for parameter in model.parameters():
        storage = parameter.untyped_storage()
        unique[storage.data_ptr()] = storage.nbytes()
    return dict(eligible_weights=sum(eligible.values()), projections=eligible,
                dense_parameter_bytes=sum(unique.values()))


def benchmark_cuda(model, token_id, warmups=2, repetitions=5):
    results = []
    for length in (512, 2048):
        prompt = torch.full((1, length), token_id, device=model.device, dtype=torch.long)
        for iteration in range(warmups + repetitions):
            torch.cuda.synchronize(model.device)
            torch.cuda.reset_peak_memory_stats(model.device)
            start = perf_counter()
            output = model(prompt, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize(model.device)
            prefill_ms = (perf_counter() - start) * 1000
            cache = output.past_key_values
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize(model.device)
            start = perf_counter()
            for _ in range(64):
                output = model(next_token, past_key_values=cache,
                               use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize(model.device)
            decode_ms = (perf_counter() - start) * 1000
            peak_bytes = torch.cuda.max_memory_allocated(model.device)
            if iteration >= warmups:
                results.append(dict(prompt_tokens=length, decode_steps=64,
                                    repetition=iteration - warmups + 1,
                                    prefill_ms=prefill_ms, decode_ms=decode_ms,
                                    peak_allocated_bytes=peak_bytes))
            del output, cache, next_token
    return dict(batch_size=1, warmups_per_shape=warmups, repetitions=repetitions,
                repeated_prompt_token_id=token_id, samples=results)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Score the Task 1 FP16 baseline.")
    parser.add_argument("--course-data", type=Path, required=True,
                        help="Root of the materialized atml-pa4-datasets checkout")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.course_data / "data"
    revisions = json.loads((data / "local_source_revisions.json").read_text(encoding="utf-8"))
    revision = revisions["revisions"]["model_base"]
    filenames = dict(wikitext="wikitext2_test_windows_locked.jsonl.gz",
                     c4="c4_validation_windows_locked.jsonl.gz",
                     hellaswag="hellaswag_val_locked.jsonl.gz")
    manifests = {name: read_jsonl(data / "public_locked" / filename)
                 for name, filename in filenames.items()}
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required for the FP16 baseline.")
    torch.manual_seed(6304)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, dtype=torch.float16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    calibration = read_jsonl(data / "public_locked" / "c4_calibration_locked.jsonl.gz")
    for row in calibration[:4]:
        score_tokens(model, row["token_ids"], 1)
    result = dict(
        model=MODEL_ID, revision=model.config._commit_hash,
        tokenizer=type(tokenizer).__name__, seed=6304, dtype="float16",
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        transformers_version=transformers.__version__, source_revisions=revisions,
        hellaswag_format="lm-evaluation-harness v0.4.9 zero-shot",
        cuda_version=torch.version.cuda, attention_backend=model.config._attn_implementation,
        parameters=parameter_inventory(model),
        benchmark=benchmark_cuda(model, manifests['wikitext'][0]['token_ids'][0]),
        wikitext=perplexity(model, manifests["wikitext"]),
        c4=perplexity(model, manifests["c4"]),
        hellaswag=hellaswag(model, tokenizer, manifests["hellaswag"]),
    )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
