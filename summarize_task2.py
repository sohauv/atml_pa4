"""Summarize the saved Task 2 benchmark archives."""
import json
import random
from pathlib import Path
from statistics import mean, median
from zipfile import ZipFile


def read_archive(path):
    with ZipFile(path) as archive:
        return {Path(name).name: json.loads(archive.read(name))
                for name in sorted(archive.namelist()) if name.endswith(".json")}


def paired_interval(values):
    rng = random.Random(6304)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return [draws[49], draws[1949]]


def aggregate_kv(rows):
    output = {}
    baseline = [row["policies"]["full"]["exact_match"] for row in rows]
    for policy in rows[0]["policies"]:
        records = [r["policies"][policy] for r in rows]
        diffs = [int(result["exact_match"]) - int(ref)
                 for result, ref in zip(records, baseline, strict=True)]
        evidence = [x["evidence_fp16_fraction"] for x in records
                    if x["evidence_fp16_fraction"] is not None]
        output[policy] = dict(
            prompts=len(records), exact_match=mean(x["exact_match"] for x in records),
            difference_from_fp16=mean(diffs),
            exact_match_given_fp16_correct=mean(x["exact_match"] for x, ref in
                                                   zip(records, baseline) if ref),
            answer_nll_per_token=sum(x["answer_nll_sum"] for x in records) /
                                 sum(x["answer_tokens"] for x in records),
            mean_kl=mean(x["reference_to_policy_kl"] for x in records),
            mean_fp16_evidence_fraction=mean(evidence) if evidence else None,
            mean_context_bytes=mean(x["packed_context_bytes"] for x in records),
            mean_fp16_context_byte_fraction=mean(x["fp16_context_byte_fraction"] for x in records),
            median_scoring_ms=median(x["question_scoring_ms"] for x in records),
            median_preprocessing_ms=median(x["preprocessing_ms"] for x in records),
            median_decode_ms=median(x["decode_ms"] for x in records),
            peak_allocated_bytes=max(x["peak_allocated_bytes"] for x in records),
        )
        if len(rows) == 180:
            output[policy]["difference_from_fp16_ci95"] = paired_interval(diffs)
    return output


def main():
    attention = read_archive("pa4-attention-results.zip")["attention.json"]
    mqar = read_archive("pa4-mqar-results.zip")
    intervention = read_archive("pa4-mqar-intervention.zip")
    sweep = intervention.pop("inference_sweep.json")
    assert len(attention["runs"]) == 72 and len(mqar) == 39 and len(intervention) == 3
    assert len(sweep["runs"]) == 16
    training = mqar | intervention
    assert all(row["finished"] and row["last_step"] == 4500 for row in training.values())
    recall = [{"condition": name.removesuffix(".json"), "method": row["method"], "K": row["K"],
               "training_regime": row["training_regime"], "seed": row["seed"],
               "test": row["test"], "test_only_window": row.get("test_only_window"),
               "learning_curve": row["learning_curve"]} for name, row in training.items()]

    kv = read_archive("pa4-kv-residual-results.zip")
    rows = [row for name, row in kv.items() if name.startswith("KV-")]
    assert len(rows) == 180
    cells = {}
    for field in ("target_tokens", "position", "distractors"):
        for value in sorted({r[field] for r in rows}):
            cells[f"{field}={value}"] = aggregate_kv([r for r in rows if r[field] == value])
    packing = read_archive("pa4-kv-sweep-results.zip")["packing_sweep.json"]
    assert len(packing) == 16 and all(len(row["samples"]) == 10 for row in packing)
    output = dict(
        attention=attention, causality=read_archive("pa4-causality-results.zip")["causality.json"],
        recall=recall, recall_inference=sweep, kv=aggregate_kv(rows), kv_breakdowns=cells,
        kv_reference=read_archive("pa4-kv-reference-results.zip")["reference_generation.json"],
        kv_append_boundary=read_archive("pa4-kv-boundary-results.zip")["boundary.json"],
        kv_packing=packing,
    )
    Path("task2-summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
