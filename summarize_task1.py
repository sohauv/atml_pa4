"""Build paired Task 1 summaries from saved baseline and compression runs."""
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median
from zipfile import ZipFile

from compare_pruning import paired_quality


def archive_results(filename):
    with ZipFile(filename) as archive:
        return [record for name in sorted(archive.namelist()) if name.endswith(".json")
                if "condition" in (record := json.loads(archive.read(name)))]


def interval(rows, statistic):
    rng = random.Random(6304)
    draws = sorted(statistic(rng.choices(rows, k=len(rows))) for _ in range(2000))
    return [draws[49], draws[1949]]


def paired_intervals(reference, result):
    paired_quality(reference, result)
    intervals = {}
    for corpus in ("wikitext", "c4"):
        pairs = [(x["nll_sum"], y["nll_sum"], x["scored_tokens"])
                 for x, y in zip(result[corpus]["windows"], reference[corpus]["windows"], strict=True)]

        def delta(sample):
            n = sum(row[2] for row in sample)
            return (math.exp(math.fsum(row[0] for row in sample) / n)
                    - math.exp(math.fsum(row[1] for row in sample) / n))

        intervals[corpus] = interval(pairs, delta)
    for metric in ("correct", "correct_norm"):
        gaps = [int(x[metric]) - int(y[metric]) for x, y in zip(
            result["hellaswag"]["examples"], reference["hellaswag"]["examples"], strict=True)]
        intervals[f"hellaswag_{metric}_percentage_points"] = interval(
            gaps, lambda sample: 100 * sum(sample) / len(sample))
    return intervals


def timing(record):
    rows = defaultdict(list)
    for sample in record["benchmark"]["samples"]:
        rows[sample["prompt_tokens"]].append(sample)
    assert set(rows) == {512, 2048} and all(len(group) == 5 for group in rows.values())
    return {length: dict(prefill_ms_median=median(x["prefill_ms"] for x in group),
                         decode_64_steps_ms_median=median(x["decode_ms"] for x in group),
                         peak_allocated_bytes=max(x["peak_allocated_bytes"] for x in group))
            for length, group in rows.items()}


def main():
    baseline = json.loads(Path("baseline-20261121T134102Z.json").read_text())
    records = [json.loads(path.read_text()) for pattern in ("wanda*.json", "magnitude*.json")
               for path in sorted(Path(".").glob(pattern))]
    for name in ("nonuniform", "q1", "awq-folded"):
        records.extend(archive_results(f"pa4-{name}-results.zip"))
    by_name = {record["condition"]: record for record in records}
    assert len(records) == len(by_name) == 17
    results = {}
    for name, record in sorted(by_name.items()):
        for field in ("model", "revision", "source_revisions", "dtype", "attention_backend", "gpu"):
            if record[field] != baseline[field]:
                raise ValueError(f"{name}: mismatched {field}")
        packed = record.get("estimated_packed_model_bytes", record.get("estimated_mask16_model_bytes"))
        projection = record["projections"]
        allocation = {}
        for key, row in projection.items():
            fields = ("relative_squared_error", "masked", "weights", "bits", "exponent",
                      "compensation_bytes", "estimated_packed_bytes", "folded_into")
            allocation[key] = {field: row[field] for field in fields if field in row}
        if name.startswith("folded_"):
            assert sum(row.get("folded_into") == "up_proj_output" for row in projection.values()) == 28
            assert all(row["compensation_bytes"] == 0 for row in projection.values() if "folded_into" in row)
            inventory = baseline["parameters"]
            assert packed == (inventory["dense_parameter_bytes"] - 2 * inventory["eligible_weights"]
                              + sum(row["estimated_packed_bytes"] for row in projection.values()))
        results[name] = dict(
            paired_vs_fp16=paired_quality(baseline, record),
            timing=timing(record),
            estimated_packed_model_bytes=packed,
            packed_fraction_of_fp16=packed / baseline["parameters"]["dense_parameter_bytes"],
            calibration_seconds=record.get("calibration_seconds"),
            compression_seconds=record.get("pruning_seconds", record.get("quantization_seconds")),
            allocation=allocation,
        )

    for percent in (50, 75):
        chosen, control = f"nonuniform_{percent}", f"wanda_{percent}"
        a, b = by_name[chosen], by_name[control]
        assert a["masked_weights"] == b["masked_weights"]
        results[chosen].update(comparator=control, paired_vs_comparator=paired_quality(b, a),
                               comparator_delta_ci95=paired_intervals(b, a))
    for bits in (3, 4):
        chosen, control = f"folded_mixed_int{bits}_budget", f"folded_int{bits}"
        a, b = by_name[chosen], by_name[control]
        assert a["estimated_packed_model_bytes"] <= b["estimated_packed_model_bytes"]
        assert a["uniform_budget_bytes"] == b["estimated_packed_model_bytes"]
        results[chosen].update(comparator=control, paired_vs_comparator=paired_quality(b, a),
                               comparator_delta_ci95=paired_intervals(b, a))

    summary = dict(
        baseline=dict(quality=paired_quality(baseline, baseline), timing=timing(baseline),
                      parameter_bytes=baseline["parameters"]["dense_parameter_bytes"]),
        bootstrap=dict(seed=6304, resamples=2000, nll_unit="whole window",
                       hellaswag_unit="example"),
        awq_storage="Q2/Q3 use the folded runs, with explicit scales counted where folding was unavailable.",
        conditions=results,
    )
    Path("task1-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
