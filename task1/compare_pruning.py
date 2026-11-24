"""Compare saved pruning results to the matched FP16 baseline."""
import argparse
import json
import math
from pathlib import Path


def read_result(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_quality(baseline, result):
    metrics = {}
    for dataset in ("wikitext", "c4"):
        base = baseline[dataset]
        current = result[dataset]
        keys = ("source_row", "token_offset", "scored_tokens")
        def windows(data):
            return [tuple(row[key] for key in keys) for row in data["windows"]]
        if windows(base) != windows(current):
            raise ValueError(f"Unmatched {dataset} windows")
        change = current["perplexity"] - base["perplexity"]
        if not math.isfinite(change):
            raise ValueError(f"Non-finite {dataset} perplexity")
        metrics[dataset] = dict(ppl=current["perplexity"], delta_ppl=change)

    base_examples = baseline["hellaswag"]["examples"]
    pairs = list(zip(base_examples[:len(result["hellaswag"]["examples"])],
                     result["hellaswag"]["examples"], strict=True))
    if len(pairs) not in (512, 2000):
        raise ValueError("Expected 512 or 2,000 HellaSwag examples")
    if any((a["id"], a["order"], a["label"]) != (b["id"], b["order"], b["label"])
           for a, b in pairs):
        raise ValueError("Unmatched HellaSwag examples")
    metrics["hellaswag"] = dict(examples=len(pairs))
    for field in ("correct", "correct_norm"):
        rate = sum(b[field] for _, b in pairs) / len(pairs)
        change = sum(int(b[field]) - int(a[field]) for a, b in pairs) / len(pairs)
        metrics["hellaswag"][field] = dict(accuracy=rate, delta_percentage_points=100 * change)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    baseline = read_result(args.baseline)
    profiles = ("model", "revision", "source_revisions", "attention_backend",
                "dtype", "tokenizer", "hellaswag_format")
    for path in sorted(args.results_dir.glob("*.json")):
        result = read_result(path)
        if path.name.endswith(".failure.json"):
            print(json.dumps(result))
            continue
        mismatch = [key for key in profiles if baseline[key] != result[key]]
        if mismatch:
            raise ValueError(f"{path.name}: mismatched fields {mismatch}")
        quality = paired_quality(baseline, result)
        print(json.dumps(dict(condition=result["condition"], **quality), allow_nan=False))


if __name__ == "__main__":
    main()

