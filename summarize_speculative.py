"""Summarize saved PA4 speculative evaluation without loading models."""
import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile


def summarize_pairs(groups):
    ratios, totals = [], []
    for pairs in groups:
        prompt_ratios = []
        for pair in pairs:
            target, spec = pair["target"], pair["speculative"]
            prompt_ratios.append(len(spec["token_ids"]) * target["total_ms"] /
                                 (len(target["token_ids"]) * spec["total_ms"]))
        ratios.append(statistics.median(prompt_ratios))
        totals.append(tuple(sum(r[field] for p in pairs for r in p["speculative"]["rounds"])
                            for field in ("accepted", "examined", "proposed")))
    rng = random.Random(6304)
    speed_draws, accept_draws = [], []
    for _ in range(2000):
        sample = rng.choices(range(len(groups)), k=len(groups))
        speed_draws.append(statistics.median(ratios[i] for i in sample))
        accept_draws.append(sum(totals[i][0] for i in sample) / sum(totals[i][1] for i in sample))
    def interval(draws):
        ordered = sorted(draws)
        return [ordered[49], ordered[1949]]
    accepted, examined, proposed = map(sum, zip(*totals))
    rounds = [r for pairs in groups for p in pairs for r in p["speculative"]["rounds"]]
    return dict(prompts=len(groups), median_throughput_ratio=statistics.median(ratios),
                throughput_ratio_ci95=interval(speed_draws),
                accepted_per_examined=accepted / examined, accepted_per_proposed=accepted / proposed,
                accepted_per_examined_ci95=interval(accept_draws),
                mean_round_ms={field: statistics.mean(r[field] for r in rounds) for field in
                               ("draft_ms", "verification_ms", "sampling_ms", "cache_ms")})


def cost_model(samples, gamma, alpha):
    by_length = defaultdict(dict)
    for sample in samples:
        by_length[sample["prompt_length"]][sample["model"], sample["width"]] = statistics.median(
            sample["samples_ms"]
        )
    results = {}
    for length, costs in sorted(by_length.items()):
        target_ms = costs["target", 1]
        c = costs["draft", 1] / target_ms
        verification = costs["target", gamma] / target_ms
        cost = gamma * c + verification

        def expected_tokens(acceptance):
            return sum(acceptance ** step for step in range(gamma + 1))

        threshold = None
        if cost < gamma + 1:
            lo, hi = 0., 1.
            for _ in range(48):
                mid = (lo + hi) / 2
                if expected_tokens(mid) > cost:
                    hi = mid
                else:
                    lo = mid
            threshold = hi if cost >= 1 else 0.
        results[length] = dict(
            draft_to_target_step_ratio=c, verification_to_target_step_ratio=verification,
            empirical_acceptance=alpha, projected_speedup=expected_tokens(alpha) / cost,
            break_even_acceptance_threshold=threshold,
        )
    return results


def deployment_cost(rows, gamma, upfront_seconds):
    pairs = [p for row in rows for p in row["paired_runs"]
             if p["gamma"] == gamma and not p["fixed_64_ignores_eos"]]
    mean_request_ms = statistics.mean(p["target"]["total_ms"] - p["speculative"]["total_ms"]
                                      for p in pairs)
    mean_matched_ms = statistics.mean(
        p["target"]["total_ms"] * len(p["speculative"]["token_ids"]) /
        len(p["target"]["token_ids"]) - p["speculative"]["total_ms"] for p in pairs
    )
    return dict(
        upfront_seconds=upfront_seconds, mean_raw_savings_ms=mean_request_ms,
        mean_length_adjusted_savings_ms=mean_matched_ms,
        break_even_requests=(math.ceil(upfront_seconds * 1000 / mean_matched_ms)
                             if mean_matched_ms > 0 else None),
    )


def bank_consistency(rows):
    """Pair the two target-law banks by prompt and response position."""
    result = {}
    for domain in ("all", "math", "familiar", "shifted"):
        for span, lo, hi in (("short", 1, 64), ("long", 65, 192)):
            by_prompt = []
            for row in rows:
                if domain not in ("all", row["domain"]):
                    continue
                banks = row["prefix_metrics"]
                teacher = {m["position"]: m for m in banks["teacher"] if lo <= m["position"] <= hi}
                speculative = {m["position"]: m for m in banks["exact_spec_original_gamma4"]
                               if lo <= m["position"] <= hi}
                positions = teacher.keys() & speculative.keys()
                if positions:
                    by_prompt.append((
                        len(positions),
                        sum(speculative[p]["expected_acceptance"] - teacher[p]["expected_acceptance"]
                            for p in positions),
                        sum(speculative[p]["forward_kl"] - teacher[p]["forward_kl"] for p in positions),
                    ))
            if not by_prompt:
                continue
            count, acceptance, kl = map(sum, zip(*by_prompt))
            rng = random.Random(6304)
            estimates = [[], []]
            for _ in range(2000):
                n, a, k = map(sum, zip(*rng.choices(by_prompt, k=len(by_prompt))))
                estimates[0].append(a / n)
                estimates[1].append(k / n)
            for values in estimates:
                values.sort()
            result[f"{domain}/{span}"] = dict(
                prompts=len(by_prompt), paired_positions=count,
                speculative_minus_teacher_expected_acceptance=acceptance / count,
                expected_acceptance_ci95=[estimates[0][49], estimates[0][1949]],
                speculative_minus_teacher_forward_kl=kl / count,
                forward_kl_ci95=[estimates[1][49], estimates[1][1949]],
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--training-archive", type=Path)
    parser.add_argument("--course-bank-archive", type=Path)
    args = parser.parse_args()
    drafts = defaultdict(list)
    checks = {}
    training = {}
    if args.training_archive:
        with ZipFile(args.training_archive) as archive:
            for name in archive.namelist():
                if name.endswith("/training.json"):
                    training[name.split("/")[-2]] = json.loads(archive.read(name))
    with ZipFile(args.archive) as archive:
        for name in sorted(archive.namelist()):
            if "/per_prompt/" in name and name.endswith(".json"):
                drafts[name.split("/")[-2]].append(json.loads(archive.read(name)))
            if "/model_checks/" in name and name.endswith(".json"):
                checks[name.rsplit("/", 1)[-1]] = json.loads(archive.read(name))
    if args.course_bank_archive:
        course = {}
        with ZipFile(args.course_bank_archive) as archive:
            for name in archive.namelist():
                if "/course_alignment/" in name and name.endswith(".json"):
                    row = json.loads(archive.read(name))
                    course[name.split("/")[-2], row["id"]] = row["prefix_metrics"]
        if len(course) != 600:
            raise ValueError("Expected course scores for all 600 draft/prompt pairs")
        for draft, rows in drafts.items():
            for row in rows:
                row["prefix_metrics"].update(course[draft, row["id"]])
    if len(drafts) != 5:
        raise ValueError("Expected five draft configurations")
    summary = {}
    for draft, rows in drafts.items():
        if len(rows) != 120 or len({row["id"] for row in rows}) != 120:
            raise ValueError(f"Incomplete prompt set for {draft}")
        speed = {}
        banks = defaultdict(list)
        standalone = defaultdict(list)
        for row in rows:
            standalone[row["domain"]].append(row["standalone"])
            for bank, metrics in row["prefix_metrics"].items():
                for metric in metrics:
                    span = "1-64" if metric["position"] <= 64 else "65-192"
                    banks[f"{bank}/{row['domain']}/{span}"].append(metric)
        for gamma in (2, 4):
            for domain in ("all", "math", "familiar", "shifted"):
                groups = [[p for p in r["paired_runs"] if p["gamma"] == gamma
                           and not p["fixed_64_ignores_eos"]] for r in rows
                          if domain in ("all", r["domain"])]
                if any(len(p) != 2 for p in groups):
                    raise ValueError(f"Incomplete timing pairs for {draft}, gamma {gamma}")
                speed[f"gamma{gamma}/{domain}"] = summarize_pairs(groups)
        projection = {}
        deployment = {}
        cost_samples = checks[f"{draft}-cost.json"]["samples"]
        record = training.get(draft)
        for gamma in (2, 4):
            key = f"gamma{gamma}/all"
            projection[key] = cost_model(cost_samples, gamma, speed[key]["accepted_per_examined"])
            if record:
                upfront = (sum(record["times"].values()) + record["shared_teacher_rollout_seconds"]
                           + checks[f"{draft}.json"]["merge_ms"] / 1000)
                deployment[key] = deployment_cost(rows, gamma, upfront)
        summary[draft] = dict(
            performance=speed, projected_speedup_by_prompt_length=projection,
            deployment_vs_target_only=deployment, target_law_bank_diagnostic=bank_consistency(rows),
            standalone={domain: dict(prompts=len(items), accuracy=statistics.mean(x["correct"] for x in items),
                                     truncation_rate=statistics.mean(x["truncated"] for x in items),
                                     format_compliance=statistics.mean(x["format_compliant"] for x in items)
                                     if domain != "math" else None)
                        for domain, items in standalone.items()},
            alignment={key: dict(prefixes=len(items), **{field: statistics.mean(x[field] for x in items)
                            for field in ("forward_kl", "reverse_kl", "tv", "expected_acceptance", "greedy_agreement")})
                       for key, items in banks.items()},
        )
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
