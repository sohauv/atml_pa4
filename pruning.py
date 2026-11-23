"""Matched Task 1 magnitude and Wanda pruning experiments."""
import argparse
import heapq
import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline import (MODEL_ID, PROJECTIONS, benchmark_cuda, hellaswag,
                      parameter_inventory, perplexity, read_jsonl)


PRUNING_CONDITIONS = {
    f"{method}_{percent}": (method, percent / 100)
    for method in ("magnitude", "wanda")
    for percent in (25, 50, 75)
}
PRUNING_CONDITIONS["wanda_2of4"] = ("wanda_2of4", 0.5)
PRUNING_CONDITIONS.update(nonuniform_50=("nonuniform", .5), nonuniform_75=("nonuniform", .75))
QUANTIZATION_CONDITIONS = {f"uniform_int{bits}": ("uniform_quantization", bits)
                           for bits in (8, 4, 3)}
QUANTIZATION_CONDITIONS.update(
    {f"activation_aware_int{bits}": ("activation_aware_quantization", bits)
     for bits in (8, 4, 3)}
)
QUANTIZATION_CONDITIONS.update(
    {f"mixed_int{bits}_budget": ("mixed_quantization", bits) for bits in (4, 3)}
)
QUANTIZATION_CONDITIONS.update(
    {f"folded_int{bits}": ("folded_quantization", bits) for bits in (8, 4, 3)}
)
QUANTIZATION_CONDITIONS.update(
    {f"folded_mixed_int{bits}_budget": ("folded_mixed_quantization", bits) for bits in (4, 3)}
)
CONDITIONS = PRUNING_CONDITIONS | QUANTIZATION_CONDITIONS
DIAGNOSTIC_POSITIONS = (0, 128, 256, 384)


def projections(model):
    for index, block in enumerate(model.model.layers):
        for name in PROJECTIONS:
            parent = block.self_attn if name in PROJECTIONS[:4] else block.mlp
            yield f"{index}.{name}", getattr(parent, name)


def load_model(revision):
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, dtype=torch.float16, attn_implementation="sdpa"
    ).to("cuda").eval()


def collect_calibration(model, rows):
    energy, samples, hooks = {}, {}, []
    for name, module in projections(model):
        energy[name] = torch.zeros(module.in_features, device=model.device)
        samples[name] = []

        def capture(_, inputs, key=name):
            x = inputs[0][0]
            energy[key].add_(x.float().square().sum(dim=0))
            samples[key].append(x[list(DIAGNOSTIC_POSITIONS)].cpu())

        hooks.append(module.register_forward_pre_hook(capture))
    try:
        for row in rows:
            tokens = torch.tensor([row["token_ids"]], device=model.device)
            model(tokens, use_cache=False, logits_to_keep=1)
    finally:
        for hook in hooks:
            hook.remove()
    norms = {name: value.sqrt().cpu() for name, value in energy.items()}
    for name, value in norms.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite calibration activation at {name}")
    return norms, {name: torch.cat(parts) for name, parts in samples.items()}


def allocate_budget(shapes, costs, sparsity):
    target = sum(rows * round(cols * sparsity) for rows, cols in shapes.values())
    masks = {name: [(cols + 9) // 10, 0] for name, (_, cols) in shapes.items()}
    upper = {name: 85 * cols // 100 for name, (_, cols) in shapes.items()}
    remaining = target - sum(shapes[name][0] * counts[0] for name, counts in masks.items())
    queue = [(costs[name][counts[0]], name) for name, counts in masks.items()
             if counts[0] < upper[name]]
    heapq.heapify(queue)
    while remaining > 0 and queue:
        _, name = heapq.heappop(queue)
        rows, _ = shapes[name]
        if remaining < rows:
            masks[name][1] = remaining
            remaining = 0
            break
        remaining -= rows
        masks[name][0] += 1
        rank = masks[name][0]
        if rank < upper[name]:
            heapq.heappush(queue, (costs[name][rank], name))
    if remaining:
        raise ValueError("Cannot meet target mask budget within projection bounds")
    return masks


def sensitivity(model, norms, samples):
    shapes, costs = {}, {}
    for name, module in projections(model):
        w = module.weight.float()
        x = samples[name].to(device=w.device, dtype=w.dtype)
        reference = (x @ w.T).square().sum().item()
        if not math.isfinite(reference) or reference <= 0:
            raise FloatingPointError(f"Invalid output energy at {name}")
        ranked = (w.abs() * norms[name].to(w.device)).sort(dim=1).values
        costs[name] = (ranked.square().mean(dim=0) / reference).tolist()
        shapes[name] = tuple(w.shape)
    return shapes, costs


def prune_projection(weight, norm, method, sparsity, allocation=None):
    scores = weight.float().abs()
    if method != "magnitude":
        scores.mul_(norm)
    quantile_levels = torch.tensor([0., .25, .5, .75, 1.], device=weight.device)
    score_quantiles = torch.quantile(scores.flatten()[::257].contiguous(), quantile_levels)
    if method == "wanda_2of4":
        if weight.shape[1] % 4:
            raise ValueError("2:4 pruning requires a multiple of four columns")
        grouped = scores.reshape(weight.shape[0], -1, 4)
        indices = grouped.argsort(dim=-1, stable=True)[..., :2]
        mask = torch.zeros_like(grouped, dtype=torch.bool).scatter_(-1, indices, True)
        mask = mask.reshape_as(weight)
    else:
        count, extra = allocation if allocation is not None else (round(sparsity * weight.shape[1]), 0)
        order = scores.argsort(dim=-1, stable=True)
        mask = torch.zeros_like(weight, dtype=torch.bool).scatter_(-1, order[:, :count], True)
        if extra:
            next_columns = order[:, count:count + 1]
            next_scores = scores.gather(1, next_columns).flatten()
            rows = next_scores.argsort(stable=True)[:extra]
            mask[rows, next_columns[rows, 0]] = True
    return mask, score_quantiles.tolist()


def apply_pruning(model, norms, samples, method, sparsity, allocations):
    records = {}
    for name, module in projections(model):
        weight = module.weight
        mask, score_quantiles = prune_projection(weight, norms[name].to(weight.device),
                                                 method, sparsity, allocations.get(name))
        x = samples[name].to(device=weight.device, dtype=torch.float32)
        original = weight.float()
        removed = original.masked_fill(~mask, 0)
        residual_energy = (x @ removed.T).square().sum().item()
        output_energy = (x @ original.T).square().sum().item()
        if not all(map(math.isfinite, (residual_energy, output_energy))):
            raise FloatingPointError(f"Non-finite reconstruction error at {name}")
        weight.masked_fill_(mask, 0)
        nonzeros = torch.count_nonzero(weight).item()
        records[name] = dict(
            weights=weight.numel(), masked=mask.sum().item(), nonzeros=nonzeros,
            mask16_bytes=(weight.numel() + 7) // 8 + 2 * nonzeros,
            score_quantiles_approx=score_quantiles, squared_error_sum=residual_energy,
            fp32_reference_energy=output_energy,
            relative_squared_error=residual_energy / output_energy if output_energy else None,
        )
    return records


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")
    os.replace(temporary, path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(PRUNING_CONDITIONS))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required.")
    data = args.course_data / "data"
    lock = json.loads((data / "local_source_revisions.json").read_text(encoding="utf-8"))
    revision = lock["revisions"]["model_base"]
    paths = data / "public_locked"
    calibration = read_jsonl(paths / "c4_calibration_locked.jsonl.gz")
    if len(calibration) != 128 or any(len(row["token_ids"]) != 512 for row in calibration):
        raise ValueError("Expected 128 fixed 512-token C4 calibration windows")
    wiki = read_jsonl(paths / "wikitext2_test_windows_locked.jsonl.gz")
    c4 = read_jsonl(paths / "c4_validation_windows_locked.jsonl.gz")
    swag = read_jsonl(paths / "hellaswag_val_locked.jsonl.gz")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    torch.manual_seed(6304)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(revision)
    start = perf_counter()
    norms, samples = collect_calibration(model, calibration)
    calibration_seconds = perf_counter() - start
    inventory = parameter_inventory(model)
    allocations = {}
    if any(CONDITIONS[key][0] == "nonuniform" for key in args.conditions):
        start = perf_counter()
        shapes, costs = sensitivity(model, norms, samples)
        allocations = {level: allocate_budget(shapes, costs, level) for level in (.5, .75)}
        write_json(args.output_dir / "allocations.json", dict(
            method="mean ranked squared Wanda score divided by sampled output energy",
            revision=revision, shapes=shapes, marginal_costs=costs,
            row_masks_and_extra=allocations, allocation_seconds=perf_counter() - start))
    mixed = {}
    folded_profiles = {}
    if any(CONDITIONS[key][0] == "mixed_quantization" for key in args.conditions):
        from quantization import allocate_precision, profile_precision

        start = perf_counter()
        profiles = profile_precision(projections(model), samples)
        mixed = {bits: allocate_precision(profiles, bits) for bits in (4, 3)}
        write_json(args.output_dir / "precision_allocations.json", dict(
            rule="greedy reduction in relative calibration output error per added byte",
            profiles=profiles, allocations=mixed, profiling_seconds=perf_counter() - start,
            revision=revision, source_revisions=lock,
        ))
    if any(CONDITIONS[key][0] == "folded_mixed_quantization" for key in args.conditions):
        from quantization import profile_precision

        profile_start = perf_counter()
        folded_profiles = profile_precision(projections(model), samples)
        for name, table in folded_profiles.items():
            if name.endswith(".down_proj"):
                width = samples[name].shape[-1]
                for row in table.values():
                    row["estimated_packed_bytes"] -= 2 * width
        write_json(args.output_dir / "folded_precision_profiles.json", dict(
            profiles=folded_profiles, seconds=perf_counter() - profile_start,
            sensitivity="unfolded calibration reconstruction error",
            costs="upper bound with down_proj folded and shared norm folds omitted",
        ))
    del model
    torch.cuda.empty_cache()
    metadata = dict(
        model=MODEL_ID, revision=revision, source_revisions=lock, seed=6304,
        dtype="float16", attention_backend="sdpa", tokenizer=type(tokenizer).__name__,
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        transformers_version=transformers.__version__, cuda_version=torch.version.cuda,
        hellaswag_format="lm-evaluation-harness v0.4.9 zero-shot",
        calibration_seconds=calibration_seconds, diagnostic_token_offsets=DIAGNOSTIC_POSITIONS,
        calibration_windows=calibration, parameters=inventory, execution="dense FP16",
    )
    failures = []
    for condition in args.conditions:
        method, sparsity = CONDITIONS[condition]
        destination = args.output_dir / f"{condition}.json"
        failure = args.output_dir / f"{condition}.failure.json"
        if destination.exists():
            prior = json.loads(destination.read_text(encoding="utf-8"))
            if any(prior[key] != metadata[key] for key in
                   ("revision", "source_revisions", "gpu", "attention_backend",
                    "torch_version", "transformers_version")):
                raise ValueError(f"Existing result has incompatible provenance: {destination}")
            print(f"Already completed: {condition}", flush=True)
            continue
        print(f"Starting {condition}", flush=True)
        model = None
        try:
            model = load_model(revision)
            start = perf_counter()
            if condition in QUANTIZATION_CONDITIONS:
                from quantization import allocate_precision, apply_folded_quantization, apply_quantization

                bits = sparsity
                precision = mixed[bits][0] if method == "mixed_quantization" else bits
                folded_mixed = method == "folded_mixed_quantization"
                excluded = inventory["dense_parameter_bytes"] - 2 * inventory["eligible_weights"]
                if folded_mixed:
                    reference_file = args.output_dir / f"folded_int{bits}.json"
                    if not reference_file.exists():
                        raise ValueError(f"Run folded_int{bits} before {condition}")
                    reference_bytes = json.loads(reference_file.read_text())["estimated_packed_model_bytes"]
                    precision, _, upper_bound = allocate_precision(
                        folded_profiles, bits, budget=reference_bytes - excluded
                    )
                if method.startswith("folded"):
                    layers = apply_folded_quantization(model, samples, precision)
                else:
                    layers = apply_quantization(
                        projections(model), samples, precision,
                        activation_aware=method != "uniform_quantization",
                    )
                compression_seconds = perf_counter() - start
                packed = inventory["dense_parameter_bytes"] - 2 * inventory["eligible_weights"]
                packed += sum(row["estimated_packed_bytes"] for row in layers.values())
                compression = dict(
                    bits=bits, nominal_bits_per_eligible_weight=bits,
                    estimated_packed_model_bytes=packed,
                    quantization_seconds=compression_seconds,
                )
                if method == "mixed_quantization":
                    allocation, eligible_budget, expected = mixed[bits]
                    excluded_bytes = inventory["dense_parameter_bytes"] - 2 * inventory["eligible_weights"]
                    if packed != expected + excluded_bytes or packed > eligible_budget + excluded_bytes:
                        raise ValueError("Mixed-precision model failed packed-byte matching")
                    compression.update(
                        bit_assignment=allocation, uniform_budget_bits=bits,
                        uniform_budget_bytes=eligible_budget + excluded_bytes,
                        nominal_bits_per_eligible_weight=sum(
                            row["bits"] * row["weights"] for row in layers.values()
                        ) / inventory["eligible_weights"],
                    )
                    del compression["bits"]
                if folded_mixed:
                    if packed > upper_bound + excluded or packed > reference_bytes:
                        raise ValueError("Folded mixed-precision budget exceeded")
                    compression.update(
                        bit_assignment=precision, uniform_budget_bits=bits,
                        uniform_budget_bytes=reference_bytes, allocation_upper_bound_bytes=upper_bound + excluded,
                        nominal_bits_per_eligible_weight=sum(
                            row["bits"] * row["weights"] for row in layers.values()
                        ) / inventory["eligible_weights"],
                    )
                    del compression["bits"]
                swag_limit = 512 if bits == 8 else 2000
            else:
                compression, swag_limit, layers = pruning_condition(
                    model, norms, samples, method, sparsity, allocations, inventory, start
                )
            result = dict(
                **metadata, **compression, condition=condition, method=method,
                projections=layers, wikitext=perplexity(model, wiki),
                c4=perplexity(model, c4),
                hellaswag=hellaswag(model, tokenizer, swag, limit=swag_limit),
                benchmark=benchmark_cuda(model, wiki[0]["token_ids"][0]),
            )
            write_json(destination, result)
            failure.unlink(missing_ok=True)
            print(destination, flush=True)
        except (FloatingPointError, torch.cuda.OutOfMemoryError) as error:
            write_json(failure, dict(condition=condition, error=str(error)))
            failures.append(condition)
            print(f"{condition} failed: {error}", flush=True)
        finally:
            del model
            torch.cuda.empty_cache()
    if failures:
        raise SystemExit(f"Failed conditions: {', '.join(failures)}")


def pruning_condition(model, norms, samples, method, sparsity, allocations, inventory, start):
    allocation = allocations[sparsity] if method == "nonuniform" else {}
    layers = apply_pruning(model, norms, samples, method, sparsity, allocation)
    if allocation:
        target = sum(rows * round(sparsity * cols) for rows, cols in (module.weight.shape for _, module in projections(model)))
        assert sum(row["masked"] for row in layers.values()) == target
        assert all(.10 <= row["masked"] / row["weights"] <= .85 for row in layers.values())
    pruning_seconds = perf_counter() - start
    eligible = inventory["eligible_weights"]
    nonzeros = sum(row["nonzeros"] for row in layers.values())
    packed = inventory["dense_parameter_bytes"] - 2 * eligible + sum(
        row["mask16_bytes"] for row in layers.values()
    )
    compression = dict(
        target_sparsity=sparsity, masked_weights=sum(row["masked"] for row in layers.values()),
        eligible_nonzeros=nonzeros, effective_nonzero_fraction=nonzeros / eligible,
        nominal_bits_per_eligible_weight=1 + 16 * nonzeros / eligible,
        estimated_mask16_model_bytes=packed, pruning_seconds=pruning_seconds,
    )
    return compression, (2000 if sparsity == .5 else 512), layers


if __name__ == "__main__":
    main()
