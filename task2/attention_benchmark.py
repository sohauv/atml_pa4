"""Task 2A causal attention correctness and CUDA benchmark."""
import argparse
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention


LENGTHS = (256, 512, 1024, 2048, 4096, 8192)


def dense(q, k, v, valid=None):
    positions = torch.arange(q.size(-2), device=q.device)
    keep = positions[:, None] >= positions
    if valid is not None:
        keep = keep & valid
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.size(-1))
    scores = scores.masked_fill(~keep, -torch.inf)
    empty = ~keep.any(dim=-1, keepdim=True)
    probabilities = scores.masked_fill(empty, 0).softmax(dim=-1).masked_fill(empty, 0)
    return (probabilities @ v.float()).to(q.dtype)


def tiled(q, k, v, tile=256, valid=None):
    length = q.size(-2)
    positions = torch.arange(length, device=q.device)
    q, k, v = q.float(), k.float(), v.float()
    outputs = []
    for i in range(0, length, tile):
        stop = min(i + tile, length)
        block = q[..., i:stop, :]
        maximum = torch.full_like(block[..., :1], -torch.inf)
        normalizer = torch.zeros_like(maximum)
        accumulator = torch.zeros_like(block)
        for j in range(0, stop, tile):
            end = min(j + tile, stop)
            keep = positions[i:stop, None] >= positions[j:end]
            if valid is not None:
                keep = keep & valid[..., j:end]
            scores = (block @ k[..., j:end, :].transpose(-1, -2)) / math.sqrt(q.size(-1))
            scores = scores.masked_fill(~keep, -torch.inf)
            updated = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
            safe = updated.masked_fill(torch.isneginf(updated), 0)
            old_factor = (maximum - safe).exp()
            weights = (scores - safe).exp()
            accumulator = old_factor * accumulator + weights @ v[..., j:end, :]
            normalizer = old_factor * normalizer + weights.sum(dim=-1, keepdim=True)
            maximum = updated
        outputs.append(accumulator / normalizer.masked_fill(normalizer == 0, 1))
    return torch.cat(outputs, dim=-2)


def sdpa(q, k, v, backend):
    with sdpa_kernel(backend):
        return scaled_dot_product_attention(q, k, v, is_causal=True)


def evaluate(operator, inputs, upstream):
    leaves = tuple(x.detach().requires_grad_() for x in inputs)
    out = operator(*leaves)
    gradients = torch.autograd.grad(out, leaves, upstream.to(out.dtype))
    return tuple(x.detach().float() for x in (out, *gradients))


def error(actual, reference):
    delta = actual - reference
    return dict(max_abs=delta.abs().max().item(),
                relative_l2=(delta.norm() / reference.norm().clamp_min(1e-12)).item())


def benchmark(operator, inputs, upstream, backward):
    leaves = tuple(x.detach().requires_grad_(backward) for x in inputs)
    samples = []
    torch.cuda.empty_cache()
    for repeat in range(13):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        out = operator(*leaves)
        if backward:
            gradients = torch.autograd.grad(out, leaves, upstream)
        torch.cuda.synchronize()
        elapsed = (perf_counter() - start) * 1000
        if repeat >= 3:
            samples.append(dict(ms=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        del out
        if backward:
            del gradients
    return dict(samples=samples, median_ms=median(row["ms"] for row in samples))


def edge_checks(device):
    inputs = [torch.randn(2, 4, 17, 64, device=device) for _ in range(3)]
    inputs[0] *= 1000
    inputs[1] *= 1000
    upstream = torch.randn_like(inputs[0])
    valid = torch.arange(17, device=device) >= 3
    reference = evaluate(lambda q, k, v: dense(q, k, v, valid), inputs, upstream)
    cases = []
    for tile in (4, 9, 32):
        actual = evaluate(lambda q, k, v: tiled(q, k, v, tile, valid), inputs, upstream)
        if not all(torch.isfinite(x).all() for x in actual):
            raise FloatingPointError("Non-finite extreme-score or masked-row output/gradient")
        if actual[0][..., :3, :].count_nonzero():
            raise ValueError("All-masked rows must return zero")
        cases.append(dict(tile=tile, errors=[error(x, y) for x, y in zip(actual, reference)]))
    return cases


@torch.inference_mode()
def check_causality():
    generator = torch.Generator(device="cuda").manual_seed(6304)
    records = []
    for dtype in (torch.float32, torch.float16):
        q, k, v = [torch.randn(2, 4, 137, 64, device="cuda", dtype=dtype,
                               generator=generator) for _ in range(3)]
        cut = 65
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[..., cut:, :] *= -3
        changed_v[..., cut:, :] += 7
        for name, operator in (("dense", dense), ("tiled_32", lambda q, k, v: tiled(q, k, v, 32)),
                               ("tiled_128", lambda q, k, v: tiled(q, k, v, 128))):
            original = operator(q, k, v)
            changed = operator(q, changed_k, changed_v)
            earlier = (original[..., :cut, :] - changed[..., :cut, :]).abs().max().item()
            later = (original[..., cut:, :] - changed[..., cut:, :]).abs().max().item()
            if not math.isfinite(earlier) or not math.isfinite(later) or earlier > 1e-5 or later <= 1e-3:
                raise AssertionError(f"Causality check failed for {name} {dtype}: {earlier}, {later}")
            records.append(dict(operator=name, dtype=str(dtype), first_perturbed_position=cut,
                                prefix_max_abs_change=earlier, suffix_max_abs_change=later))
    return dict(seed=6304, gpu=torch.cuda.get_device_name(), checks=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--causality-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required.")
    if args.causality_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(check_causality(), indent=2, allow_nan=False), encoding="utf-8")
        return
    torch.manual_seed(6304)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = dict(seed=6304, gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  warmups=3, repetitions=10, reference_max_length=2048,
                  errors_order=["output", "dq", "dk", "dv"],
                  edge_checks=edge_checks("cuda"), runs=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for length in LENGTHS:
        fp32 = tuple(torch.randn(2, 4, length, 64, device="cuda") for _ in range(3))
        upstream = torch.randn_like(fp32[0])
        for dtype in (torch.float32, torch.float16):
            inputs = tuple(x.to(dtype) for x in fp32)
            rounded_fp32 = tuple(x.float() for x in inputs)
            grad = upstream.to(dtype)
            reference = evaluate(dense, rounded_fp32, grad) if length <= 2048 else None
            operators = {"dense": dense}
            for size in (128, 256):
                operators[f"tiled_{size}"] = (
                    lambda q, k, v, size=size: tiled(q, k, v, size).to(q.dtype)
                )
            for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
                operators[f"sdpa_{backend.name.lower()}"] = (
                    lambda q, k, v, backend=backend: sdpa(q, k, v, backend)
                )
            for name, operator in operators.items():
                row = dict(length=length, dtype=str(dtype), operator=name)
                print(length, dtype, name, flush=True)
                if reference is not None:
                    try:
                        actual = evaluate(operator, inputs, grad)
                        row["errors"] = [error(x, y) for x, y in zip(actual, reference)]
                        if not all(math.isfinite(v) for e in row["errors"] for v in e.values()):
                            raise FloatingPointError("Non-finite error metric")
                    except (RuntimeError, FloatingPointError) as exc:
                        row.pop("errors", None)
                        row["correctness_failure"] = str(exc)
                for stage, backward in (("forward", False), ("forward_backward", True)):
                    try:
                        row[stage] = benchmark(operator, inputs, grad, backward)
                    except RuntimeError as exc:
                        row[stage] = dict(failure=str(exc))
                result["runs"].append(row)
                args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()

