"""Asymmetric groupwise post-training weight quantization."""
import math

import torch
import torch.nn.functional as F


GROUP_SIZE = 128


def quantize(weight, bits):
    """Use FP16 scales, uint8 zero points, and round-to-nearest-even codes."""
    rows, columns = weight.shape
    padding = (-columns) % GROUP_SIZE
    groups = F.pad(weight.float(), (0, padding)).reshape(rows, -1, GROUP_SIZE)
    # Include zero so one-sided weight groups still span their full data range.
    lower = groups.amin(dim=-1, keepdim=True).clamp_max(0)
    upper = groups.amax(dim=-1, keepdim=True).clamp_min(0)
    max_code = (1 << bits) - 1
    scale = ((upper - lower) / max_code).clamp_min(2 ** -24).half().float()
    zero = (-lower / scale).round().clamp_(0, max_code)
    codes = (groups / scale).round().add_(zero).clamp_(0, max_code)
    reconstructed = ((codes - zero) * scale).reshape(rows, -1)[:, :columns].half()
    # Each padded group is independently byte-aligned, then stores FP16 scale + uint8 zero.
    packed_bytes = rows * groups.shape[1] * ((GROUP_SIZE * bits + 7) // 8 + 3)
    return reconstructed, packed_bytes, padding


AWQ_EXPONENTS = (0., .25, .5, .75, 1.)


def activation_aware_weight(weight, x, target, bits, return_scaled=False):
    mean_abs = x.abs().mean(dim=0).clamp_min(1e-6)
    best = None
    candidates = []
    for alpha in AWQ_EXPONENTS:
        scale = mean_abs.pow(alpha)
        scale = (scale / (scale.max() * scale.min()).sqrt()).half().float()
        scaled = weight * scale
        inverse_input_scale = scale.reciprocal()
        # Verify the full-precision identity on fixed calibration activations.
        rescaled_target = (x * inverse_input_scale) @ scaled.T
        identity_error = (rescaled_target - target).square().sum().item()
        if not math.isfinite(identity_error) or identity_error > 1e-8 * target.square().sum().item():
            raise FloatingPointError("Activation rescaling failed its full-precision identity check")
        quantized, packed, padding = quantize(scaled, bits)
        reconstructed = (quantized.float() * inverse_input_scale).half()
        error = (x @ (reconstructed.float() - weight).T).square().sum().item()
        candidates.append(dict(exponent=alpha, squared_error_sum=error,
                               scaling_identity_squared_error=identity_error))
        if math.isfinite(error) and (best is None or error < best[0]):
            best = (error, quantized if return_scaled else reconstructed, packed, padding, alpha, scale)
    if best is None:
        raise FloatingPointError("No finite activation-aware scale candidate")
    error, reconstructed, packed, padding, alpha, scale = best
    return reconstructed, packed, padding, error, dict(
        exponent=alpha, scale_fp16=scale.tolist(), candidates=candidates,
        compensation_bytes=2 * weight.shape[1],
    )


MIXED_BITS = (2, 3, 4, 8)


def profile_precision(projections, samples):
    profiles = {}
    for name, module in projections:
        w = module.weight.float()
        x = samples[name].to(device=w.device, dtype=w.dtype)
        target = x @ w.T
        energy = target.square().sum().item()
        if not math.isfinite(energy) or energy <= 0:
            raise FloatingPointError(f"Invalid calibration energy at {name}")
        profiles[name] = {}
        for bits in MIXED_BITS:
            _, packed, _, error, details = activation_aware_weight(w, x, target, bits)
            profiles[name][bits] = dict(
                estimated_packed_bytes=packed + details["compensation_bytes"],
                relative_squared_error=error / energy,
                exponent=details["exponent"],
            )
    return profiles


def allocate_precision(profiles, uniform_bits, budget=None):
    if budget is None:
        budget = sum(row[uniform_bits]["estimated_packed_bytes"] for row in profiles.values())
    assignment = dict.fromkeys(profiles, min(MIXED_BITS))
    used = sum(row[min(MIXED_BITS)]["estimated_packed_bytes"] for row in profiles.values())
    if used > budget:
        raise ValueError("Minimum precision already exceeds the byte budget")
    while True:
        best = None
        for name, table in profiles.items():
            current = table[assignment[name]]
            for bits in MIXED_BITS:
                if bits <= assignment[name]:
                    continue
                candidate = table[bits]
                cost = candidate["estimated_packed_bytes"] - current["estimated_packed_bytes"]
                gain = current["relative_squared_error"] - candidate["relative_squared_error"]
                if gain > 0 and 0 < cost <= budget - used:
                    option = (gain / cost, name, bits, cost)
                    if best is None or option > best:
                        best = option
        if best is None:
            return assignment, budget, used
        _, name, bits, cost = best
        assignment[name] = bits
        used += cost


def apply_quantization(projections, samples, precision, activation_aware=False):
    records = {}
    for name, module in projections:
        bits = precision[name] if isinstance(precision, dict) else precision
        weight = module.weight
        x = samples[name].to(device=weight.device, dtype=torch.float32)
        original = weight.float()
        target = x @ original.T
        reference = target.square().sum().item()
        diagnostics = {}
        if activation_aware:
            reconstructed, packed, padding, error, diagnostics = activation_aware_weight(
                original, x, target, bits
            )
            packed += diagnostics["compensation_bytes"]
        else:
            reconstructed, packed, padding = quantize(weight, bits)
            error = (x @ (reconstructed.float() - original).T).square().sum().item()
        if not all(map(math.isfinite, (reference, error))):
            raise FloatingPointError(f"Non-finite quantization error at {name}")
        weight.copy_(reconstructed)
        records[name] = dict(
            **diagnostics, weights=weight.numel(), group_size=GROUP_SIZE, bits=bits,
            padded_columns_per_row=padding, estimated_packed_bytes=packed,
            scale_dtype="float16", zero_point_dtype="uint8",
            nonzeros=torch.count_nonzero(weight).item(),
            squared_error_sum=error, fp32_reference_energy=reference,
            relative_squared_error=error / reference if reference else None,
        )
    return records



def fold_mlp_up_down(up_weight, down_weight, scale, up_bias=None):
    """Move the inverse input scale of down_proj into the up_proj output."""
    if scale.ndim != 1 or scale.numel() != up_weight.shape[0] or scale.numel() != down_weight.shape[1]:
        raise ValueError("Scale must match the MLP intermediate width")
    scale = scale.float()
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Fold requires finite positive scales")
    return (up_weight.float() / scale[:, None],
            down_weight.float() * scale[None, :],
            None if up_bias is None else up_bias.float() / scale)

    

def apply_folded_quantization(model, samples, precision):
    """Quantize MLP pairs and fold exactly shared scales into RMSNorm."""
    records = {}
    for index, block in enumerate(model.model.layers):
        prefix = f"{index}."
        modules = {name: getattr(block.self_attn if name in ("q_proj", "k_proj", "v_proj", "o_proj")
                                 else block.mlp, name) for name in (
                                     "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")}
        weights = {name: module.weight for name, module in modules.items()}
        selected = {}
        for name in ("down_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"):
            key = prefix + name
            bits = precision[key] if isinstance(precision, dict) else precision
            x = samples[key].to(device=weights[name].device, dtype=torch.float32)
            original = weights[name].float()
            if name == "up_proj":
                original = original / down_scale[:, None]
            target = x @ original.T
            scaled, packed, padding, error, diagnostics = activation_aware_weight(
                original, x, target, bits, return_scaled=True
            )
            scale = torch.tensor(diagnostics["scale_fp16"], device=x.device, dtype=torch.float32)
            if name == "down_proj":
                down_scale = scale
            energy = target.square().sum().item()
            selected[name] = (scaled, scale)
            records[key] = dict(
                **diagnostics, weights=weights[name].numel(), group_size=GROUP_SIZE, bits=bits,
                padded_columns_per_row=padding, estimated_packed_bytes=packed + diagnostics["compensation_bytes"],
                scale_dtype="float16", zero_point_dtype="uint8",
                squared_error_sum=error, fp32_reference_energy=energy,
                relative_squared_error=error / energy if energy else None,
                error_coordinate_system="folded_up_output" if name == "up_proj" else "original_projection",
            )

        folds = {"down_proj": "up_proj_output"}
        for names, norm in ((("q_proj", "k_proj", "v_proj"), block.input_layernorm),
                            (("gate_proj", "up_proj"), block.post_attention_layernorm)):
            scale = selected[names[0]][1]
            if all(torch.equal(scale, selected[name][1]) for name in names[1:]):
                norm.weight.copy_((norm.weight.float() / scale).half())
                folds.update({name: "input_rmsnorm" for name in names})

        for name, (scaled, scale) in selected.items():
            row = records[prefix + name]
            if name in folds:
                row["estimated_packed_bytes"] -= row["compensation_bytes"]
                row["compensation_bytes"] = 0
                row["folded_into"] = folds[name]
                weight = scaled
            else:
                weight = (scaled.float() * scale.reciprocal()).half()
            modules[name].weight.copy_(weight)
            row["nonzeros"] = torch.count_nonzero(weight).item()
    return records
