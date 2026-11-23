"""PA4 Task 2B MQAR training and frozen evaluation."""
import argparse
import gzip
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F


SEEDS = (6304, 6305, 6306)
STEPS = 4500
EVALUATE = (500, 1000, 1500, 2000, 3000, 4500)
REGIMES = ("near", "far", "mixed")
KS = (4, 8, 16)
GRID = (
    [("dense", k, "mixed") for k in KS] + [("dense", 16, "near")]
    + [("window", k, "near") for k in KS] + [("window", 16, "mixed")]
    + [("linear", k, "mixed") for k in KS]
    + [("dilated", k, "mixed") for k in (4, 16)]
    + [("curriculum", 16, "near_then_mixed")]
)


def linear_attention(q, k, v, block=32):
    length = q.shape[-2]
    padding = (-length) % block
    def chunks(x):
        return F.pad(x.float(), (0, 0, 0, padding)).unflatten(-2, (-1, block))
    q = chunks(F.elu(q.float()) + 1)
    k = chunks(F.elu(k.float()) + 1)
    v = chunks(v)
    state = k.transpose(-1, -2) @ v
    key_sum = k.sum(dim=-2, keepdim=True)
    previous = F.pad(state.cumsum(dim=-3)[..., :-1, :, :], (0, 0, 0, 0, 1, 0))
    previous_keys = F.pad(key_sum.cumsum(dim=-3)[..., :-1, :, :], (0, 0, 0, 0, 1, 0))
    local = (q @ k.transpose(-1, -2)).tril()
    numerator = local @ v + q @ previous
    denominator = local.sum(dim=-1, keepdim=True) + q @ previous_keys.transpose(-1, -2)
    return (numerator / denominator.clamp_min(1e-6)).flatten(-3, -2)[..., :length, :]


def attention(q, k, v, method):
    if method == "dense":
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if method == "linear":
        return linear_attention(q, k, v).to(q.dtype)
    pos = torch.arange(q.size(-2), device=q.device)
    gap = pos[:, None] - pos
    local = (gap >= 0) & (gap < 64)
    if method == "window":
        mask = local
    elif method == "dilated":
        dilated = (gap >= 0) & (gap <= 252) & (gap % 4 == 0)
        mask = torch.stack((local, local, dilated, dilated))
    else:
        raise ValueError(f"Unknown attention method: {method}")
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(128)
        self.qkv = nn.Linear(128, 384)
        self.output = nn.Linear(128, 128)
        self.mlp_norm = nn.LayerNorm(128)
        self.mlp = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 128))

    def forward(self, x, method):
        batch, length, _ = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).reshape(batch, length, 3, 4, 32).permute(2, 0, 3, 1, 4)
        out = attention(q, k, v, method).transpose(1, 2).reshape(batch, length, 128)
        x = x + self.output(out)
        return x + self.mlp(self.mlp_norm(x))


class MQAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 128)
        angles = torch.arange(256)[:, None] / 10000 ** (torch.arange(0, 128, 2) / 128)
        self.register_buffer("positions", torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1))
        self.blocks = nn.ModuleList([Block(), Block()])
        self.norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 256)

    def forward(self, tokens, queries, method):
        x = self.embedding(tokens) + self.positions[:tokens.shape[1]]
        for block in self.blocks:
            x = block(x, method)
        x = x.gather(1, queries[..., None].expand(-1, -1, 128))
        return self.head(self.norm(x))


def tensor_batch(rows):
    fields = ("tokens", "query_positions", "answer_ids", "distances")
    return tuple(torch.tensor([row[key] for row in rows]) for key in fields)


def frozen_data(data, split, validate):
    groups = defaultdict(list)
    with gzip.open(data / f"mqar_{split}.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            validate(row)
            assert row["split"] == split
            groups[row["K"], row["regime"]].append(row)
    expected = 1024 if split == "test" else 256
    assert set(groups) == {(k, r) for k in KS for r in REGIMES}
    assert all(len(rows) == len({r["id"] for r in rows}) == expected for rows in groups.values())
    return {key: tensor_batch(rows) for key, rows in groups.items()}


@torch.inference_mode()
def evaluate(model, data, method):
    model.eval()
    reports = {}
    for (k, regime), matrices in sorted(data.items()):
        count = torch.zeros(5, device="cuda", dtype=torch.long)
        correct = torch.zeros_like(count)
        loss = torch.zeros((), device="cuda")
        for start in range(0, matrices[0].shape[0], 32):
            tokens, queries, labels, distances = (x[start:start + 32].cuda() for x in matrices)
            logits = model(tokens, queries, method)
            loss += F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum")
            matches = logits.argmax(-1) == labels
            count[0] += labels.numel()
            correct[0] += matches.sum()
            for i, (lo, hi) in enumerate(((0, 32), (32, 64), (64, 128), (128, 256)), 1):
                selected = (distances >= lo) & (distances < hi)
                count[i] += selected.sum()
                correct[i] += (matches & selected).sum()
        counts, hits = count.tolist(), correct.tolist()
        reports[f"K{k}_{regime}"] = dict(
            mean_loss=loss.item() / counts[0], random_label_accuracy=1 / 64,
            random_example_value_accuracy=1 / k,
            accuracy={label: dict(correct=c, queries=n, accuracy=c / n if n else None)
                      for label, c, n in zip(("overall", "0-31", "32-63", "64-127", "128-255"),
                                             hits, counts)},
        )
    return reports


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path, model, optimizer, step, elapsed):
    temporary = path.with_suffix(".tmp")
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    step=step, elapsed=elapsed), temporary)
    os.replace(temporary, path)


def train(folder, method, k, regime, seed, make_example, development, test, deadline):
    name = f"{method}_K{k}_{regime}_seed{seed}"
    operator = "dense" if method == "curriculum" else method
    result_path, checkpoint = folder / f"{name}.json", folder / f"{name}.pt"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("finished"):
            return True
    else:
        result = dict(method=method, K=k, training_regime=regime, seed=seed, learning_curve=[],
                      gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                      batch_size=32, total_updates=STEPS, training_precision="float32")
    if method == "curriculum":
        result.update(
            near_updates=1500, mixed_updates=3000, extra_persistent_state_bytes=0,
            development_selection="First logged update where all dense K16 near-only seeds "
                                  "reach at least 95% on the K16 mixed development set.",
            prediction="More seeds reach high mixed and far recall than with mixed-only training.",
        )
    torch.manual_seed(seed)
    model = MQAR().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0008, weight_decay=.01)
    start_step, prior_elapsed = 0, 0.
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step, prior_elapsed = saved["step"], saved["elapsed"]
        del saved
        result["learning_curve"] = [entry for entry in result["learning_curve"] if entry["step"] <= start_step]
    start = perf_counter()
    accumulated = torch.zeros((), device="cuda")
    for step in range(start_step + 1, STEPS + 1):
        lr = .0008 * (step / 200 if step <= 200 else 1 - .85 * (step - 200) / (STEPS - 200))
        for group in optimizer.param_groups:
            group["lr"] = lr
        offset = (seed - SEEDS[0]) * STEPS * 32 + (step - 1) * 32
        active_regime = regime
        if method == "curriculum":
            active_regime = "near" if step <= 1500 else "mixed"
        rows = [make_example("train", k, active_regime, offset + i) for i in range(32)]
        tokens, queries, labels, _ = (x.cuda() for x in tensor_batch(rows))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens, queries, operator)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        accumulated += loss.detach()
        if step % 100 == 0:
            average = accumulated.item() / (step - max(start_step, step - 100))
            result["learning_curve"].append(dict(step=step, train_loss=average, lr=lr))
            accumulated.zero_()
        if step in EVALUATE:
            result["learning_curve"][-1]["development"] = evaluate(model, development, operator)
            print(name, step, flush=True)
        expired = perf_counter() >= deadline
        if step % 100 == 0 or expired or step == STEPS:
            elapsed = prior_elapsed + perf_counter() - start
            save_checkpoint(checkpoint, model, optimizer, step, elapsed)
            result.update(last_step=step, elapsed_seconds=elapsed)
            save_json(result_path, result)
        if expired and step < STEPS:
            return False
    result.update(
        finished=True, test=evaluate(model, test, operator),
        trainable_parameters=sum(p.numel() for p in model.parameters()),
    )
    if method == "dense":
        result["test_only_window"] = evaluate(model, test, "window")
    save_json(result_path, result)
    return True


@torch.inference_mode()
def inference_sweep(destination):
    from statistics import median

    torch.manual_seed(6304)
    result = dict(
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        batch_size=2, heads=4, head_dim=32, precision="float32",
        warmups=3, repetitions=10,
        persistent_linear_state_bytes_per_layer=2 * 4 * (32 * 32 + 32) * 4,
        implementations=dict(window="dense boolean mask", dilated="dense per-head boolean mask",
                             linear="block prefix states and within-block pair products"),
        runs=[],
    )
    for length in (256, 1024, 4096, 8192):
        inputs = tuple(torch.randn(2, 4, length, 32, device="cuda") for _ in range(3))
        dense_pairs = length * (length + 1) // 2
        local_pairs = sum(min(p + 1, 64) for p in range(length))
        dilated_pairs = sum(min(p // 4 + 1, 64) for p in range(length))
        pairs = dict(dense=8 * dense_pairs, window=8 * local_pairs,
                     dilated=4 * (local_pairs + dilated_pairs), linear=None)
        for method in ("dense", "window", "linear", "dilated"):
            row = dict(length=length, method=method, explicit_theoretical_pairs=pairs[method])
            samples = []
            torch.cuda.empty_cache()
            try:
                for i in range(13):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = perf_counter()
                    output = attention(*inputs, method)
                    torch.cuda.synchronize()
                    elapsed = (perf_counter() - start) * 1000
                    if i >= 3:
                        samples.append(dict(ms=elapsed,
                                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                            peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                    del output
                row.update(samples=samples, median_ms=median(s["ms"] for s in samples))
            except RuntimeError as error:
                row.update(failure=str(error))
            result["runs"].append(row)
            save_json(destination, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--inference-sweep", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=("dense", "window", "linear", "dilated", "curriculum"),
                        default=("dense", "window", "linear", "dilated"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    deadline = perf_counter() + args.hours * 3600
    sys.path.insert(0, str(args.course_data / "scripts"))
    from mqar_generator import make_example, validate

    for k in KS:
        for regime in REGIMES:
            validate(make_example("train", k, regime, 0))
    data = args.course_data / "data" / "mqar"
    development = frozen_data(data, "development", validate)
    test = frozen_data(data, "test", validate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.inference_sweep:
        inference_sweep(args.output_dir / "inference_sweep.json")
    for method, k, regime in GRID:
        if method not in args.methods:
            continue
        for seed in SEEDS:
            print(f"Starting {method} K={k} {regime} seed={seed}", flush=True)
            if not train(args.output_dir, method, k, regime, seed, make_example, development, test, deadline):
                print("Time limit reached; last training state was saved.", flush=True)
                return
            if perf_counter() >= deadline:
                return


if __name__ == "__main__":
    main()

