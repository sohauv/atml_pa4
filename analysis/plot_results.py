"""Render PA4 figures from the locally generated summary JSON files."""
import json
from pathlib import Path
from statistics import mean, median
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("figures")


def read(name):
    return json.loads(Path(name).read_text(encoding="utf-8"))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def compression(data):
    conditions = data["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained", sharey=True)
    groups = (
        (("Magnitude", ["magnitude_25", "magnitude_50", "magnitude_75"]),
         ("Wanda", ["wanda_25", "wanda_50", "wanda_75"]),
         ("Nonuniform Wanda", ["nonuniform_50", "nonuniform_75"]),
         ("Wanda 2:4", ["wanda_2of4"])),
        (("Uniform PTQ", ["uniform_int8", "uniform_int4", "uniform_int3"]),
         ("Folded AWQ", ["folded_int8", "folded_int4", "folded_int3"]),
         ("Folded mixed precision", ["folded_mixed_int4_budget", "folded_mixed_int3_budget"])),
    )
    for axis, group, title in zip(axes, groups, ("Pruning", "Quantization")):
        for label, keys in group:
            rows = [conditions[key] for key in keys]
            axis.plot([r["estimated_packed_model_bytes"] / 2**30 for r in rows],
                      [r["paired_vs_fp16"]["wikitext"]["ppl"] for r in rows],
                      marker="o", label=label)
        axis.axhline(data["baseline"]["quality"]["wikitext"]["ppl"], color="0.45",
                     linestyle="--", label="FP16 quality")
        axis.set(title=title, xlabel="Estimated packed model size (GiB)", yscale="log")
        axis.legend(fontsize=8)
    axes[0].set_ylabel("WikiText-2 perplexity")
    save(fig, "task1_compression")

    keys = ["nonuniform_50", "nonuniform_75", "folded_mixed_int4_budget", "folded_mixed_int3_budget"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained", sharex=True, sharey=True)
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    for axis, key in zip(axes.flat, keys):
        rows = conditions[key]["allocation"]
        values = [[rows[f"{layer}.{name}"]["masked"] / rows[f"{layer}.{name}"]["weights"]
                   if key.startswith("nonuniform") else rows[f"{layer}.{name}"]["bits"]
                   for layer in range(28)] for name in names]
        image = axis.imshow(values, aspect="auto", interpolation="nearest", cmap="viridis",
                            vmin=.1 if key.startswith("nonuniform") else 2,
                            vmax=.85 if key.startswith("nonuniform") else 8)
        axis.set(title=key.replace("_", " "), yticks=range(len(names)), yticklabels=names,
                 xlabel="Layer index")
        fig.colorbar(image, ax=axis, label="Masked fraction" if key.startswith("nonuniform") else "Bits")
    save(fig, "task1_projection_allocations")


def attention(data):
    rows = data["attention"]["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for name in sorted({r["operator"] for r in rows}):
        selected = sorted((r for r in rows if r["operator"] == name and r["dtype"] == "torch.float16"),
                          key=lambda row: row["length"])
        for axis, field in zip(axes, ("forward", "forward_backward")):
            points = [(r["length"], r[field]["median_ms"]) for r in selected if "median_ms" in r[field]]
            if points:
                axis.plot(*zip(*points), marker="o", label=name.removeprefix("sdpa_"))
    for axis, title in zip(axes, ("Forward", "Forward + backward")):
        axis.set(title=title, xlabel="Sequence length", ylabel="Median latency (ms)",
                 xscale="log", yscale="log")
    axes[1].legend(fontsize=8)
    save(fig, "task2_attention_latency")


def recall(data):
    runs = data["recall"]
    groups = sorted({(r["method"], r["K"], r["training_regime"]) for r in runs})
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout="constrained", height_ratios=(1., 1.5))
    axis = axes[0]
    for method, regime in (("dense", "near"), ("dense", "mixed"), ("curriculum", "near_then_mixed")):
        curves = [r["learning_curve"] for r in runs if (r["method"], r["K"], r["training_regime"])
                  == (method, 16, regime)]
        assert len(curves) == 3
        steps = [entry["step"] for entry in curves[0] if "development" in entry]
        values = [[100 * entry["development"]["K16_mixed"]["accuracy"]["overall"]["accuracy"]
                   for entry in curve if "development" in entry] for curve in curves]
        axis.plot(steps, np.mean(values, axis=0), marker="o", label=f"{method} · {regime}")
        axis.fill_between(steps, np.min(values, axis=0), np.max(values, axis=0), alpha=.15)
    axis.set(xlabel="Optimizer updates", ylabel="Mixed development accuracy (%)", ylim=(0, 103))
    axis.legend(fontsize=8)
    matrix = []
    for method, k, regime in groups:
        subset = [r for r in runs if (r["method"], r["K"], r["training_regime"]) == (method, k, regime)]
        lookup = [r["test"][f"K{k}_mixed"]["accuracy"] for r in subset]
        matrix.append([mean(row[span]["accuracy"] for row in lookup) * 100
                       for span in ("0-31", "32-63", "64-127", "128-255")])
    image = axes[1].imshow(matrix, aspect="auto", vmin=0, vmax=100, cmap="viridis")
    axes[1].set(xticks=range(4), xticklabels=("0–31", "32–63", "64–127", "128–255"),
                yticks=range(len(groups)),
                yticklabels=[f"{m}, K={k}, {r}" for m, k, r in groups],
                xlabel="Source–query distance on mixed test set")
    fig.colorbar(image, ax=axes[1], label="Mean accuracy across seeds (%)")
    save(fig, "task2_mqar")


def speculative(data):
    names = list(data)
    fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    for i, gamma in enumerate((2, 4)):
        rows = [data[n]["performance"][f"gamma{gamma}/all"] for n in names]
        y = np.arange(len(names)) + (-.12 if i == 0 else .12)
        x = np.array([row["median_throughput_ratio"] for row in rows])
        low, high = np.array([row["throughput_ratio_ci95"] for row in rows]).T
        axis.errorbar(x, y, xerr=[x-low, high-x], fmt="o", capsize=3, label=f"γ={gamma}")
    axis.axvline(1, color="0.45", linestyle="--")
    axis.set(yticks=range(len(names)), yticklabels=names,
             xlabel="Median throughput relative to target-only (paired 95% interval)")
    axis.legend()
    save(fig, "task3_throughput")


def archived_json(path, basename):
    with ZipFile(path) as archive:
        name = next(name for name in archive.namelist() if Path(name).name == Path(basename).name)
        return json.loads(archive.read(name))


def sensitivity():
    prune = archived_json("results/task1/pa4-nonuniform-results.zip", "allocations.json")["marginal_costs"]
    quant = archived_json("results/task1/pa4-awq-folded-results.zip", "folded_precision_profiles.json")["profiles"]
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), layout="constrained", sharex=True)
    for axis, table, title in (
        (axes[0], {key: row[len(row)//2] for key, row in prune.items()},
         "Pruning sensitivity · marginal score at 50% row rank"),
        (axes[1], {key: row["3"]["relative_squared_error"] for key, row in quant.items()},
         "Quantization sensitivity · INT3 calibration error before folding"),
    ):
        matrix = [[np.log10(max(table[f"{layer}.{name}"], 1e-30))
                   for layer in range(28)] for name in names]
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
        axis.set(title=title, yticks=range(len(names)), yticklabels=names)
        fig.colorbar(image, ax=axis, label="log₁₀ sensitivity")
    axes[1].set_xlabel("Layer index")
    save(fig, "task1_sensitivity")


def kv_figures(data):
    labels = {"full": "FP16", "sink_recent": "Sink + recent", "question": "Question-aware",
              "int4_token": "INT4 token-wise", "int4_channel": "INT4 channel-wise K",
              "hybrid": "Hybrid"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    axis = axes[0]
    for method, row in data["kv"].items():
        axis.plot(100 * row["mean_fp16_context_byte_fraction"], 100 * row["exact_match"],
                  marker="o", linestyle="none", label=labels[method], alpha=.85)
    axis.set(xlabel="Mean packed context size (% of FP16)", ylabel="Exact match (%)", ylim=(-4, 105))
    axis.legend(fontsize=7)
    axis = axes[1]
    for method in ("full", "int4_token", "int4_channel", "hybrid"):
        rows = sorted((row for row in data["kv_packing"] if row["policy"] == method),
                      key=lambda row: (row["batch_size"], row["length"]))
        axis.plot(range(len(rows)), [median(s["decode_with_unpack_ms"] for s in row["samples"])
                                    for row in rows], marker="o", label=labels[method])
    axis.set(xticks=range(4), xticklabels=("1 × 2048", "1 × 4096", "8 × 2048", "8 × 4096"),
             xlabel="Batch × context length", ylabel="Median unpack + decode (ms)", yscale="log")
    axis.legend(fontsize=7)
    save(fig, "task2_kv_cache")


def distillation_figures(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    axis = axes[0]
    with ZipFile("results/task3/pa4-distillation-results.zip") as archive:
        names = sorted(name for name in archive.namelist() if name.endswith("/training.json"))
        for name in names:
            row = json.loads(archive.read(name))
            updates = row["updates"]
            smoothed = [mean(r["mean_kl"] for r in updates[max(0, i-9):i+1])
                        for i in range(len(updates))]
            axis.plot([r["scored_tokens"] for r in updates], smoothed,
                      label=f"{row['method'].replace('_', '-')} {row['seed']}")
    axis.set(xlabel="Teacher-scored response tokens", ylabel="Forward KL (10-update mean)")
    axis.legend(fontsize=7)
    axis = axes[1]
    groups = ("math/1-64", "math/65-192", "familiar/1-64", "shifted/1-64")
    for draft, row in data.items():
        values = [row["alignment"].get(f"teacher/{key}", {}).get("expected_acceptance", np.nan)
                  for key in groups]
        axis.plot(range(len(groups)), values, marker="o", label=draft.replace("_", "-"))
    axis.set(xticks=range(len(groups)), xticklabels=("Math 1–64", "Math 65–192", "Familiar", "Shifted"),
             ylabel="Mean expected acceptance on teacher prefixes")
    axis.legend(fontsize=7)
    save(fig, "task3_training_and_alignment")


def main():
    OUT.mkdir(exist_ok=True)
    compression(read("results/task1/task1-summary.json"))
    task2 = read("results/task2/task2-summary.json")
    attention(task2)
    recall(task2)
    task3 = read("results/task3/task3-summary.json")
    speculative(task3)
    sensitivity()
    kv_figures(task2)
    distillation_figures(task3)


if __name__ == "__main__":
    main()
