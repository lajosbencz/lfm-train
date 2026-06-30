#!/usr/bin/env python3
"""
Generate a 2x2 training metrics figure from a training_log.jsonl file.

Panels:
  [0,0] Loss (train + eval) over step
  [0,1] Learning rate schedule over step
  [1,0] GPU memory (MB) over step
  [1,1] Train loss per epoch (mean across steps in epoch)
"""
from __future__ import annotations

import json
from pathlib import Path


def plot_run(log_path: str | Path, out_path: str | Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    records = [
        json.loads(line)
        for line in Path(log_path).read_text().splitlines()
        if line.strip()
    ]
    if not records:
        return

    steps      = [r["step"] for r in records]
    train_loss = [r.get("loss") for r in records]
    eval_loss  = [r.get("eval_loss") for r in records]
    lr         = [r.get("learning_rate") for r in records]
    mem_mb     = [r.get("gpu_mem_mb") for r in records]
    epochs     = [r.get("epoch", 0.0) for r in records]

    # per-epoch mean train loss
    epoch_loss: dict[int, list[float]] = {}
    for r in records:
        if r.get("loss") is None:
            continue
        ep = int(r.get("epoch", 0.0))
        epoch_loss.setdefault(ep, []).append(r["loss"])
    epoch_keys = sorted(epoch_loss)
    epoch_means = [sum(epoch_loss[e]) / len(epoch_loss[e]) for e in epoch_keys]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(title or "Training metrics", fontsize=13, fontweight="bold")

    # == [0,0] loss ==
    ax = axes[0, 0]
    train_steps = [s for s, v in zip(steps, train_loss) if v is not None]
    train_vals  = [v for v in train_loss if v is not None]
    eval_steps  = [s for s, v in zip(steps, eval_loss) if v is not None]
    eval_vals   = [v for v in eval_loss if v is not None]

    ax.plot(train_steps, train_vals, color="#2196F3", linewidth=1.5, label="train loss")
    if eval_vals:
        ax.plot(eval_steps, eval_vals, color="#FF9800", linewidth=1.5,
                linestyle="--", marker="o", markersize=3, label="eval loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # == [0,1] learning rate ==
    ax = axes[0, 1]
    lr_steps = [s for s, v in zip(steps, lr) if v is not None]
    lr_vals  = [v for v in lr if v is not None]
    ax.plot(lr_steps, lr_vals, color="#4CAF50", linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning rate schedule")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))
    ax.grid(True, alpha=0.3)

    # == [1,0] GPU memory ==
    ax = axes[1, 0]
    mem_steps = [s for s, v in zip(steps, mem_mb) if v is not None]
    mem_vals  = [v / 1024 for v in mem_mb if v is not None]  # -> GB
    if mem_vals:
        ax.fill_between(mem_steps, mem_vals, alpha=0.25, color="#9C27B0")
        ax.plot(mem_steps, mem_vals, color="#9C27B0", linewidth=1.2)
        ax.set_ylabel("GPU memory (GB)")
    else:
        ax.text(0.5, 0.5, "no GPU memory data", transform=ax.transAxes,
                ha="center", va="center", color="gray")
    ax.set_xlabel("step")
    ax.set_title("GPU memory usage")
    ax.grid(True, alpha=0.3)

    # == [1,1] loss per epoch ==
    ax = axes[1, 1]
    if epoch_keys:
        ax.bar(epoch_keys, epoch_means, color="#2196F3", alpha=0.75, width=0.6)
        ax.plot(epoch_keys, epoch_means, color="#1565C0", marker="o",
                markersize=5, linewidth=1.5)
        for ep, mean in zip(epoch_keys, epoch_means):
            ax.text(ep, mean + 0.002, f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=7.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("mean train loss")
        ax.set_title("Loss per epoch")
        ax.set_xticks(epoch_keys)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark(results_path: str | Path, out_path: str | Path) -> None:
    """Comparison bar chart from outputs/benchmark.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads(Path(results_path).read_text())
    models = [d["model"].split("/")[-1] for d in data]
    overall = [d["accuracy"].get("__overall__", {}).get("pct", 0) for d in data]

    buckets = ["single_seen", "single_unseen", "composed_trained", "composed_heldout"]
    bucket_pcts = {
        b: [d.get("buckets", {}).get(b, {}).get("pct", 0) for d in data]
        for b in buckets
    }

    model_colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(models), 1)))
    bucket_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(buckets)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Benchmark - model comparison", fontsize=13, fontweight="bold")

    # == overall accuracy bars ==
    ax = axes[0]
    bars = ax.bar(models, overall, color=model_colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, overall):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("exact-match accuracy (%)")
    ax.set_title("Overall accuracy")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(True, alpha=0.3, axis="y")

    # == generalization buckets: grouped by model, one bar per bucket ==
    ax = axes[1]
    x = np.arange(len(models))
    width = 0.8 / len(buckets)
    for i, (bucket, color) in enumerate(zip(buckets, bucket_colors)):
        vals = bucket_pcts[bucket]
        offset = (i - (len(buckets) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=bucket, color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8, rotation=30, ha="right")
    ax.set_ylim(0, 115)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Generalization buckets")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Benchmark chart -> {out_path}")
