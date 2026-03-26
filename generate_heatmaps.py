#!/usr/bin/env python3
"""Generate R_{t,i} accuracy matrix heatmaps for paper appendix.

Shows the full accuracy matrix where entry (i,j) = accuracy on task j
after training on task i. Compares LS-Tuned vs DER++ vs GPM.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("paper/figures")

METHODS = {
    "latent_shift_tuned": "LS-Tuned (Ours)",
    "der": "DER++",
    "gpm": "GPM",
}

BENCHMARKS = {
    "split_cifar100": "Split-CIFAR-100",
    "split_tinyimagenet": "Split-TinyImageNet",
}

SEED = 42  # Use single seed for heatmap clarity


def load_accuracy_matrix(method, benchmark, seed):
    path = RESULTS_DIR / f"{method}_{benchmark}_seed{seed}.json"
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return np.array(d.get("accuracy_matrix", []))


def plot_heatmap(ax, matrix, title, num_tasks):
    """Plot a single accuracy matrix heatmap."""
    # Mask upper triangle (future tasks not yet trained)
    masked = np.full_like(matrix, np.nan)
    for i in range(num_tasks):
        for j in range(i + 1):
            masked[i][j] = matrix[i][j]

    im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
    ax.set_xlabel("Task evaluated", fontsize=9)
    ax.set_ylabel("After training task", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks(range(num_tasks))
    ax.set_yticks(range(num_tasks))
    ax.set_xticklabels(range(num_tasks), fontsize=7)
    ax.set_yticklabels(range(num_tasks), fontsize=7)

    # Add text annotations for key cells
    for i in range(num_tasks):
        for j in range(i + 1):
            val = matrix[i][j]
            color = "white" if val < 0.3 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5, color=color)

    return im


for bench_key, bench_name in BENCHMARKS.items():
    fig, axes = plt.subplots(1, len(METHODS), figsize=(5 * len(METHODS), 4.5))
    if len(METHODS) == 1:
        axes = [axes]

    has_data = False
    for idx, (method_key, method_name) in enumerate(METHODS.items()):
        am = load_accuracy_matrix(method_key, bench_key, SEED)
        if am is not None and am.size > 0:
            num_tasks = am.shape[0]
            im = plot_heatmap(axes[idx], am, method_name, num_tasks)
            has_data = True
        else:
            axes[idx].set_title(f"{method_name}\n(no data)", fontsize=10)
            axes[idx].axis("off")

    if has_data:
        fig.colorbar(im, ax=axes, shrink=0.6, label="Accuracy")
        fig.suptitle(f"Accuracy Matrix $R_{{t,i}}$ — {bench_name} (seed {SEED})",
                     fontsize=12, fontweight="bold", y=1.02)
        fig.subplots_adjust(wspace=0.3)

        out_path = OUTPUT_DIR / f"accuracy_heatmap_{bench_key}.pdf"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {out_path}")
    else:
        print(f"No data for {bench_name}, skipping.")
    plt.close(fig)
