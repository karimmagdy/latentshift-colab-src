#!/usr/bin/env python3
"""Generate capacity saturation analysis figure for the paper.

Shows per-task final accuracy overlaid with archive rank growth,
highlighting which tasks had gradient projection protection (free_dim > 0)
vs no protection (free_dim = 0).
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("paper/figures")

# Configurations to analyze: (method_key, display_name, benchmarks)
METHODS = [
    ("latent_shift", "LS (d=256)"),
    ("latent_shift_tuned", "LS-Tuned (d=1024)"),
]

BENCHMARKS = {
    "split_cifar100": ("Split-CIFAR-100", 10),
    "split_tinyimagenet": ("Split-TinyImageNet", 10),
}

SEEDS = [42, 123, 456]


def load_results(method, benchmark, seed):
    path = RESULTS_DIR / f"{method}_{benchmark}_seed{seed}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_per_task_data(result):
    """Extract per-task final accuracy and archive/free dims."""
    am = result.get("accuracy_matrix", [])
    extras = result.get("per_task_extras", {})
    num_tasks = len(am)

    # Final accuracy for each task = accuracy on that task after ALL tasks trained
    final_acc = [am[-1][i] for i in range(num_tasks)] if am else []

    archive_ranks = []
    free_dims = []
    for t in range(num_tasks):
        e = extras.get(str(t), {})
        archive_ranks.append(e.get("archive_rank", 0))
        free_dims.append(e.get("free_dim", 0))

    return final_acc, archive_ranks, free_dims


def make_figure():
    fig, axes = plt.subplots(
        len(METHODS), len(BENCHMARKS),
        figsize=(5.5 * len(BENCHMARKS), 3.5 * len(METHODS)),
        squeeze=False,
    )

    for row, (method_key, method_name) in enumerate(METHODS):
        for col, (bench_key, (bench_name, num_tasks)) in enumerate(BENCHMARKS.items()):
            ax = axes[row][col]
            ax2 = ax.twinx()

            # Collect multi-seed data
            all_final_acc = []
            all_archive = []
            all_free = []

            for seed in SEEDS:
                r = load_results(method_key, bench_key, seed)
                if r is None:
                    continue
                fa, ar, fd = extract_per_task_data(r)
                all_final_acc.append(fa)
                all_archive.append(ar)
                all_free.append(fd)

            if not all_final_acc:
                ax.set_title(f"{method_name}\n{bench_name} (no data)")
                continue

            tasks = np.arange(num_tasks)
            acc_mean = np.mean(all_final_acc, axis=0)
            acc_std = np.std(all_final_acc, axis=0)
            arch_mean = np.mean(all_archive, axis=0)
            free_mean = np.mean(all_free, axis=0)

            # Determine saturation point (first task where free_dim ≈ 0)
            sat_task = num_tasks  # default: never saturated
            for t in range(num_tasks):
                if free_mean[t] < 1:
                    sat_task = t
                    break

            # Color bars by protection status
            colors = ["#2ecc71" if free_mean[t] > 0 else "#e74c3c" for t in tasks]
            bars = ax.bar(tasks, acc_mean * 100, color=colors, alpha=0.7, width=0.6,
                          edgecolor="white", linewidth=0.5)
            ax.errorbar(tasks, acc_mean * 100, yerr=acc_std * 100,
                        fmt="none", ecolor="black", capsize=2, linewidth=0.8)

            # Archive rank line on secondary axis
            ax2.plot(tasks, arch_mean, "k--", linewidth=1.5, marker="s",
                     markersize=4, label="Archive rank")
            latent_dim = int(max(arch_mean))
            ax2.axhline(y=latent_dim, color="gray", linestyle=":", alpha=0.5)
            ax2.set_ylabel("Archive rank", fontsize=9)
            ax2.set_ylim(0, latent_dim * 1.15)

            # Labels
            ax.set_xlabel("Task index", fontsize=9)
            ax.set_ylabel("Final accuracy (%)", fontsize=9)
            ax.set_title(f"{method_name} — {bench_name}", fontsize=10, fontweight="bold")
            ax.set_xticks(tasks)
            ax.set_ylim(0, max(acc_mean * 100) * 1.35)

            # Add saturation annotation
            if sat_task < num_tasks:
                ax.axvline(x=sat_task - 0.5, color="red", linestyle="--",
                           alpha=0.5, linewidth=1)
                ax.text(sat_task - 0.3, ax.get_ylim()[1] * 0.95,
                        f"Saturated\n(task {sat_task})",
                        fontsize=7, color="red", ha="left", va="top")

    # Common legend
    protected_patch = mpatches.Patch(color="#2ecc71", alpha=0.7, label="Protected (free dim > 0)")
    unprotected_patch = mpatches.Patch(color="#e74c3c", alpha=0.7, label="Unprotected (free dim = 0)")
    archive_line = plt.Line2D([0], [0], color="black", linestyle="--",
                               marker="s", markersize=4, label="Archive rank")
    fig.legend(
        handles=[protected_patch, unprotected_patch, archive_line],
        loc="lower center", ncol=3, fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = OUTPUT_DIR / "capacity_saturation.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")

    # Also print summary statistics
    print("\n=== Capacity Saturation Summary ===")
    for method_key, method_name in METHODS:
        for bench_key, (bench_name, num_tasks) in BENCHMARKS.items():
            accs_pre, accs_post = [], []
            for seed in SEEDS:
                r = load_results(method_key, bench_key, seed)
                if r is None:
                    continue
                fa, ar, fd = extract_per_task_data(r)
                for t in range(num_tasks):
                    if fd[t] > 0:
                        accs_pre.append(fa[t])
                    else:
                        accs_post.append(fa[t])
            if accs_pre and accs_post:
                print(f"\n{method_name} on {bench_name}:")
                print(f"  Protected tasks:   mean acc = {np.mean(accs_pre)*100:.1f}% "
                      f"(n={len(accs_pre)//len(SEEDS)} tasks × {len(SEEDS)} seeds)")
                print(f"  Unprotected tasks: mean acc = {np.mean(accs_post)*100:.1f}% "
                      f"(n={len(accs_post)//len(SEEDS)} tasks × {len(SEEDS)} seeds)")
                print(f"  Δ = {(np.mean(accs_pre) - np.mean(accs_post))*100:+.1f}%")
            elif accs_pre:
                print(f"\n{method_name} on {bench_name}: All tasks protected")
            elif accs_post:
                print(f"\n{method_name} on {bench_name}: No data or all unprotected")


if __name__ == "__main__":
    make_figure()
