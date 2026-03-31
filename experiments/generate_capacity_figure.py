from __future__ import annotations

"""Generate capacity saturation figure showing subspace dimension growth vs. tasks.

Reads existing LatentShift result JSONs (which contain per_task_extras with
archive_rank) and plots how the archive subspace grows across tasks, showing
when capacity saturates (approaches latent_dim).

Supports multiple benchmarks for comparison and multiple seeds for error bands.

Saves:
    paper/figures/capacity_saturation.pdf

Usage:
    python experiments/generate_capacity_figure.py
    python experiments/generate_capacity_figure.py --results results --output paper/figures
    python experiments/generate_capacity_figure.py --benchmarks split_cifar100 seq_cifar100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.plots import load_multiseed_results, load_result

# Paper-quality defaults (matching plots.py)
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

# Colors for different benchmarks
BENCHMARK_STYLES = {
    "split_mnist":        {"color": "#4CAF50", "label": "Split-MNIST (5 tasks)"},
    "split_cifar10":      {"color": "#FF9800", "label": "Split-CIFAR-10 (5 tasks)"},
    "split_cifar100":     {"color": "#2196F3", "label": "Split-CIFAR-100 (10 tasks)"},
    "seq_cifar100":       {"color": "#9C27B0", "label": "Seq-CIFAR-100 (50 tasks)"},
    "permuted_mnist":     {"color": "#F44336", "label": "Permuted-MNIST (10 tasks)"},
    "split_tinyimagenet": {"color": "#00BCD4", "label": "Split-TinyImageNet (10 tasks)"},
}


def extract_archive_ranks(result: dict) -> tuple[list[int], list[int], int]:
    """Extract archive rank growth from a result dict.

    Returns:
        tasks: List of task indices.
        ranks: List of archive ranks after each task.
        latent_dim: The latent space dimensionality (capacity).
    """
    # Try per_task_extras at top level, then nested under metrics
    extras = result.get("per_task_extras", {})
    if not extras:
        extras = result.get("metrics", {}).get("per_task_extras", {})

    if not extras:
        return [], [], 0

    tasks = sorted(int(k) for k in extras.keys())
    ranks = []
    for t in tasks:
        ex = extras[str(t)]
        if isinstance(ex, dict):
            ranks.append(ex.get("archive_rank", 0))
        else:
            ranks.append(0)

    latent_dim = result.get("config", {}).get("latent_dim", 256)
    return tasks, ranks, latent_dim


def find_saturation_task(ranks: list[int], latent_dim: int, threshold: float = 0.95) -> int | None:
    """Find the first task where archive rank exceeds threshold * latent_dim.

    Returns the task index, or None if saturation is never reached.
    """
    for i, r in enumerate(ranks):
        if r >= threshold * latent_dim:
            return i
    return None


def plot_capacity_saturation(
    multiseed_data: dict[str, list[dict]],
    save_path: str | None = None,
):
    """Plot archive dimension growth vs. tasks for LatentShift across benchmarks.

    Shows mean +/- std when multiple seeds are available, and marks the
    capacity limit with a dashed line.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    latent_dim_global = None

    for bench, style in BENCHMARK_STYLES.items():
        if bench not in multiseed_data:
            continue

        runs = multiseed_data[bench]
        all_tasks = []
        all_ranks = []

        for r in runs:
            tasks, ranks, latent_dim = extract_archive_ranks(r)
            if not tasks:
                continue
            all_tasks.append(tasks)
            all_ranks.append(ranks)
            if latent_dim_global is None:
                latent_dim_global = latent_dim

        if not all_ranks:
            continue

        # Align to shortest sequence
        min_len = min(len(r) for r in all_ranks)
        ranks_arr = np.array([r[:min_len] for r in all_ranks])
        tasks_arr = np.arange(min_len)

        mean_ranks = ranks_arr.mean(axis=0)
        std_ranks = ranks_arr.std(axis=0)

        # Reduce marker density for long sequences
        markevery = max(1, min_len // 10)

        ax.plot(
            tasks_arr, mean_ranks, "o-",
            color=style["color"], linewidth=2, markersize=5,
            label=style["label"], markevery=markevery,
        )

        if len(all_ranks) > 1:
            ax.fill_between(
                tasks_arr,
                mean_ranks - std_ranks,
                np.minimum(mean_ranks + std_ranks, latent_dim or 256),
                alpha=0.12, color=style["color"],
            )

        # Mark saturation point
        sat_task = find_saturation_task(mean_ranks.tolist(), latent_dim or 256)
        if sat_task is not None:
            ax.axvline(x=sat_task, color=style["color"], linestyle=":",
                        alpha=0.5, linewidth=1)
            ax.annotate(
                f"95% at T={sat_task}",
                xy=(sat_task, mean_ranks[sat_task]),
                xytext=(sat_task + 1, mean_ranks[sat_task] + 10),
                fontsize=8, color=style["color"],
                arrowprops=dict(arrowstyle="->", color=style["color"], lw=0.8),
            )

    # Draw capacity line
    if latent_dim_global is not None:
        ax.axhline(y=latent_dim_global, color="#F44336", linestyle="--",
                    linewidth=1.5, label=f"Capacity ($d={latent_dim_global}$)")
        ax.axhline(y=0.95 * latent_dim_global, color="#F44336", linestyle=":",
                    linewidth=1, alpha=0.4, label="95% Capacity")

    ax.set_xlabel("Task")
    ax.set_ylabel("Archive Rank (Occupied Dimensions)")
    ax.set_title("Subspace Capacity Saturation Across Benchmarks")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, (latent_dim_global or 256) * 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate capacity saturation figure"
    )
    parser.add_argument("--results", type=str, default="results",
                        help="Directory with result JSONs")
    parser.add_argument("--output", type=str, default="paper/figures",
                        help="Directory for output figures")
    parser.add_argument(
        "--benchmarks", type=str, nargs="+",
        default=["split_mnist", "split_cifar10", "split_cifar100",
                 "seq_cifar100", "permuted_mnist", "split_tinyimagenet"],
        help="Benchmarks to include in the figure",
    )
    args = parser.parse_args()

    result_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading LatentShift results from {result_dir}...")

    # Load LatentShift results across benchmarks and seeds
    multiseed_data: dict[str, list[dict]] = {}

    for bench in args.benchmarks:
        multi = load_multiseed_results(result_dir, bench)
        # Only keep LatentShift results (both standard and tuned)
        ls_runs = []
        for method in ["latent_shift", "latent_shift_tuned"]:
            if method in multi:
                ls_runs.extend(multi[method])

        if ls_runs:
            # Check that at least one run has archive_rank data
            has_data = any(
                extract_archive_ranks(r)[0]  # non-empty tasks list
                for r in ls_runs
            )
            if has_data:
                multiseed_data[bench] = ls_runs
                n_tasks = len(extract_archive_ranks(ls_runs[0])[0])
                print(f"  {bench}: {len(ls_runs)} run(s), {n_tasks} tasks")

    if not multiseed_data:
        print("No LatentShift results with archive_rank data found.")
        print("Run experiments first, then re-run this script.")
        sys.exit(1)

    save_path = str(output_dir / "capacity_saturation.pdf")
    plot_capacity_saturation(multiseed_data, save_path=save_path)

    # Print saturation summary
    print(f"\nCapacity Saturation Summary:")
    for bench, runs in multiseed_data.items():
        for r in runs[:1]:  # Just report first seed
            tasks, ranks, latent_dim = extract_archive_ranks(r)
            if ranks:
                final_rank = ranks[-1]
                utilization = final_rank / latent_dim * 100
                sat = find_saturation_task(ranks, latent_dim)
                sat_str = f"task {sat}" if sat is not None else "not reached"
                print(f"  {bench}: final_rank={final_rank}/{latent_dim} "
                      f"({utilization:.1f}%), 95% saturation: {sat_str}")

    print(f"\nFigure saved to {save_path}")


if __name__ == "__main__":
    main()
