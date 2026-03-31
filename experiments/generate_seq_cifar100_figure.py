from __future__ import annotations

"""Generate accuracy-over-tasks figure for Seq-CIFAR-100 (50-task experiment).

Reads existing result JSONs for the seq_cifar100 benchmark, computes average
accuracy after each task, and produces a publication-quality line plot
comparing all methods across 50 tasks.

Saves:
    paper/figures/split_cifar100/accuracy_over_50_tasks.pdf

Usage:
    python experiments/generate_seq_cifar100_figure.py
    python experiments/generate_seq_cifar100_figure.py --results results --output paper/figures
    python experiments/generate_seq_cifar100_figure.py --seeds 42 123 456
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.plots import (
    load_multiseed_results,
    METHOD_STYLES,
)

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

# Extended styles for prompt-based methods that use vit encoder
# (they appear in results as the same method name, just with _vit in filename)
EXTENDED_STYLES = {
    **METHOD_STYLES,
    "er":                 {"color": "#00BCD4", "marker": "P", "label": "ER"},
    "trgp":               {"color": "#388E3C", "marker": "p", "label": "TRGP"},
}


def compute_avg_accuracy_curve(accuracy_matrix: list[list[float]]) -> list[float]:
    """Compute average accuracy after training on each task.

    For task t: average accuracy = mean(accuracy on tasks 0..t after training task t).
    """
    mat = np.array(accuracy_matrix)
    T = mat.shape[0]
    avg_accs = []
    for t in range(T):
        # Accuracy on tasks 0..t after training task t
        accs = [mat[t, j] for j in range(t + 1)]
        avg_accs.append(float(np.mean(accs)))
    return avg_accs


def plot_accuracy_over_50_tasks(
    multi_results: dict[str, list[dict]],
    save_path: str | None = None,
):
    """Plot average accuracy vs. tasks for all methods on Seq-CIFAR-100.

    When multiple seeds are available, plots mean with shaded std band.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Process methods in a consistent order
    method_order = [m for m in METHOD_STYLES if m in multi_results]

    for method in method_order:
        runs = multi_results[method]
        style = METHOD_STYLES[method]

        # Compute accuracy curves for all seeds
        curves = []
        for r in runs:
            mat = r.get("accuracy_matrix")
            if mat is None:
                continue
            curve = compute_avg_accuracy_curve(mat)
            curves.append(curve)

        if not curves:
            continue

        # Ensure all curves have the same length (trim to shortest)
        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]
        curves_arr = np.array(curves)

        tasks = np.arange(min_len)
        mean_curve = curves_arr.mean(axis=0)
        std_curve = curves_arr.std(axis=0)

        # Reduce marker density for 50-task plots
        markevery = max(1, min_len // 10)

        ax.plot(
            tasks, mean_curve,
            color=style["color"], marker=style["marker"],
            label=style["label"], linewidth=2, markersize=5,
            markevery=markevery,
        )

        if len(curves) > 1:
            ax.fill_between(
                tasks,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.12, color=style["color"],
            )

    ax.set_xlabel("Tasks Trained")
    ax.set_ylabel("Average Accuracy")
    ax.set_title("Average Accuracy Over Tasks — Seq-CIFAR-100 (50 Tasks)")
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, None)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate Seq-CIFAR-100 accuracy-over-tasks figure"
    )
    parser.add_argument("--results", type=str, default="results",
                        help="Directory with result JSONs")
    parser.add_argument("--output", type=str, default="paper/figures",
                        help="Base directory for output figures")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Filter to specific seeds (default: use all available)")
    args = parser.parse_args()

    result_dir = Path(args.results)
    output_dir = Path(args.output) / "split_cifar100"
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = "seq_cifar100"
    print(f"Loading Seq-CIFAR-100 results from {result_dir}...")
    multi = load_multiseed_results(result_dir, benchmark, seeds=args.seeds)

    if not multi:
        print(f"No results found for '{benchmark}' in {result_dir}/")
        print("Expected files matching: *_seq_cifar100_seed*.json")
        sys.exit(1)

    # Report what was found
    for method, runs in sorted(multi.items()):
        n_tasks = len(runs[0].get("accuracy_matrix", []))
        print(f"  {method}: {len(runs)} seed(s), {n_tasks} tasks")

    save_path = str(output_dir / "accuracy_over_50_tasks.pdf")
    plot_accuracy_over_50_tasks(multi, save_path=save_path)

    print(f"\nFigure saved to {save_path}")


if __name__ == "__main__":
    main()
