from __future__ import annotations

"""Visualization utilities for LatentShift continual learning experiments.

All functions accept data loaded from result JSON files and produce
matplotlib figures suitable for conference papers.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Paper-quality defaults
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

# Consistent colors/markers per method
METHOD_STYLES = {
    "latent_shift":       {"color": "#2196F3", "marker": "o", "label": "LatentShift (Ours)"},
    "latent_shift_tuned": {"color": "#1565C0", "marker": "*", "label": "LatentShift-Tuned (Ours)"},
    "der":                {"color": "#009688", "marker": "H", "label": "DER++"},
    "er":                 {"color": "#00BCD4", "marker": "P", "label": "ER"},
    "gpm":                {"color": "#4CAF50", "marker": "s", "label": "GPM"},
    "trgp":               {"color": "#388E3C", "marker": "p", "label": "TRGP"},
    "gpm_lastlayer":      {"color": "#81C784", "marker": "d", "label": "GPM (Last Layer)"},
    "ewc":                {"color": "#FF9800", "marker": "^", "label": "EWC"},
    "packnet":            {"color": "#9C27B0", "marker": "D", "label": "PackNet"},
    "hat":                {"color": "#F44336", "marker": "v", "label": "HAT"},
    "naive":              {"color": "#9E9E9E", "marker": "x", "label": "Naive"},
}


def load_result(path: str | Path) -> dict:
    """Load a single result JSON."""
    with open(path) as f:
        return json.load(f)


def load_results(result_dir: str | Path, benchmark: str, seed: int = 42) -> dict[str, dict]:
    """Load all result JSONs for a given benchmark from a directory.

    When multiple seeds exist for the same method, prefer the specified seed.
    Returns: {method_name: result_dict}
    """
    result_dir = Path(result_dir)
    results: dict[str, dict] = {}
    for p in sorted(result_dir.glob(f"*_{benchmark}_*.json")):
        if p.stem.endswith("_ci"):
            continue
        data = load_result(p)
        method = data["config"]["method"]
        # Prefer the requested seed; don't overwrite if already loaded with it
        if method not in results or f"seed{seed}" in p.name:
            results[method] = data
    return results


# ---------------------------------------------------------------
# 1. Accuracy Matrix Heatmaps
# ---------------------------------------------------------------

def plot_accuracy_heatmaps(
    results: dict[str, dict],
    benchmark_name: str = "",
    save_path: str | None = None,
):
    """Plot T×T accuracy matrix heatmaps side-by-side for all methods."""
    methods = [m for m in METHOD_STYLES if m in results]
    n = len(methods)
    if n == 0:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.2), squeeze=False)

    for idx, method in enumerate(methods):
        ax = axes[0, idx]
        mat = np.array(results[method]["accuracy_matrix"])
        T = mat.shape[0]

        im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues", aspect="equal")
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(range(T))
        ax.set_yticklabels(range(T))
        ax.set_xlabel("Eval Task")
        if idx == 0:
            ax.set_ylabel("After Training Task")
        ax.set_title(METHOD_STYLES[method]["label"])

        # Annotate cells
        for i in range(T):
            for j in range(T):
                if j <= i:
                    val = mat[i, j]
                    color = "white" if val > 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

    fig.suptitle(f"Accuracy Matrices — {benchmark_name}", fontsize=14, y=1.02)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Accuracy")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------
# 2. Accuracy-over-Tasks Line Plot
# ---------------------------------------------------------------

def plot_accuracy_over_tasks(
    results: dict[str, dict],
    benchmark_name: str = "",
    save_path: str | None = None,
):
    """Plot average accuracy vs. number of tasks trained."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for method in METHOD_STYLES:
        if method not in results:
            continue
        style = METHOD_STYLES[method]
        mat = np.array(results[method]["accuracy_matrix"])
        T = mat.shape[0]

        # Average accuracy after training on tasks 0..t
        avg_accs = []
        for t in range(T):
            # Accuracy on tasks 0..t after training task t
            accs = [mat[t, j] for j in range(t + 1)]
            avg_accs.append(np.mean(accs))

        ax.plot(
            range(T), avg_accs,
            color=style["color"], marker=style["marker"],
            label=style["label"], linewidth=2, markersize=6,
        )

    ax.set_xlabel("Tasks Trained")
    ax.set_ylabel("Average Accuracy")
    ax.set_title(f"Average Accuracy Over Tasks — {benchmark_name}")
    ax.legend(loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------
# 3. Subspace Growth Curve
# ---------------------------------------------------------------

def plot_subspace_growth(
    results: dict[str, dict],
    benchmark_name: str = "",
    save_path: str | None = None,
):
    """Plot archive rank vs. task for LatentShift (subspace growth curve)."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for method in ["latent_shift"]:
        if method not in results:
            continue
        data = results[method]
        metrics = data.get("metrics", {})
        extras = metrics.get("per_task_extras", {})

        if not extras:
            print("No per_task_extras found in results — re-run with updated trainer.")
            return fig

        tasks = sorted(int(k) for k in extras.keys())
        ranks = [extras[str(t)]["archive_rank"] for t in tasks]
        latent_dim = data["config"].get("latent_dim", 256)

        ax.plot(tasks, ranks, "o-", color="#2196F3", linewidth=2,
                markersize=8, label="Archive Rank")
        ax.axhline(y=latent_dim, color="#F44336", linestyle="--",
                    linewidth=1.5, label=f"Capacity (d={latent_dim})")
        ax.fill_between(tasks, ranks, latent_dim, alpha=0.15, color="#4CAF50",
                         label="Free Subspace")

    ax.set_xlabel("Task")
    ax.set_ylabel("Archive Rank (Occupied Dimensions)")
    ax.set_title(f"Subspace Growth — {benchmark_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------
# 4. Forgetting Per Task
# ---------------------------------------------------------------

def plot_per_task_forgetting(
    results: dict[str, dict],
    benchmark_name: str = "",
    save_path: str | None = None,
):
    """Bar chart of per-task forgetting for each method."""
    methods = [m for m in METHOD_STYLES if m in results]
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    mat_ref = np.array(results[methods[0]]["accuracy_matrix"])
    T = mat_ref.shape[0]
    x = np.arange(T - 1)
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        style = METHOD_STYLES[method]
        mat = np.array(results[method]["accuracy_matrix"])
        forgetting = []
        for j in range(T - 1):
            peak = max(mat[t, j] for t in range(j, T))
            final = mat[T - 1, j]
            forgetting.append(peak - final)

        ax.bar(x + i * width, forgetting, width,
               color=style["color"], label=style["label"], alpha=0.85)

    ax.set_xlabel("Task")
    ax.set_ylabel("Forgetting")
    ax.set_title(f"Per-Task Forgetting — {benchmark_name}")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(range(T - 1))
    ax.legend(loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------
# 5. Ablation Line Plot
# ---------------------------------------------------------------

def plot_ablation(
    ablation_dir: str | Path,
    param_name: str,
    metric: str = "average_accuracy",
    save_path: str | None = None,
):
    """Plot a single ablation parameter vs. a metric.

    Reads all JSON files in the ablation directory.
    """
    ablation_dir = Path(ablation_dir)
    points = []
    for p in sorted(ablation_dir.glob("*.json")):
        data = load_result(p)
        if "ablation_params" not in data:
            continue
        param_val = data["ablation_params"].get(param_name)
        metric_val = data["metrics"].get(metric)
        if param_val is not None and metric_val is not None:
            points.append((param_val, metric_val))

    if not points:
        print(f"No ablation data found for {param_name}")
        return

    points.sort()
    xs, ys = zip(*points)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, "o-", color="#2196F3", linewidth=2, markersize=8)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xlabel(param_name.replace("_", " ").title())
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Ablation: {param_name}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------
# 6. Summary Bar Chart
# ---------------------------------------------------------------

def plot_summary_bars(
    results: dict[str, dict],
    benchmark_name: str = "",
    save_path: str | None = None,
):
    """Grouped bar chart of average accuracy and forgetting for all methods."""
    methods = [m for m in METHOD_STYLES if m in results]
    if not methods:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    accs = [results[m]["metrics"]["average_accuracy"] for m in methods]
    fgts = [results[m]["metrics"]["average_forgetting"] for m in methods]
    colors = [METHOD_STYLES[m]["color"] for m in methods]
    labels = [METHOD_STYLES[m]["label"] for m in methods]

    x = np.arange(len(methods))

    ax1.bar(x, accs, color=colors, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("Average Accuracy")
    ax1.set_title("Average Accuracy")
    ax1.set_ylim(0, 1.05)
    for i, v in enumerate(accs):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    ax2.bar(x, fgts, color=colors, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("Average Forgetting")
    ax2.set_title("Average Forgetting")
    for i, v in enumerate(fgts):
        ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle(f"Method Comparison — {benchmark_name}", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------
# 7. Multi-Seed Summary (mean ± std)
# ---------------------------------------------------------------

def load_multiseed_results(
    result_dir: str | Path, benchmark: str, seeds: list[int] | None = None,
) -> dict[str, list[dict]]:
    """Load results across multiple seeds for each method.

    Returns: {method_name: [result_dict_seed1, ...]}
    """
    result_dir = Path(result_dir)
    out: dict[str, list[dict]] = {}
    for p in sorted(result_dir.glob(f"*_{benchmark}_seed*.json")):
        if p.stem.endswith("_ci"):
            continue
        data = load_result(p)
        method = data["config"]["method"]
        if seeds is not None:
            seed_str = p.stem.split("seed")[-1]
            if int(seed_str) not in seeds:
                continue
        out.setdefault(method, []).append(data)
    return out


def plot_multiseed_bars(
    multi_results: dict[str, list[dict]],
    benchmark_name: str = "",
    save_path: str | None = None,
):
    """Bar chart with error bars showing mean ± std across seeds."""
    methods = [m for m in METHOD_STYLES if m in multi_results]
    if not methods:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    acc_means, acc_stds = [], []
    fgt_means, fgt_stds = [], []
    colors = []
    labels = []

    for m in methods:
        runs = multi_results[m]
        accs = [r["metrics"]["average_accuracy"] for r in runs]
        fgts = [r["metrics"]["average_forgetting"] for r in runs]
        acc_means.append(np.mean(accs))
        acc_stds.append(np.std(accs))
        fgt_means.append(np.mean(fgts))
        fgt_stds.append(np.std(fgts))
        colors.append(METHOD_STYLES[m]["color"])
        labels.append(f"{METHOD_STYLES[m]['label']} (n={len(runs)})")

    x = np.arange(len(methods))

    ax1.bar(x, acc_means, yerr=acc_stds, color=colors, alpha=0.85, capsize=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Average Accuracy")
    ax1.set_title("Average Accuracy (mean ± std)")
    ax1.set_ylim(0, 1.05)

    ax2.bar(x, fgt_means, yerr=fgt_stds, color=colors, alpha=0.85, capsize=4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Average Forgetting")
    ax2.set_title("Average Forgetting (mean ± std)")

    fig.suptitle(f"Multi-Seed Comparison — {benchmark_name}", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------
# 8. Computation Cost Table (LaTeX)
# ---------------------------------------------------------------

def generate_cost_table(
    result_dir: str | Path,
    benchmarks: list[str] | None = None,
    save_path: str | None = None,
) -> str:
    """Generate a LaTeX table comparing wall-clock training time per method.

    Searches across all available seeds to find timing data.
    """
    if benchmarks is None:
        benchmarks = ["split_mnist", "split_cifar10", "split_cifar100", "permuted_mnist", "split_tinyimagenet"]
    result_dir = Path(result_dir)

    data: dict[tuple[str, str], float] = {}
    for bench in benchmarks:
        # Load all seeds so we can find any run that has timing data
        multi = load_multiseed_results(result_dir, bench)
        for method, runs in multi.items():
            for r in runs:
                extras = r.get("per_task_extras", {})
                # Also check nested metrics.per_task_extras
                if not extras:
                    extras = r.get("metrics", {}).get("per_task_extras", {})
                total = 0.0
                for tid_str, ex in extras.items():
                    if isinstance(ex, dict):
                        total += ex.get("wall_clock_train", 0.0)
                        total += ex.get("wall_clock_after_task", 0.0)
                if total > 0:
                    data[(bench, method)] = total
                    break  # found timing for this method, no need to check more seeds

    methods_seen = sorted({m for _, m in data})
    methods_ordered = [m for m in METHOD_STYLES if m in methods_seen]

    header = " & ".join(["Method"] + [b.replace("_", " ").title()
                        .replace("Cifar", "CIFAR").replace("Mnist", "MNIST").replace("Tinyimagenet", "TinyImageNet")
                        for b in benchmarks])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Wall-clock training time (seconds) per benchmark.}",
        r"\label{tab:cost}",
        r"\begin{tabular}{l" + "r" * len(benchmarks) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for m in methods_ordered:
        label = METHOD_STYLES[m]["label"]
        vals = []
        for b in benchmarks:
            t = data.get((b, m))
            vals.append(f"{t:.0f}" if t else "---")
        lines.append(label + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex = "\n".join(lines)
    if save_path:
        Path(save_path).write_text(tex)
        print(f"Saved: {save_path}")
    return tex


# ---------------------------------------------------------------
# 9. Consolidated Results Summary Table (LaTeX)
# ---------------------------------------------------------------

def generate_summary_table(
    result_dir: str | Path,
    benchmarks: list[str] | None = None,
    save_path: str | None = None,
) -> str:
    """Generate a LaTeX table: 7 methods x N benchmarks with mean +/- std."""
    if benchmarks is None:
        benchmarks = ["split_mnist", "split_cifar10", "split_cifar100", "permuted_mnist", "split_tinyimagenet"]
    result_dir = Path(result_dir)

    # Gather data: {(bench, method): ([accs], [fgts])}
    data: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
    for bench in benchmarks:
        multi = load_multiseed_results(result_dir, bench)
        for method, runs in multi.items():
            accs = [r["metrics"]["average_accuracy"] * 100 for r in runs]
            fgts = [r["metrics"]["average_forgetting"] * 100 for r in runs]
            data[(bench, method)] = (accs, fgts)

    methods_ordered = [m for m in METHOD_STYLES if any((b, m) in data for b in benchmarks)]

    def _fmt(vals: list[float]) -> str:
        if len(vals) > 1:
            return f"${np.mean(vals):.1f} \\pm {np.std(vals):.1f}$"
        if len(vals) == 1:
            return f"${vals[0]:.1f}$"
        return "---"

    bench_labels = []
    for b in benchmarks:
        nice = b.replace("_", " ").title().replace("Cifar", "CIFAR").replace("Mnist", "MNIST").replace("Tinyimagenet", "TinyImageNet")
        bench_labels.append(nice)

    # Build header — two columns per benchmark (Acc, Fgt)
    header_parts = ["Method"]
    for label in bench_labels:
        header_parts.append(f"\\multicolumn{{2}}{{c}}{{{label}}}")
    subheader_parts = [""]
    for _ in benchmarks:
        subheader_parts.extend(["Acc (\\%)", "Fgt (\\%)"])

    ncols = 1 + 2 * len(benchmarks)
    col_spec = "l" + "rr" * len(benchmarks)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Consolidated results across all benchmarks. Accuracy (\%) $\uparrow$ and Forgetting (\%) $\downarrow$. Mean $\pm$ std shown where multiple seeds are available.}",
        r"\label{tab:summary}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Benchmark group header with cmidrule
    lines.append(" & ".join(header_parts) + r" \\")
    cmidrules = []
    for i, _ in enumerate(benchmarks):
        start = 2 + 2 * i
        end = start + 1
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrules))
    lines.append(" & ".join(subheader_parts) + r" \\")
    lines.append(r"\midrule")

    for m in methods_ordered:
        label = METHOD_STYLES[m]["label"]
        vals = [label]
        for b in benchmarks:
            if (b, m) in data:
                accs, fgts = data[(b, m)]
                vals.append(_fmt(accs))
                vals.append(_fmt(fgts))
            else:
                vals.extend(["---", "---"])
        lines.append(" & ".join(vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    tex = "\n".join(lines)
    if save_path:
        Path(save_path).write_text(tex)
        print(f"Saved: {save_path}")
    return tex


# ---------------------------------------------------------------
# 10. t-SNE of Latent Representations
# ---------------------------------------------------------------

def plot_latent_tsne(
    embeddings: np.ndarray,
    task_labels: np.ndarray,
    title: str = "Latent Space t-SNE",
    save_path: str | None = None,
):
    """Plot t-SNE of latent activations colored by task.

    Args:
        embeddings: (N, d) numpy array of latent activations.
        task_labels: (N,) integer array of task IDs.
        title: Plot title.
        save_path: Optional path to save the figure.
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(6, 5))
    unique_tasks = sorted(set(task_labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_tasks))
    for i, t in enumerate(unique_tasks):
        mask = task_labels == t
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)],
                   label=f"Task {t}", s=8, alpha=0.6)
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------
# 11. Archive Growth Over Tasks
# ---------------------------------------------------------------

def plot_archive_growth(
    result_paths: list[str | Path],
    labels: list[str] | None = None,
    save_path: str | None = None,
):
    """Plot archive rank vs. task for multiple LatentShift runs (e.g., different configs).

    Args:
        result_paths: List of result JSON paths (must be LatentShift results).
        labels: Legend labels for each run.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.get_cmap("tab10", len(result_paths))

    for i, p in enumerate(result_paths):
        data = load_result(p)
        extras = data.get("per_task_extras", {})
        if not extras:
            extras = data.get("metrics", {}).get("per_task_extras", {})
        if not extras:
            continue
        tasks = sorted(int(k) for k in extras.keys())
        ranks = [extras[str(t)].get("archive_rank", 0) for t in tasks]
        label = labels[i] if labels else Path(p).stem
        ax.plot(tasks, ranks, "o-", color=colors(i), linewidth=2, markersize=6, label=label)

    ax.set_xlabel("Task")
    ax.set_ylabel("Archive Rank")
    ax.set_title("Archive Growth Over Tasks")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------
# 12. Subspace Overlap Matrix (Task x Task)
# ---------------------------------------------------------------

def plot_subspace_overlap_matrix(
    overlap_matrix: np.ndarray,
    title: str = "Task Subspace Overlap",
    save_path: str | None = None,
):
    """Plot heatmap of pairwise task-subspace overlaps.

    Args:
        overlap_matrix: (T, T) matrix where entry (i,j) = ||V_i^T V_j||_F^2 / min(r_i, r_j).
        title: Plot title.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))
    T = overlap_matrix.shape[0]

    im = ax.imshow(overlap_matrix, vmin=0, vmax=1, cmap="YlOrRd", aspect="equal")
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xlabel("Task")
    ax.set_ylabel("Task")
    ax.set_title(title)

    for i in range(T):
        for j in range(T):
            val = overlap_matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Overlap")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------
# 13. Class-Incremental Evaluation Table (LaTeX)
# ---------------------------------------------------------------

def generate_ci_table(
    result_dir: str | Path,
    benchmarks: list[str] | None = None,
    save_path: str | None = None,
) -> str:
    """Generate a LaTeX table of class-incremental accuracy per method."""
    if benchmarks is None:
        benchmarks = ["split_cifar10", "split_cifar100"]
    result_dir = Path(result_dir)

    data: dict[tuple[str, str], float] = {}
    for bench in benchmarks:
        for p in sorted(result_dir.glob(f"*_{bench}_*_ci.json")):
            r = load_result(p)
            method = r["config"]["method"]
            ci_acc = r.get("class_incremental_accuracy",
                     r["metrics"].get("class_incremental_accuracy", 0.0))
            data[(bench, method)] = ci_acc * 100

    methods_ordered = [m for m in METHOD_STYLES if any((b, m) in data for b in benchmarks)]

    bench_labels = []
    for b in benchmarks:
        nice = b.replace("_", " ").title().replace("Cifar", "CIFAR").replace("Mnist", "MNIST").replace("Tinyimagenet", "TinyImageNet")
        bench_labels.append(nice)

    col_spec = "l" + "r" * len(benchmarks)
    header = " & ".join(["Method"] + bench_labels)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Class-incremental accuracy (\%) $\uparrow$. Single-head evaluation without task ID at inference.}",
        r"\label{tab:ci}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for m in methods_ordered:
        label = METHOD_STYLES[m]["label"]
        vals = [label]
        for b in benchmarks:
            if (b, m) in data:
                vals.append(f"${data[(b, m)]:.1f}$")
            else:
                vals.append("---")
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex = "\n".join(lines)
    if save_path:
        Path(save_path).write_text(tex)
        print(f"Saved: {save_path}")
    return tex
