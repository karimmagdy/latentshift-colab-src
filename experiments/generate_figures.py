from __future__ import annotations

"""Generate all paper figures from experiment results.

Usage:
    python experiments/generate_figures.py --results results --output paper/figures
    python experiments/generate_figures.py --results results --benchmark split_mnist
    python experiments/generate_figures.py --ablation results/ablations/latent_dim --param latent_dim
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualization.plots import (
    load_results,
    load_multiseed_results,
    plot_accuracy_heatmaps,
    plot_accuracy_over_tasks,
    plot_subspace_growth,
    plot_per_task_forgetting,
    plot_summary_bars,
    plot_multiseed_bars,
    plot_ablation,
    generate_cost_table,
    generate_summary_table,
    generate_ci_table,
)


BENCHMARKS = ["split_mnist", "split_cifar10", "split_cifar100", "permuted_mnist", "split_tinyimagenet"]


def generate_benchmark_figures(
    result_dir: Path, output_dir: Path, benchmark: str
) -> None:
    """Generate all figures for a single benchmark."""
    results = load_results(result_dir, benchmark)
    if not results:
        print(f"  No results found for {benchmark}, skipping.")
        return

    bench_dir = output_dir / benchmark
    bench_dir.mkdir(parents=True, exist_ok=True)

    nice_name = benchmark.replace("_", " ").title().replace("Cifar", "CIFAR").replace("Mnist", "MNIST").replace("Tinyimagenet", "TinyImageNet")
    print(f"  Generating figures for {nice_name} ({len(results)} methods)...")

    plot_accuracy_heatmaps(
        results, nice_name, save_path=str(bench_dir / "accuracy_heatmaps.pdf")
    )
    plot_accuracy_over_tasks(
        results, nice_name, save_path=str(bench_dir / "accuracy_over_tasks.pdf")
    )
    plot_subspace_growth(
        results, nice_name, save_path=str(bench_dir / "subspace_growth.pdf")
    )
    plot_per_task_forgetting(
        results, nice_name, save_path=str(bench_dir / "per_task_forgetting.pdf")
    )
    plot_summary_bars(
        results, nice_name, save_path=str(bench_dir / "summary_bars.pdf")
    )


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results", type=str, default="results",
                        help="Directory with result JSONs")
    parser.add_argument("--output", type=str, default="paper/figures",
                        help="Directory to save figures")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Generate figures for a specific benchmark only")
    parser.add_argument("--ablation", type=str, default=None,
                        help="Path to ablation results directory")
    parser.add_argument("--param", type=str, default=None,
                        help="Ablation parameter name (for --ablation)")
    args = parser.parse_args()

    result_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.ablation:
        # Generate ablation figure
        param = args.param
        if not param:
            param = Path(args.ablation).name
        print(f"Generating ablation figure: {param}")
        ablation_dir = output_dir / "ablations"
        ablation_dir.mkdir(parents=True, exist_ok=True)

        for metric in ["average_accuracy", "average_forgetting"]:
            plot_ablation(
                args.ablation, param, metric=metric,
                save_path=str(ablation_dir / f"ablation_{param}_{metric}.pdf"),
            )
    else:
        # Generate benchmark comparison figures
        benchmarks = [args.benchmark] if args.benchmark else BENCHMARKS
        for benchmark in benchmarks:
            generate_benchmark_figures(result_dir, output_dir, benchmark)

        # Multi-seed figures (if multiple seeds available)
        if not args.benchmark:
            multiseed_dir = output_dir / "multiseed"
            multiseed_dir.mkdir(parents=True, exist_ok=True)
            for benchmark in BENCHMARKS:
                multi = load_multiseed_results(result_dir, benchmark)
                # Only plot if at least one method has >1 seed
                if any(len(v) > 1 for v in multi.values()):
                    nice = benchmark.replace("_", " ").title().replace("Cifar", "CIFAR").replace("Mnist", "MNIST").replace("Tinyimagenet", "TinyImageNet")
                    print(f"  Multi-seed figure for {nice}...")
                    plot_multiseed_bars(
                        multi, nice,
                        save_path=str(multiseed_dir / f"multiseed_{benchmark}.pdf"),
                    )

            # Cost table
            print("  Generating cost table...")
            generate_cost_table(
                result_dir,
                save_path=str(output_dir / "cost_table.tex"),
            )

            # Consolidated summary table
            print("  Generating summary table...")
            generate_summary_table(
                result_dir,
                save_path=str(output_dir / "summary_table.tex"),
            )

            # Class-incremental evaluation table
            print("  Generating CI table...")
            generate_ci_table(
                result_dir,
                save_path=str(output_dir / "ci_table.tex"),
            )

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
