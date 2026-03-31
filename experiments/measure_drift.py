from __future__ import annotations

"""Measure representation drift in LatentShift and compare to Proposition 5 bound.

Runs LatentShift on Split-CIFAR-100 across 3 seeds, measures the actual
squared drift ||z_new - z_old||^2 of archived representations after each
task, and compares it to the theoretical O(eta^2 * G^2 * N^2) bound from
Proposition 5.

Saves:
    results/drift_analysis.json   — raw drift and bound data per seed/task
    paper/figures/drift_vs_bound.pdf — drift vs. bound figure

Usage:
    python experiments/measure_drift.py
    python experiments/measure_drift.py --config configs/latent_shift_split_cifar100.yaml
    python experiments/measure_drift.py --device cuda --seeds 42 123 456
"""

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from experiments.run_experiment import build_benchmark, build_encoder, build_method
from src.models.decoder import MultiHeadDecoder
from src.utils.metrics import ContinualMetrics

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


# ---------------------------------------------------------------
# Drift measurement helpers
# ---------------------------------------------------------------

@torch.no_grad()
def collect_representations(encoder, loader, device, max_samples=300):
    """Collect latent representations for a set of samples."""
    encoder.eval()
    zs, xs_store = [], []
    count = 0
    for x, _ in loader:
        x = x.to(device)
        z = encoder(x)
        zs.append(z.cpu())
        xs_store.append(x.cpu())
        count += x.size(0)
        if count >= max_samples:
            break
    encoder.train()
    return torch.cat(zs, dim=0)[:max_samples], torch.cat(xs_store, dim=0)[:max_samples]


@torch.no_grad()
def recompute_representations(encoder, xs, device):
    """Re-encode stored inputs to get updated representations."""
    encoder.eval()
    zs = []
    # Process in batches to avoid OOM
    batch_size = 64
    for i in range(0, xs.size(0), batch_size):
        x_batch = xs[i:i + batch_size].to(device)
        z = encoder(x_batch)
        zs.append(z.cpu())
    encoder.train()
    return torch.cat(zs, dim=0)


def compute_drift(z_old: torch.Tensor, z_new: torch.Tensor) -> float:
    """Compute mean squared drift ||z_new - z_old||^2 per sample."""
    diff = z_new - z_old
    return float((diff ** 2).sum(dim=1).mean().item())


def proposition5_bound(eta: float, G: float, N: int) -> float:
    """Compute the Proposition 5 drift bound: C * eta^2 * G^2 * N^2.

    This is the theoretical upper bound on representation drift for
    archived representations under gradient projection.

    Args:
        eta: Learning rate.
        G: Estimated gradient norm upper bound.
        N: Number of gradient steps (epochs * batches).

    Returns:
        The bound value (with C=1 as a scaling constant).
    """
    return eta ** 2 * G ** 2 * N ** 2


# ---------------------------------------------------------------
# Gradient norm estimation
# ---------------------------------------------------------------

def estimate_gradient_norm(encoder, decoder, loader, task_id, device, max_batches=20):
    """Estimate the average gradient norm during training on a task.

    Runs a few forward-backward passes and records ||grad||.
    """
    encoder.train()
    decoder.train()
    criterion = torch.nn.CrossEntropyLoss()
    norms = []

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        z = encoder(x)
        logits = decoder(z, task_id)
        loss = criterion(logits, y)
        loss.backward()

        total_norm = 0.0
        for p in encoder.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        norms.append(total_norm ** 0.5)

        encoder.zero_grad()
        decoder.zero_grad()

    return float(np.mean(norms)) if norms else 1.0


# ---------------------------------------------------------------
# Single-seed drift experiment
# ---------------------------------------------------------------

def run_drift_experiment(cfg: dict, device: torch.device, seed: int) -> dict:
    """Run LatentShift on Split-CIFAR-100 and measure drift after each task."""
    torch.manual_seed(seed)
    benchmark = build_benchmark(cfg)
    encoder = build_encoder(cfg).to(device)
    decoder = MultiHeadDecoder(
        latent_dim=cfg.get("latent_dim", 256),
        classes_per_task=benchmark.classes_per_task,
    ).to(device)
    method = build_method(cfg, encoder, decoder, device)

    lr = cfg.get("lr", 0.01)
    epochs = cfg.get("epochs", 20)

    # Storage for per-task archived representations
    # archived_data[t] = (xs, z_old) — inputs and representations from task t
    archived_data: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    # Results per task
    task_drifts: dict[int, dict] = {}  # task_id -> {drift for each archived task}
    task_bounds: dict[int, float] = {}
    task_grad_norms: dict[int, float] = {}

    for task_id in range(benchmark.num_tasks):
        print(f"\n  [seed={seed}] Task {task_id}/{benchmark.num_tasks - 1}")
        train_loader, test_loader = benchmark.get_task_loaders(task_id)

        # Prepare
        method.prepare_task(task_id, train_loader)

        # Estimate gradient norm before training (for bound computation)
        G = estimate_gradient_norm(encoder, decoder, train_loader, task_id, device)
        task_grad_norms[task_id] = G

        # Count total gradient steps for bound
        n_batches = len(train_loader)
        N = epochs * n_batches

        # Train
        method.train_task(task_id, train_loader, epochs, lr)

        # After task: update subspace archive
        method.after_task(task_id, train_loader)

        # Measure drift on all previously archived tasks
        drifts = {}
        for prev_t, (xs_prev, z_old) in archived_data.items():
            z_new = recompute_representations(encoder, xs_prev, device)
            drift = compute_drift(z_old, z_new)
            drifts[prev_t] = drift
            print(f"    Drift on task {prev_t}: {drift:.6f}")

        task_drifts[task_id] = drifts

        # Compute theoretical bound
        bound = proposition5_bound(lr, G, N)
        task_bounds[task_id] = bound
        print(f"    Bound (eta={lr}, G={G:.2f}, N={N}): {bound:.6f}")

        # Archive current task representations (store inputs + z)
        z_current, xs_current = collect_representations(
            encoder, train_loader, device, max_samples=300
        )
        archived_data[task_id] = (xs_current, z_current)

    # Aggregate: for each task t, compute mean drift across all archived tasks
    mean_drifts = []
    bounds_list = []
    for t in range(benchmark.num_tasks):
        d = task_drifts[t]
        if d:
            mean_drifts.append(float(np.mean(list(d.values()))))
        else:
            mean_drifts.append(0.0)
        bounds_list.append(task_bounds[t])

    return {
        "seed": seed,
        "num_tasks": benchmark.num_tasks,
        "mean_drifts": mean_drifts,
        "bounds": bounds_list,
        "per_task_drifts": {str(k): v for k, v in task_drifts.items()},
        "grad_norms": {str(k): v for k, v in task_grad_norms.items()},
        "lr": lr,
        "epochs": epochs,
    }


# ---------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------

def plot_drift_vs_bound(all_results: list[dict], save_path: str | None = None):
    """Plot actual drift vs. Proposition 5 bound across tasks, averaged over seeds."""
    num_tasks = all_results[0]["num_tasks"]
    tasks = np.arange(num_tasks)

    # Stack drifts and bounds across seeds
    all_drifts = np.array([r["mean_drifts"] for r in all_results])
    all_bounds = np.array([r["bounds"] for r in all_results])

    drift_mean = all_drifts.mean(axis=0)
    drift_std = all_drifts.std(axis=0)
    bound_mean = all_bounds.mean(axis=0)
    bound_std = all_bounds.std(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Actual drift
    ax.plot(tasks, drift_mean, "o-", color="#2196F3", linewidth=2, markersize=6,
            label="Actual Drift $\\|\\mathbf{z}_{\\mathrm{new}} - \\mathbf{z}_{\\mathrm{old}}\\|^2$")
    ax.fill_between(tasks, drift_mean - drift_std, drift_mean + drift_std,
                     alpha=0.15, color="#2196F3")

    # Theoretical bound
    ax.plot(tasks, bound_mean, "s--", color="#F44336", linewidth=2, markersize=6,
            label="Proposition 5 Bound $O(\\eta^2 G^2 N^2)$")
    ax.fill_between(tasks, bound_mean - bound_std, bound_mean + bound_std,
                     alpha=0.15, color="#F44336")

    ax.set_xlabel("Task")
    ax.set_ylabel("Representation Drift")
    ax.set_title("Representation Drift vs. Theoretical Bound — Split CIFAR-100")
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Measure representation drift in LatentShift")
    parser.add_argument("--config", type=str,
                        default="configs/latent_shift_split_cifar100.yaml",
                        help="Path to YAML config")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds to run (default: 3 seeds)")
    parser.add_argument("--output", type=str, default="results",
                        help="Directory for result JSONs")
    parser.add_argument("--figure-dir", type=str, default="paper/figures",
                        help="Directory for output figures")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Drift Measurement Experiment")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Benchmark: {cfg['benchmark']}")

    # Run across seeds
    all_results = []
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        t0 = time.time()
        result = run_drift_experiment(cfg, device, seed)
        result["elapsed_seconds"] = time.time() - t0
        all_results.append(result)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "drift_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Generate figure
    figure_path = Path(args.figure_dir) / "drift_vs_bound.pdf"
    plot_drift_vs_bound(all_results, save_path=str(figure_path))

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for result in all_results:
        seed = result["seed"]
        drifts = result["mean_drifts"]
        bounds = result["bounds"]
        max_drift = max(drifts)
        max_bound = max(bounds)
        print(f"  Seed {seed}: max_drift={max_drift:.6f}, max_bound={max_bound:.6f}, "
              f"ratio={max_drift / max_bound:.4f}" if max_bound > 0 else
              f"  Seed {seed}: max_drift={max_drift:.6f}, max_bound={max_bound:.6f}")


if __name__ == "__main__":
    main()
