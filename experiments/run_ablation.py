from __future__ import annotations

"""Ablation study runner for LatentShift.

Runs grid searches over hyperparameters by programmatically generating
configs and invoking the experiment runner. Results are saved as individual
JSONs plus a combined CSV summary.
"""

import argparse
import csv
import itertools
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from experiments.run_experiment import build_benchmark, build_encoder, build_method
from src.models.decoder import MultiHeadDecoder
from src.training.trainer import run_continual_learning


# ---------------------------------------------------------------
# Ablation definitions
# ---------------------------------------------------------------

ABLATIONS = {
    "latent_dim": {
        "description": "Effect of latent space dimensionality",
        "base_config": "configs/latent_shift_split_cifar10.yaml",
        "grid": {"latent_dim": [64, 128, 256, 512]},
    },
    "threshold": {
        "description": "Effect of SVD variance threshold",
        "base_config": "configs/latent_shift_split_cifar10.yaml",
        "grid": {"threshold": [0.90, 0.95, 0.99, 0.999]},
    },
    "num_samples": {
        "description": "Effect of number of activation samples for SVD",
        "base_config": "configs/latent_shift_split_cifar10.yaml",
        "grid": {"num_samples": [100, 300, 500, 1000]},
    },
    "latent_dim_mnist": {
        "description": "Latent dim ablation on Split-MNIST",
        "base_config": "configs/latent_shift_split_mnist.yaml",
        "grid": {"latent_dim": [64, 128, 256, 512]},
    },
    "threshold_mnist": {
        "description": "SVD threshold ablation on Split-MNIST",
        "base_config": "configs/latent_shift_split_mnist.yaml",
        "grid": {"threshold": [0.90, 0.95, 0.99, 0.999]},
    },
    "prune_ratio": {
        "description": "PackNet pruning ratio sensitivity",
        "base_config": "configs/packnet_split_cifar10.yaml",
        "grid": {"prune_ratio": [0.5, 0.6, 0.75, 0.9]},
    },
    "ewc_lambda": {
        "description": "EWC regularization strength",
        "base_config": "configs/ewc_split_cifar10.yaml",
        "grid": {"ewc_lambda": [10.0, 100.0, 400.0, 1000.0, 5000.0]},
    },
    "latent_dim_cifar100": {
        "description": "Latent dim ablation on Split-CIFAR-100",
        "base_config": "configs/latent_shift_split_cifar100.yaml",
        "grid": {"latent_dim": [128, 256, 512, 1024]},
    },
    "threshold_cifar100": {
        "description": "SVD threshold ablation on Split-CIFAR-100",
        "base_config": "configs/latent_shift_split_cifar100.yaml",
        "grid": {"threshold": [0.90, 0.95, 0.99, 0.999]},
    },
    "num_samples_cifar100": {
        "description": "Num samples ablation on Split-CIFAR-100",
        "base_config": "configs/latent_shift_split_cifar100.yaml",
        "grid": {"num_samples": [100, 300, 500, 1000]},
    },
}


def expand_grid(grid: dict[str, list]) -> list[dict]:
    """Expand a grid specification into a list of parameter dicts."""
    keys = sorted(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    return [dict(zip(keys, vals)) for vals in combos]


def run_single(cfg: dict, device: torch.device, seed: int) -> dict:
    """Run a single experiment with the given config dict. Returns results."""
    torch.manual_seed(seed)
    benchmark = build_benchmark(cfg)
    encoder = build_encoder(cfg).to(device)
    decoder = MultiHeadDecoder(
        latent_dim=cfg.get("latent_dim", 256),
        classes_per_task=benchmark.classes_per_task,
    ).to(device)
    method = build_method(cfg, encoder, decoder, device)

    t0 = time.time()
    metrics = run_continual_learning(
        method, benchmark,
        epochs_per_task=cfg.get("epochs", 10),
        lr=cfg.get("lr", 0.01),
        verbose=False,
    )
    elapsed = time.time() - t0

    summary = metrics.summary()
    return {
        "config": cfg,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "metrics": summary,
        "accuracy_matrix": metrics.get_accuracy_matrix().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="LatentShift Ablation Runner")
    parser.add_argument(
        "--ablation", type=str, required=True,
        choices=list(ABLATIONS.keys()),
        help="Which ablation to run",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/ablations")
    parser.add_argument(
        "--base-config", type=str, default=None,
        help="Override the base config file",
    )
    args = parser.parse_args()

    ablation = ABLATIONS[args.ablation]
    base_config_path = args.base_config or ablation["base_config"]

    # Device
    if args.device == "auto":
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    print(f"Ablation: {args.ablation} — {ablation['description']}")
    print(f"Base config: {base_config_path}")
    print(f"Device: {device}")

    # Load base config
    with open(base_config_path) as f:
        base_cfg = yaml.safe_load(f)

    # Expand grid
    param_combos = expand_grid(ablation["grid"])
    print(f"Grid: {len(param_combos)} configurations")

    # Output directory
    out_dir = Path(args.output) / args.ablation
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run all configurations
    all_results = []
    for i, params in enumerate(param_combos):
        cfg = deepcopy(base_cfg)
        cfg.update(params)

        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        print(f"\n[{i+1}/{len(param_combos)}] {param_str}")

        result = run_single(cfg, device, args.seed)
        result["ablation_params"] = params
        all_results.append(result)

        # Save individual result
        out_path = out_dir / f"{param_str}_seed{args.seed}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        m = result["metrics"]
        print(f"  Acc={m['average_accuracy']:.4f}  "
              f"Fgt={m['average_forgetting']:.4f}  "
              f"BWT={m['backward_transfer']:.4f}  "
              f"({result['elapsed_seconds']:.1f}s)")

    # Save combined CSV summary
    csv_path = out_dir / f"summary_seed{args.seed}.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = (
            sorted(ablation["grid"].keys())
            + ["average_accuracy", "average_forgetting", "backward_transfer",
               "forward_transfer", "elapsed_seconds"]
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {}
            row.update(r["ablation_params"])
            row.update(r["metrics"])
            row["elapsed_seconds"] = round(r["elapsed_seconds"], 1)
            writer.writerow(row)

    print(f"\nSummary saved to {csv_path}")
    print(f"Individual results in {out_dir}/")


if __name__ == "__main__":
    main()
