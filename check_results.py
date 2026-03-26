#!/usr/bin/env python3
"""Quick check of all experiment results."""
import json, glob, os

results_dir = "results"

# DER++ results
print("=== DER++ Results ===")
for f in sorted(glob.glob(f"{results_dir}/der_*.json")):
    if "_ci" in f:
        continue
    d = json.load(open(f))
    alpha = d["config"].get("alpha", "?")
    acc = d["metrics"]["average_accuracy"]
    forg = d["metrics"]["average_forgetting"]
    tag = "FIXED" if alpha == 1.0 else "OLD"
    print(f"  {tag:5s} {os.path.basename(f):45s} acc={acc:.4f} forg={forg:.4f}")

# ER results
print("\n=== ER Results ===")
for f in sorted(glob.glob(f"{results_dir}/er_*.json")):
    if "_ci" in f:
        continue
    d = json.load(open(f))
    acc = d["metrics"]["average_accuracy"]
    forg = d["metrics"]["average_forgetting"]
    print(f"        {os.path.basename(f):45s} acc={acc:.4f} forg={forg:.4f}")

# LS-Tuned results
print("\n=== LatentShift-Tuned Results ===")
for f in sorted(glob.glob(f"{results_dir}/latent_shift_tuned_*.json")):
    d = json.load(open(f))
    acc = d["metrics"]["average_accuracy"]
    forg = d["metrics"]["average_forgetting"]
    print(f"        {os.path.basename(f):45s} acc={acc:.4f} forg={forg:.4f}")

# TRGP results
print("\n=== TRGP Results ===")
for f in sorted(glob.glob(f"{results_dir}/trgp_*.json")):
    if "_ci" in f:
        continue
    d = json.load(open(f))
    acc = d["metrics"]["average_accuracy"]
    forg = d["metrics"]["average_forgetting"]
    print(f"        {os.path.basename(f):45s} acc={acc:.4f} forg={forg:.4f}")

# Summary of what's missing
print("\n=== Missing Experiments ===")
methods_needed = {
    "der": ["split_mnist", "permuted_mnist", "split_cifar10", "split_cifar100", "split_tinyimagenet"],
    "er": ["split_mnist", "permuted_mnist", "split_cifar10", "split_cifar100", "split_tinyimagenet"],
    "latent_shift_tuned": ["split_mnist", "permuted_mnist", "split_cifar10", "split_cifar100", "split_tinyimagenet"],
    "trgp": ["split_mnist", "permuted_mnist", "split_cifar10", "split_cifar100", "split_tinyimagenet"],
}
seeds = [42, 123, 456]

for method, benchmarks in methods_needed.items():
    for bench in benchmarks:
        for seed in seeds:
            fname = f"{results_dir}/{method}_{bench}_seed{seed}.json"
            if not os.path.exists(fname):
                print(f"  MISSING: {method}_{bench}_seed{seed}")
            elif method == "der":
                d = json.load(open(fname))
                if d["config"].get("alpha") != 1.0:
                    print(f"  STALE:   {method}_{bench}_seed{seed} (old config)")
