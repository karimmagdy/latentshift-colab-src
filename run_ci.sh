#!/usr/bin/env bash
# Run class-incremental experiments for DER++, ER, TRGP, LS-Tuned
set -e
cd /Users/kmagdy-ma-eg/Workspace/Research/LatentShift

# DER++ CI
for bench in split_cifar10 split_cifar100; do
    echo "===== DER++ CI ${bench} ====="
    python3 experiments/run_experiment.py --config configs/der_${bench}.yaml --seed 42 --output results --class-incremental
    echo "DONE"
done

# ER CI
for bench in split_cifar10 split_cifar100; do
    echo "===== ER CI ${bench} ====="
    python3 experiments/run_experiment.py --config configs/er_${bench}.yaml --seed 42 --output results --class-incremental
    echo "DONE"
done

# TRGP CI
for bench in split_cifar10 split_cifar100; do
    echo "===== TRGP CI ${bench} ====="
    python3 experiments/run_experiment.py --config configs/trgp_${bench}.yaml --seed 42 --output results --class-incremental
    echo "DONE"
done

# LS-Tuned CI
for bench in split_cifar10 split_cifar100; do
    echo "===== LS-Tuned CI ${bench} ====="
    python3 experiments/run_experiment.py --config configs/latent_shift_tuned_${bench}.yaml --seed 42 --output results --class-incremental
    echo "DONE"
done

echo "ALL CI EXPERIMENTS COMPLETE"
