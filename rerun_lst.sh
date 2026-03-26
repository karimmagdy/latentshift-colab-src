#!/usr/bin/env bash
# Run missing LS-Tuned experiments (7 total)
set -e
cd /Users/kmagdy-ma-eg/Workspace/Research/LatentShift

# Split-MNIST seeds 123, 456
for seed in 123 456; do
    echo "===== LS-Tuned split_mnist seed=${seed} ====="
    python3 experiments/run_experiment.py --config configs/latent_shift_tuned_split_mnist.yaml --seed ${seed} --output results
    echo "DONE"
done

# Permuted-MNIST seeds 123, 456
for seed in 123 456; do
    echo "===== LS-Tuned permuted_mnist seed=${seed} ====="
    python3 experiments/run_experiment.py --config configs/latent_shift_tuned_permuted_mnist.yaml --seed ${seed} --output results
    echo "DONE"
done

# Split-TinyImageNet all 3 seeds
for seed in 42 123 456; do
    echo "===== LS-Tuned split_tinyimagenet seed=${seed} ====="
    python3 experiments/run_experiment.py --config configs/latent_shift_tuned_split_tinyimagenet.yaml --seed ${seed} --output results
    echo "DONE"
done

echo "ALL LS-TUNED EXPERIMENTS COMPLETE"
