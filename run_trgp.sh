#!/usr/bin/env bash
# Run all TRGP experiments: 5 benchmarks × 3 seeds = 15
set -e
cd /Users/kmagdy-ma-eg/Workspace/Research/LatentShift

for bench in split_mnist permuted_mnist split_cifar10 split_cifar100 split_tinyimagenet; do
    for seed in 42 123 456; do
        echo "===== TRGP ${bench} seed=${seed} ====="
        python3 experiments/run_experiment.py --config configs/trgp_${bench}.yaml --seed ${seed} --output results
        echo "DONE TRGP ${bench} seed=${seed}"
    done
done

echo "ALL TRGP EXPERIMENTS COMPLETE"
