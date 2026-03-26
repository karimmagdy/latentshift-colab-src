#!/usr/bin/env bash
# Re-run all stale DER++ experiments (12 total)
set -e
cd /Users/kmagdy-ma-eg/Workspace/Research/LatentShift

for bench in permuted_mnist split_cifar10 split_cifar100 split_tinyimagenet; do
    for seed in 42 123 456; do
        echo "===== DER++ ${bench} seed=${seed} ====="
        python3 experiments/run_experiment.py --config configs/der_${bench}.yaml --seed ${seed} --output results
        echo "DONE DER++ ${bench} seed=${seed}"
    done
done

echo "ALL DER++ EXPERIMENTS COMPLETE"
