#!/usr/bin/env bash
# Re-run all ER experiments with normalization fix (15 total)
set -e
cd /Users/kmagdy-ma-eg/Workspace/Research/LatentShift

for bench in split_mnist permuted_mnist split_cifar10 split_cifar100 split_tinyimagenet; do
    for seed in 42 123 456; do
        echo "===== ER ${bench} seed=${seed} ====="
        python3 experiments/run_experiment.py --config configs/er_${bench}.yaml --seed ${seed} --output results
        echo "DONE ER ${bench} seed=${seed}"
    done
done

echo "ALL ER EXPERIMENTS COMPLETE"
