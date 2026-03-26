#!/usr/bin/env bash
set -e
cd /Users/kmagdy-ma-eg/Workspace/Research/LatentShift

SEED=$1
METHODS="naive ewc gpm trgp packnet hat latent_shift latent_shift_tuned er der"

for method in $METHODS; do
    for bench in split_cifar10 split_cifar100; do
        outfile="results/${method}_${bench}_seed${SEED}_ci.json"
        if [ -f "$outfile" ]; then
            echo "SKIP $method $bench seed$SEED (exists)"
            continue
        fi
        echo "===== $method CI $bench seed=$SEED ====="
        python3 experiments/run_experiment.py --config configs/${method}_${bench}.yaml --seed $SEED --output results --class-incremental
        echo "DONE $method $bench seed=$SEED"
    done
done
echo "=== ALL SEED $SEED CI DONE ==="
