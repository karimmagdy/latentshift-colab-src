#!/bin/bash
# Run missing GPM-LL experiments
set -e

CONFIGS=(
  "configs/gpm_lastlayer_split_mnist.yaml"
  "configs/gpm_lastlayer_permuted_mnist.yaml"
  "configs/gpm_lastlayer_split_tinyimagenet.yaml"
)
SEEDS=(42 123 456)

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    name=$(basename "$cfg" .yaml)
    result="results/${name}_seed${seed}.json"
    if [ -f "$result" ]; then
      echo "SKIP: $result exists"
      continue
    fi
    echo "RUNNING: $cfg seed=$seed"
    python3 experiments/run_experiment.py --config "$cfg" --seed "$seed" --output results
    echo "DONE: $cfg seed=$seed"
  done
done
echo "ALL GPM-LL DONE"
