#!/bin/bash
# Re-run TRGP experiments that used OLD code (no trust_alpha)
# Only re-runs the ones from before the trust_alpha fix
set -e

cd /Users/kmagdy-ma-eg/Workspace/Research/LatentShift

# These were run before trust_alpha was added:
# split_mnist × 3 seeds
# permuted_mnist × 3 seeds
# split_cifar10 s42, s123

CONFIGS_SEEDS=(
  "configs/trgp_split_mnist.yaml 42"
  "configs/trgp_split_mnist.yaml 123"
  "configs/trgp_split_mnist.yaml 456"
  "configs/trgp_permuted_mnist.yaml 42"
  "configs/trgp_permuted_mnist.yaml 123"
  "configs/trgp_permuted_mnist.yaml 456"
  "configs/trgp_split_cifar10.yaml 42"
  "configs/trgp_split_cifar10.yaml 123"
  "configs/trgp_split_cifar10.yaml 456"
)

for entry in "${CONFIGS_SEEDS[@]}"; do
  cfg=$(echo "$entry" | awk '{print $1}')
  seed=$(echo "$entry" | awk '{print $2}')
  name=$(basename "$cfg" .yaml)
  echo "RUNNING: $name seed=$seed"
  python3 experiments/run_experiment.py --config "$cfg" --seed "$seed" --output results
  echo "DONE: $name seed=$seed"
done

echo "ALL TRGP RE-RUNS COMPLETE"
