#!/bin/bash
# Re-run ResNet-18 Split-CIFAR-100 experiments (12 runs)
# These were overwritten by ViT experiments in Phase 6.
# Estimated: ~11.2h total (sequential)
set -e

PYTHON="/Users/kmagdy-ma-eg/Workspace/Research/LatentShift/.venv/bin/python"
LOGFILE="rerun_resnet_cifar100_log.txt"

run() {
    local config=$1 seed=$2
    printf '\n=== Running: %s seed=%s ===\n' "$config" "$seed" | tee -a "$LOGFILE"
    $PYTHON experiments/run_experiment.py \
        --config "configs/$config" \
        --seed "$seed" \
        --output results 2>&1 | tee -a "$LOGFILE"
    printf '=== Done: %s seed=%s ===\n' "$config" "$seed" | tee -a "$LOGFILE"
}

echo "Starting ResNet-18 Split-CIFAR-100 re-runs at $(date)" | tee "$LOGFILE"

# Naive first (fastest: ~950s/run = 47min total)
for seed in 42 123 456; do
    run naive_split_cifar100.yaml "$seed"
done

# LatentShift (~2292s/run = 1.9h total)
for seed in 42 123 456; do
    run latent_shift_split_cifar100.yaml "$seed"
done

# LatentShift-Tuned (~2968s/run = 2.5h total)
for seed in 42 123 456; do
    run latent_shift_tuned_split_cifar100.yaml "$seed"
done

# DER++ (slowest: ~7196s/run = 6h total)
for seed in 42 123 456; do
    run der_split_cifar100.yaml "$seed"
done

echo ""
echo "All 12 ResNet-18 Split-CIFAR-100 re-runs complete at $(date)" | tee -a "$LOGFILE"
