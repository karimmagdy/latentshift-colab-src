#!/bin/bash
# Lossy compression ablation on Split-CIFAR-100
# LS (d=256) with max_archive_rank = 64, 128, 192
# LS-Tuned (d=1024) with max_archive_rank = 512
# Estimated: ~8.2h total (sequential)
set -e

PYTHON="/Users/kmagdy-ma-eg/Workspace/Research/LatentShift/.venv/bin/python"
LOGFILE="run_lossy_log.txt"

run() {
    local config=$1 seed=$2
    printf '\n=== Running: %s seed=%s ===\n' "$config" "$seed" | tee -a "$LOGFILE"
    $PYTHON experiments/run_experiment.py \
        --config "configs/$config" \
        --seed "$seed" \
        --output results/lossy 2>&1 | tee -a "$LOGFILE"
    printf '=== Done: %s seed=%s ===\n' "$config" "$seed" | tee -a "$LOGFILE"
}

echo "Starting lossy compression ablation at $(date)" | tee "$LOGFILE"

# LS lossy with max_archive_rank=64 (~2300s × 3 seeds)
for seed in 42 123 456; do
    run latent_shift_lossy64_split_cifar100.yaml "$seed"
done

# LS lossy with max_archive_rank=128 (~2300s × 3 seeds)
for seed in 42 123 456; do
    run latent_shift_lossy128_split_cifar100.yaml "$seed"
done

# LS lossy with max_archive_rank=192 (~2300s × 3 seeds)
for seed in 42 123 456; do
    run latent_shift_lossy192_split_cifar100.yaml "$seed"
done

# LS-Tuned lossy with max_archive_rank=512 (~2968s × 3 seeds)
for seed in 42 123 456; do
    run latent_shift_tuned_lossy512_split_cifar100.yaml "$seed"
done

echo ""
echo "All lossy compression ablation runs complete at $(date)" | tee -a "$LOGFILE"
