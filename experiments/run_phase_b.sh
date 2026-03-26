#!/bin/bash
# Phase B experiment batch runner
# Runs all Phase B experiments sequentially with progress tracking

set -e
cd "$(dirname "$0")/.."

SEEDS="42 123 456"
RESULTS_DIR="results"
LOG="results/phase_b_log.txt"

echo "Phase B experiments started at $(date)" | tee "$LOG"

# ---------------------------------------------------------------
# B2: DER++ remaining (Permuted-MNIST seed 123,456 + CIFAR)
# ---------------------------------------------------------------

# DER++ Permuted-MNIST (remaining seeds)
for seed in 123 456; do
    OUT="$RESULTS_DIR/der_permuted_mnist_seed${seed}.json"
    if [ -f "$OUT" ]; then echo "SKIP (exists): $OUT" | tee -a "$LOG"; continue; fi
    echo "RUN: DER++ permuted_mnist seed=$seed" | tee -a "$LOG"
    python3 experiments/run_experiment.py --config configs/der_permuted_mnist.yaml --seed $seed --output "$RESULTS_DIR" 2>&1 | tail -3 | tee -a "$LOG"
done

# DER++ CIFAR-10
for seed in $SEEDS; do
    OUT="$RESULTS_DIR/der_split_cifar10_seed${seed}.json"
    if [ -f "$OUT" ]; then echo "SKIP (exists): $OUT" | tee -a "$LOG"; continue; fi
    echo "RUN: DER++ split_cifar10 seed=$seed" | tee -a "$LOG"
    python3 experiments/run_experiment.py --config configs/der_split_cifar10.yaml --seed $seed --output "$RESULTS_DIR" 2>&1 | tail -3 | tee -a "$LOG"
done

# DER++ CIFAR-100
for seed in $SEEDS; do
    OUT="$RESULTS_DIR/der_split_cifar100_seed${seed}.json"
    if [ -f "$OUT" ]; then echo "SKIP (exists): $OUT" | tee -a "$LOG"; continue; fi
    echo "RUN: DER++ split_cifar100 seed=$seed" | tee -a "$LOG"
    python3 experiments/run_experiment.py --config configs/der_split_cifar100.yaml --seed $seed --output "$RESULTS_DIR" 2>&1 | tail -3 | tee -a "$LOG"
done

echo "B2 (DER++) complete at $(date)" | tee -a "$LOG"

# ---------------------------------------------------------------
# B1: Tuned LatentShift
# ---------------------------------------------------------------

for bench in split_cifar10 split_cifar100; do
    for seed in $SEEDS; do
        OUT="$RESULTS_DIR/latent_shift_tuned_${bench}_seed${seed}.json"
        if [ -f "$OUT" ]; then echo "SKIP (exists): $OUT" | tee -a "$LOG"; continue; fi
        echo "RUN: Tuned LS $bench seed=$seed" | tee -a "$LOG"
        python3 experiments/run_experiment.py --config "configs/latent_shift_tuned_${bench}.yaml" --seed $seed --output "$RESULTS_DIR" 2>&1 | tail -3 | tee -a "$LOG"
    done
done

echo "B1 (Tuned LS) complete at $(date)" | tee -a "$LOG"

# ---------------------------------------------------------------
# B4: GPM-last-layer ablation
# ---------------------------------------------------------------

for bench in split_cifar10 split_cifar100; do
    for seed in $SEEDS; do
        OUT="$RESULTS_DIR/gpm_lastlayer_${bench}_seed${seed}.json"
        if [ -f "$OUT" ]; then echo "SKIP (exists): $OUT" | tee -a "$LOG"; continue; fi
        echo "RUN: GPM-LL $bench seed=$seed" | tee -a "$LOG"
        python3 experiments/run_experiment.py --config "configs/gpm_lastlayer_${bench}.yaml" --seed $seed --output "$RESULTS_DIR" 2>&1 | tail -3 | tee -a "$LOG"
    done
done

echo "B4 (GPM-LL) complete at $(date)" | tee -a "$LOG"

# ---------------------------------------------------------------
# B5: Class-incremental evaluation (seed 42 first)
# ---------------------------------------------------------------

for method in naive ewc gpm packnet hat er der latent_shift; do
    for bench in split_cifar10 split_cifar100; do
        OUT="$RESULTS_DIR/${method}_${bench}_seed42_ci.json"
        if [ -f "$OUT" ]; then echo "SKIP (exists): $OUT" | tee -a "$LOG"; continue; fi
        echo "RUN: CI $method $bench seed=42" | tee -a "$LOG"
        python3 experiments/run_experiment.py --config "configs/${method}_${bench}.yaml" --seed 42 --output "$RESULTS_DIR" --class-incremental 2>&1 | tail -3 | tee -a "$LOG"
    done
done

echo "B5 (CI) complete at $(date)" | tee -a "$LOG"

# ---------------------------------------------------------------
# B3: TinyImageNet — all 8 methods × 3 seeds
# ---------------------------------------------------------------

for method in naive ewc gpm packnet hat er der latent_shift; do
    for seed in $SEEDS; do
        OUT="$RESULTS_DIR/${method}_split_tinyimagenet_seed${seed}.json"
        if [ -f "$OUT" ]; then echo "SKIP (exists): $OUT" | tee -a "$LOG"; continue; fi
        echo "RUN: TinyImageNet $method seed=$seed" | tee -a "$LOG"
        python3 experiments/run_experiment.py --config "configs/${method}_split_tinyimagenet.yaml" --seed $seed --output "$RESULTS_DIR" 2>&1 | tail -3 | tee -a "$LOG"
    done
done

echo "B3 (TinyImageNet) complete at $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Phase B ALL COMPLETE at $(date) ===" | tee -a "$LOG"
echo "Total result files:" | tee -a "$LOG"
ls "$RESULTS_DIR"/*.json | wc -l | tee -a "$LOG"
