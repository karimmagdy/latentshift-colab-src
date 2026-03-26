#!/bin/bash
# Continuation script for remaining experiments
# Picks up where run_all_fixes.sh stopped + re-runs with second HAT ResNet patch
set -e
cd "$(dirname "$0")"

RESULTS="results"
PY="python3"

run() {
    local config="$1"
    local seed="$2"
    local ci="$3"
    local ci_flag=""
    [ "$ci" = "ci" ] && ci_flag="--class-incremental"
    echo "=== Running: $config seed=$seed $ci ==="
    $PY experiments/run_experiment.py --config "configs/$config" --seed "$seed" --output "$RESULTS" $ci_flag
    echo "=== Done: $config seed=$seed $ci ==="
}

echo "============================================"
echo "Phase 3b: HAT CIFAR-100 (second patch) - remaining seeds"
echo "============================================"
# seed 42 already done with second patch
for seed in 123 456; do
    run hat_split_cifar100.yaml $seed
done
for seed in 42 123 456; do
    run hat_split_cifar100.yaml $seed ci
done

echo "============================================"
echo "Phase 4: LS/LS-Tuned CI CIFAR-100 (NCM fix)"
echo "============================================"
for seed in 42 123 456; do
    run latent_shift_split_cifar100.yaml $seed ci
done
for seed in 42 123 456; do
    run latent_shift_tuned_split_cifar100.yaml $seed ci
done

echo "============================================"
echo "Phase 5: Seq-CIFAR-100 (20 tasks x 5 classes)"
echo "============================================"
for seed in 42 123 456; do
    run naive_seq_cifar100.yaml $seed
done
for seed in 42 123 456; do
    run latent_shift_seq_cifar100.yaml $seed
done
for seed in 42 123 456; do
    run latent_shift_tuned_seq_cifar100.yaml $seed
done
for seed in 42 123 456; do
    run der_seq_cifar100.yaml $seed
done
for seed in 42 123 456; do
    run gpm_seq_cifar100.yaml $seed
done
for seed in 42 123 456; do
    run ewc_seq_cifar100.yaml $seed
done
for seed in 42 123 456; do
    run packnet_seq_cifar100.yaml $seed
done

echo "============================================"
echo "Phase 6: ViT experiments on Split-CIFAR-100"
echo "============================================"
for seed in 42 123 456; do
    run naive_vit_split_cifar100.yaml $seed
done
for seed in 42 123 456; do
    run latent_shift_vit_split_cifar100.yaml $seed
done
for seed in 42 123 456; do
    run latent_shift_tuned_vit_split_cifar100.yaml $seed
done
for seed in 42 123 456; do
    run der_vit_split_cifar100.yaml $seed
done

echo "============================================"
echo "Phase 7: HAT TinyImageNet (second patch)"
echo "============================================"
for seed in 42 123 456; do
    run hat_split_tinyimagenet.yaml $seed
done

echo ""
echo "ALL REMAINING EXPERIMENTS COMPLETE!"
