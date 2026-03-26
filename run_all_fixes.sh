#!/bin/bash
# Master experiment run script
# Priority: HAT bug fix > LS CI fix > Seq-CIFAR-100 > ViT experiments
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
echo "Phase 1: HAT re-runs (bug fix) - MNIST fast"
echo "============================================"
for seed in 42 123 456; do
    run hat_split_mnist.yaml $seed
done
for seed in 42 123 456; do
    run hat_permuted_mnist.yaml $seed
done

echo "============================================"
echo "Phase 2: LS/LS-Tuned CI re-runs (NCM fix)"
echo "============================================"
for seed in 42 123 456; do
    run latent_shift_split_cifar10.yaml $seed ci
done
for seed in 42 123 456; do
    run latent_shift_tuned_split_cifar10.yaml $seed ci
done

echo "============================================"
echo "Phase 3: HAT CIFAR-10/100 + CI"  
echo "============================================"
for seed in 42 123 456; do
    run hat_split_cifar10.yaml $seed
done
for seed in 42 123 456; do
    run hat_split_cifar10.yaml $seed ci
done
for seed in 42 123 456; do
    run hat_split_cifar100.yaml $seed
done
for seed in 42 123 456; do
    run hat_split_cifar100.yaml $seed ci
done

echo "============================================"
echo "Phase 4: LS/LS-Tuned CI CIFAR-100"
echo "============================================"
for seed in 42 123 456; do
    run latent_shift_split_cifar100.yaml $seed ci
done
for seed in 42 123 456; do
    run latent_shift_tuned_split_cifar100.yaml $seed ci
done

echo "============================================"
echo "Phase 5: Seq-CIFAR-100 (20 tasks × 5 classes)"
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
echo "Phase 7: HAT TinyImageNet (slowest)"
echo "============================================"
for seed in 42 123 456; do
    run hat_split_tinyimagenet.yaml $seed
done

echo ""
echo "ALL EXPERIMENTS COMPLETE!"
