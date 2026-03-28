#!/usr/bin/env bash
# Run all prompt-learning baselines: L2P, DualPrompt, CODA-Prompt
# 3 methods × 6 benchmarks × 3 seeds = 54 task-incremental
# 3 methods × 2 benchmarks × 3 seeds = 18 class-incremental
# Total: 72 runs

set +e  # continue through individual failures
cd "$(dirname "$0")"

PYTHON="$(dirname "$0")/.venv/bin/python"

SEEDS=(42 123 456)
METHODS=(l2p_vit dualprompt_vit coda_prompt_vit)
BENCHMARKS=(split_mnist permuted_mnist split_cifar10 split_cifar100 seq_cifar100 split_tinyimagenet)
CI_BENCHMARKS=(split_cifar10 split_cifar100)

TOTAL=0
DONE=0
FAILED=0

# Count total
for method in "${METHODS[@]}"; do
  for bench in "${BENCHMARKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      TOTAL=$((TOTAL + 1))
    done
  done
  for bench in "${CI_BENCHMARKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      TOTAL=$((TOTAL + 1))
    done
  done
done

echo "=================================="
echo "  Prompt Baselines: $TOTAL runs"
echo "=================================="

# ---- Task-Incremental ----
echo ""
echo "=== PART 1: Task-Incremental (54 runs) ==="

for method in "${METHODS[@]}"; do
  for bench in "${BENCHMARKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      config="configs/${method}_${bench}.yaml"
      outfile="results/${method}_${bench}_seed${seed}.json"

      if [ -f "$outfile" ]; then
        echo "[SKIP] $outfile already exists"
        DONE=$((DONE + 1))
        continue
      fi

      echo ""
      echo "[RUN $((DONE + FAILED + 1))/$TOTAL] $method $bench seed=$seed"
      if $PYTHON experiments/run_experiment.py --config "$config" --seed "$seed" --device auto; then
        DONE=$((DONE + 1))
        echo "[OK] $outfile"
      else
        FAILED=$((FAILED + 1))
        echo "[FAIL] $config seed=$seed"
      fi
    done
  done
done

# ---- Class-Incremental ----
echo ""
echo "=== PART 2: Class-Incremental (18 runs) ==="

for method in "${METHODS[@]}"; do
  for bench in "${CI_BENCHMARKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      config="configs/${method}_${bench}.yaml"
      outfile="results/${method}_${bench}_seed${seed}_ci.json"

      if [ -f "$outfile" ]; then
        echo "[SKIP] $outfile already exists"
        DONE=$((DONE + 1))
        continue
      fi

      echo ""
      echo "[RUN $((DONE + FAILED + 1))/$TOTAL] $method $bench seed=$seed --class-incremental"
      if $PYTHON experiments/run_experiment.py --config "$config" --seed "$seed" --device auto --class-incremental; then
        DONE=$((DONE + 1))
        echo "[OK] $outfile"
      else
        FAILED=$((FAILED + 1))
        echo "[FAIL] $config seed=$seed CI"
      fi
    done
  done
done

echo ""
echo "=================================="
echo "  DONE: $DONE  FAILED: $FAILED  TOTAL: $TOTAL"
echo "=================================="
