#!/usr/bin/env python3
"""Compute BWT (Backward Transfer) from accuracy matrices in result files.

BWT = (1/(T-1)) * sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})
where R_{t,i} is task i accuracy after training on task t.

Negative BWT = forgetting. Positive BWT = backward transfer.
"""
import json, glob, os
import numpy as np
from collections import defaultdict

RESULTS_DIR = "results"
BENCHMARKS = ["split_mnist", "permuted_mnist", "split_cifar10", "split_cifar100", "split_tinyimagenet"]
METHODS = {
    "naive": "Naive",
    "ewc": "EWC",
    "packnet": "PackNet",
    "hat": "HAT",
    "gpm": "GPM",
    "gpm_lastlayer": "GPM-LL",
    "trgp": "TRGP",
    "er": "ER",
    "der": "DER++",
    "latent_shift": "LS",
    "latent_shift_tuned": "LS-Tuned",
}
SEEDS = [42, 123, 456]

def compute_bwt(accuracy_matrix):
    """Compute BWT from accuracy matrix."""
    T = len(accuracy_matrix)
    if T < 2:
        return 0.0
    bwt = 0.0
    for i in range(T - 1):
        bwt += accuracy_matrix[T-1][i] - accuracy_matrix[i][i]
    return bwt / (T - 1)

def compute_fwt(accuracy_matrix, num_classes_per_task):
    """Compute FWT. Uses chance accuracy as baseline."""
    T = len(accuracy_matrix)
    if T < 2:
        return 0.0
    chance = 1.0 / num_classes_per_task
    fwt = 0.0
    count = 0
    for i in range(1, T):
        # R_{i-1, i} is accuracy on task i after training tasks 0..i-1
        # But our matrix has 0 for unseen tasks. Before training task i,
        # the zeroshot performance would need to be measured separately.
        # We'll skip FWT if we don't have this data.
        if accuracy_matrix[i-1][i] > 0:
            fwt += accuracy_matrix[i-1][i] - chance
            count += 1
    return fwt / count if count > 0 else None

results = defaultdict(lambda: defaultdict(list))

for method_key, method_name in METHODS.items():
    for bench in BENCHMARKS:
        for seed in SEEDS:
            fname = os.path.join(RESULTS_DIR, f"{method_key}_{bench}_seed{seed}.json")
            if not os.path.exists(fname):
                continue
            with open(fname) as f:
                data = json.load(f)
            if "accuracy_matrix" not in data:
                continue
            mat = data["accuracy_matrix"]
            bwt = compute_bwt(mat)
            results[(method_key, bench)]["bwt"].append(bwt * 100)  # Convert to %

# Print BWT table
print("\n=== BWT (Backward Transfer) — lower magnitude = less forgetting ===")
print(f"{'Method':<12}", end="")
for b in BENCHMARKS:
    short = b.replace("split_", "S-").replace("permuted_", "P-")
    print(f"  {short:>14}", end="")
print()
print("-" * 85)

for method_key, method_name in METHODS.items():
    print(f"{method_name:<12}", end="")
    for bench in BENCHMARKS:
        vals = results[(method_key, bench)]["bwt"]
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            print(f"  {mean:>+6.1f}±{std:4.1f}  ", end="")
        else:
            print(f"  {'---':>14}", end="")
    print()

# Generate LaTeX table
print("\n\n=== LaTeX BWT Table ===\n")
print(r"\begin{table}[t]")
print(r"\centering")
print(r"\caption{Backward Transfer (BWT, \%) across benchmarks (mean $\pm$ std over 3 seeds). Higher is better; negative values indicate forgetting.}")
print(r"\label{tab:bwt}")
print(r"\small")
print(r"\begin{tabular}{l" + "c" * len(BENCHMARKS) + "}")
print(r"\toprule")
header = "Method"
for b in BENCHMARKS:
    short = b.replace("split_", "S-").replace("permuted_", "P-").replace("_", " ")
    header += f" & {short}"
header += r" \\"
print(header)
print(r"\midrule")

for method_key, method_name in METHODS.items():
    line = method_name
    for bench in BENCHMARKS:
        vals = results[(method_key, bench)]["bwt"]
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            line += f" & ${mean:+.1f} \\pm {std:.1f}$"
        else:
            line += " & ---"
    line += r" \\"
    print(line)

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

# Save to file
with open("paper/figures/bwt_table.tex", "w") as f:
    f.write(r"\begin{table}[t]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\caption{Backward Transfer (BWT, \%) across benchmarks (mean $\pm$ std over 3 seeds). Higher is better; negative values indicate forgetting.}" + "\n")
    f.write(r"\label{tab:bwt}" + "\n")
    f.write(r"\small" + "\n")
    f.write(r"\begin{tabular}{l" + "c" * len(BENCHMARKS) + "}\n")
    f.write(r"\toprule" + "\n")
    header = "Method"
    for b in BENCHMARKS:
        short = b.replace("split_", "S-").replace("permuted_", "P-")
        header += f" & {short}"
    header += r" \\" + "\n"
    f.write(header)
    f.write(r"\midrule" + "\n")
    for method_key, method_name in METHODS.items():
        line = method_name
        for bench in BENCHMARKS:
            vals = results[(method_key, bench)]["bwt"]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals)
                line += f" & ${mean:+.1f} \\pm {std:.1f}$"
            else:
                line += " & ---"
        line += r" \\" + "\n"
        f.write(line)
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}" + "\n")
    f.write(r"\end{table}" + "\n")

print("\nSaved to paper/figures/bwt_table.tex")
