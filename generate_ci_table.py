#!/usr/bin/env python3
"""Generate CI (Class-Incremental) table with 3-seed mean ± std."""
import json, os
import numpy as np
from collections import defaultdict

METHODS = {
    "naive": "Naive",
    "ewc": "EWC",
    "packnet": "PackNet",
    "hat": "HAT",
    "gpm": "GPM",
    "trgp": "TRGP",
    "er": "ER",
    "der": "DER++",
    "latent_shift": "LS (Ours)",
    "latent_shift_tuned": "LS-Tuned (Ours)",
}
BENCHMARKS = ["split_cifar10", "split_cifar100"]
BENCH_DISPLAY = {"split_cifar10": "Split CIFAR-10", "split_cifar100": "Split CIFAR-100"}
SEEDS = [42, 123, 456]

data = defaultdict(list)
for mk in METHODS:
    for b in BENCHMARKS:
        for s in SEEDS:
            f = os.path.join("results", f"{mk}_{b}_seed{s}_ci.json")
            if os.path.exists(f):
                with open(f) as fh:
                    d = json.load(fh)
                data[(mk, b)].append(d["class_incremental_accuracy"] * 100)

# Find best method per benchmark
best = {}
for b in BENCHMARKS:
    best_mean = -1
    for mk in METHODS:
        vals = data.get((mk, b), [])
        if vals and np.mean(vals) > best_mean:
            best_mean = np.mean(vals)
            best[b] = mk

# Print summary
print("=== Class-Incremental Accuracy (%) — 3-seed mean ± std ===\n")
print(f"{'Method':<20} {'Split CIFAR-10':>16} {'Split CIFAR-100':>16}")
print("-" * 55)
order = sorted(METHODS.keys(), key=lambda m: -np.mean(data.get((m, "split_cifar10"), [0])))
for mk in order:
    mn = METHODS[mk]
    parts = []
    for b in BENCHMARKS:
        vals = data.get((mk, b), [])
        if vals:
            parts.append(f"{np.mean(vals):5.1f} ± {np.std(vals):4.1f}")
        else:
            parts.append("---")
    print(f"{mn:<20} {parts[0]:>16} {parts[1]:>16}")

# Generate LaTeX
# Order: DER++, ER, PackNet first (replay/structured), then rest, LS methods last
display_order = ["der", "er", "packnet", "gpm", "trgp", "hat", "ewc", "naive",
                 "latent_shift", "latent_shift_tuned"]

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Class-incremental accuracy (\%) on CIFAR benchmarks (mean $\pm$ std over 3 seeds). Single-head evaluation without task identity at inference. $\uparrow$ higher is better.}")
lines.append(r"\label{tab:ci}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lcc}")
lines.append(r"\toprule")
lines.append(r"Method & Split CIFAR-10 & Split CIFAR-100 \\")
lines.append(r"\midrule")

for mk in display_order:
    mn = METHODS[mk]
    cells = []
    for b in BENCHMARKS:
        vals = data.get((mk, b), [])
        if vals:
            m = np.mean(vals)
            s = np.std(vals)
            cell = f"${m:.1f} \\pm {s:.1f}$"
            if mk == best[b]:
                cell = f"$\\mathbf{{{m:.1f}}} \\pm {s:.1f}$"
            cells.append(cell)
        else:
            cells.append("---")
    if mk == "latent_shift":
        lines.append(r"\midrule")
    line = f"{mn} & {cells[0]} & {cells[1]} \\\\"
    lines.append(line)

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

latex = "\n".join(lines)
print("\n\n=== LaTeX CI Table ===\n")
print(latex)

with open("paper/figures/ci_table.tex", "w") as f:
    f.write(latex + "\n")
print("\nSaved to paper/figures/ci_table.tex")
