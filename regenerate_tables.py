#!/usr/bin/env python3
"""Regenerate ALL paper tables from result JSON files.

Run this after ALL experiments have completed to ensure tables match data.
Cross-checks encoder types to prevent architecture mix-ups.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("paper/figures")

METHODS_TI = {
    "naive": "Naive",
    "ewc": "EWC",
    "gpm": "GPM",
    "gpm_lastlayer": "GPM (Last Layer)",
    "trgp": "TRGP",
    "packnet": "PackNet",
    "hat": "HAT",
    "er": "ER",
    "der": "DER++",
    "latent_shift": "LatentShift (Ours)",
    "latent_shift_tuned": "LatentShift-Tuned (Ours)",
}

BENCHMARKS_TI = [
    "split_mnist", "split_cifar10", "split_cifar100",
    "permuted_mnist", "split_tinyimagenet",
]
BENCH_DISPLAY = {
    "split_mnist": "Split MNIST",
    "split_cifar10": "Split CIFAR10",
    "split_cifar100": "Split CIFAR100",
    "permuted_mnist": "Permuted MNIST",
    "split_tinyimagenet": "Split TinyImageNet",
}

SEEDS = [42, 123, 456]

# Expected default encoders per benchmark
EXPECTED_ENCODER = {
    "split_mnist": {"mlp"},
    "permuted_mnist": {"mlp"},
    "split_cifar10": {"resnet18"},
    "split_cifar100": {"resnet18"},
    "split_tinyimagenet": {"resnet18", "hat_resnet18"},
}


def load_result(method, benchmark, seed, suffix=""):
    path = RESULTS_DIR / f"{method}_{benchmark}_seed{seed}{suffix}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def validate_encoder(result, method, benchmark, seed):
    """Check that the encoder matches expectations."""
    enc = result.get("config", {}).get("encoder", "unknown")
    expected = EXPECTED_ENCODER.get(benchmark, set())
    if expected and enc not in expected:
        print(f"WARNING: {method}/{benchmark}/seed{seed} has encoder={enc}, "
              f"expected one of {expected}", file=sys.stderr)
        return False
    return True


def collect_metrics(methods, benchmarks, seeds, suffix=""):
    """Collect accuracy, forgetting, BWT for all method/benchmark combos."""
    data = {}
    missing = []
    warnings = []
    for mk in methods:
        for bk in benchmarks:
            accs, fgts, bwts, times = [], [], [], []
            for s in seeds:
                r = load_result(mk, bk, s, suffix)
                if r is None:
                    missing.append(f"{mk}/{bk}/seed{s}")
                    continue
                if not validate_encoder(r, mk, bk, s):
                    warnings.append(f"{mk}/{bk}/seed{s}")
                m = r["metrics"]
                accs.append(m["average_accuracy"] * 100)
                fgts.append(m["average_forgetting"] * 100)
                bwts.append(m["backward_transfer"] * 100)
                times.append(r.get("elapsed_seconds", 0))
            if accs:
                data[(mk, bk)] = {
                    "acc_mean": np.mean(accs), "acc_std": np.std(accs),
                    "fgt_mean": np.mean(fgts), "fgt_std": np.std(fgts),
                    "bwt_mean": np.mean(bwts), "bwt_std": np.std(bwts),
                    "time_mean": np.mean(times),
                    "n": len(accs),
                }
    return data, missing, warnings


def fmt(mean, std, bold=False):
    if bold:
        return f"$\\mathbf{{{mean:.1f}}} \\pm {std:.1f}$"
    return f"${mean:.1f} \\pm {std:.1f}$"


# =================================================================
# Collect all data
# =================================================================
print("Collecting task-incremental results...")
data, missing, warnings = collect_metrics(METHODS_TI, BENCHMARKS_TI, SEEDS)

if missing:
    print(f"\nMISSING ({len(missing)}):")
    for m in sorted(missing):
        print(f"  {m}")
if warnings:
    print(f"\nWARNINGS ({len(warnings)}):")
    for w in sorted(warnings):
        print(f"  {w}")

# =================================================================
# 1. Summary Table (Acc + Fgt)
# =================================================================
print("\n=== Generating summary_table.tex ===")
display_order = ["latent_shift", "latent_shift_tuned", "der", "er",
                 "gpm", "trgp", "gpm_lastlayer", "ewc", "packnet", "hat", "naive"]

lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Consolidated results across all benchmarks. Accuracy (\%) $\uparrow$ and Forgetting (\%) $\downarrow$. Mean $\pm$ std shown where multiple seeds are available.}")
lines.append(r"\label{tab:summary}")
lines.append(r"\begin{tabular}{lrrrrrrrrrr}")
lines.append(r"\toprule")
lines.append(r"Method & \multicolumn{2}{c}{Split MNIST} & \multicolumn{2}{c}{Split CIFAR10} & \multicolumn{2}{c}{Split CIFAR100} & \multicolumn{2}{c}{Permuted MNIST} & \multicolumn{2}{c}{Split TinyImageNet} \\")
lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}")
lines.append(r" & Acc (\%) & Fgt (\%) & Acc (\%) & Fgt (\%) & Acc (\%) & Fgt (\%) & Acc (\%) & Fgt (\%) & Acc (\%) & Fgt (\%) \\")
lines.append(r"\midrule")

for mk in display_order:
    mn = METHODS_TI[mk]
    cells = []
    for bk in BENCHMARKS_TI:
        d = data.get((mk, bk))
        if d:
            cells.append(fmt(d["acc_mean"], d["acc_std"]))
            cells.append(fmt(d["fgt_mean"], d["fgt_std"]))
        else:
            cells.extend(["---", "---"])
    lines.append(f"{mn} & {' & '.join(cells)} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table*}")

tex = "\n".join(lines)
(FIGURES_DIR / "summary_table.tex").write_text(tex)
print("  Written to paper/figures/summary_table.tex")

# =================================================================
# 2. BWT Table
# =================================================================
print("\n=== Generating bwt_table.tex ===")
bwt_display = ["naive", "ewc", "packnet", "hat", "gpm", "gpm_lastlayer",
               "trgp", "er", "der", "latent_shift", "latent_shift_tuned"]
bwt_names = {
    "naive": "Naive", "ewc": "EWC", "packnet": "PackNet", "hat": "HAT",
    "gpm": "GPM", "gpm_lastlayer": "GPM-LL", "trgp": "TRGP",
    "er": "ER", "der": "DER++",
    "latent_shift": "LS", "latent_shift_tuned": "LS-Tuned",
}
bench_short = {
    "split_mnist": "S-mnist", "permuted_mnist": "P-mnist",
    "split_cifar10": "S-cifar10", "split_cifar100": "S-cifar100",
    "split_tinyimagenet": "S-tinyimagenet",
}
bwt_bench_order = ["split_mnist", "permuted_mnist", "split_cifar10",
                   "split_cifar100", "split_tinyimagenet"]

# Find best BWT per benchmark
best_bwt = {}
for bk in bwt_bench_order:
    best = -999
    for mk in bwt_display:
        d = data.get((mk, bk))
        if d and d["bwt_mean"] > best:
            best = d["bwt_mean"]
            best_bwt[bk] = mk

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Backward Transfer (BWT, \%) across benchmarks (mean $\pm$ std over 3 seeds). Higher is better; negative values indicate forgetting.}")
lines.append(r"\label{tab:bwt}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lccccc}")
lines.append(r"\toprule")
cols = " & ".join(bench_short[b] for b in bwt_bench_order)
lines.append(f"Method & {cols} \\\\")
lines.append(r"\midrule")

for mk in bwt_display:
    mn = bwt_names[mk]
    cells = []
    for bk in bwt_bench_order:
        d = data.get((mk, bk))
        if d:
            bold = (mk == best_bwt.get(bk))
            if bold:
                cells.append(f"$\\mathbf{{{d['bwt_mean']:+.1f}}} \\pm {d['bwt_std']:.1f}$")
            else:
                cells.append(f"${d['bwt_mean']:+.1f} \\pm {d['bwt_std']:.1f}$")
        else:
            cells.append("---")
    lines.append(f"{mn} & {' & '.join(cells)} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tex = "\n".join(lines)
(FIGURES_DIR / "bwt_table.tex").write_text(tex)
print("  Written to paper/figures/bwt_table.tex")

# =================================================================
# 3. Cost Table
# =================================================================
print("\n=== Generating cost_table.tex ===")
cost_order = ["latent_shift", "latent_shift_tuned", "der", "er", "gpm",
              "trgp", "gpm_lastlayer", "ewc", "packnet", "hat", "naive"]
cost_names = {
    "latent_shift": "LatentShift (Ours)", "latent_shift_tuned": "LatentShift-Tuned (Ours)",
    "der": "DER++", "er": "ER", "gpm": "GPM", "trgp": "TRGP",
    "gpm_lastlayer": "GPM (Last Layer)", "ewc": "EWC", "packnet": "PackNet",
    "hat": "HAT", "naive": "Naive",
}

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Wall-clock training time (seconds) per benchmark.}")
lines.append(r"\label{tab:cost}")
lines.append(r"\begin{tabular}{lrrrrr}")
lines.append(r"\toprule")
lines.append(r"Method & Split MNIST & Split CIFAR10 & Split CIFAR100 & Permuted MNIST & Split TinyImageNet \\")
lines.append(r"\midrule")

for mk in cost_order:
    mn = cost_names[mk]
    cells = []
    for bk in BENCHMARKS_TI:
        d = data.get((mk, bk))
        if d:
            cells.append(f"{int(round(d['time_mean']))}")
        else:
            cells.append("---")
    lines.append(f"{mn} & {' & '.join(cells)} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tex = "\n".join(lines)
(FIGURES_DIR / "cost_table.tex").write_text(tex)
print("  Written to paper/figures/cost_table.tex")

# =================================================================
# 4. CI Table
# =================================================================
print("\n=== Generating ci_table.tex ===")
CI_METHODS = {
    "naive": "Naive", "ewc": "EWC", "packnet": "PackNet", "hat": "HAT",
    "gpm": "GPM", "trgp": "TRGP", "er": "ER", "der": "DER++",
    "latent_shift": "LS (Ours)", "latent_shift_tuned": "LS-Tuned (Ours)",
}
CI_BENCHMARKS = ["split_cifar10", "split_cifar100"]
CI_BENCH_DISPLAY = {"split_cifar10": "Split CIFAR-10", "split_cifar100": "Split CIFAR-100"}

ci_data = defaultdict(list)
for mk in CI_METHODS:
    for bk in CI_BENCHMARKS:
        for s in SEEDS:
            r = load_result(mk, bk, s, suffix="_ci")
            if r and "class_incremental_accuracy" in r:
                ci_data[(mk, bk)].append(r["class_incremental_accuracy"] * 100)

# Find best
ci_best = {}
for bk in CI_BENCHMARKS:
    best_val = -1
    for mk in CI_METHODS:
        vals = ci_data.get((mk, bk), [])
        if vals and np.mean(vals) > best_val:
            best_val = np.mean(vals)
            ci_best[bk] = mk

ci_display_order = ["der", "er", "packnet", "gpm", "trgp", "hat", "ewc", "naive",
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

for mk in ci_display_order:
    mn = CI_METHODS[mk]
    cells = []
    for bk in CI_BENCHMARKS:
        vals = ci_data.get((mk, bk), [])
        if vals:
            m = np.mean(vals)
            s = np.std(vals)
            bold = (mk == ci_best.get(bk))
            if bold:
                cells.append(f"$\\mathbf{{{m:.1f}}} \\pm {s:.1f}$")
            else:
                cells.append(f"${m:.1f} \\pm {s:.1f}$")
        else:
            cells.append("---")
    sep = ""
    if mk == "latent_shift":
        sep = r"\midrule" + "\n"
    lines.append(f"{sep}{mn} & {' & '.join(cells)} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tex = "\n".join(lines)
(FIGURES_DIR / "ci_table.tex").write_text(tex)
print("  Written to paper/figures/ci_table.tex")

# =================================================================
# Summary
# =================================================================
print(f"\n{'='*60}")
print(f"Total results loaded: {len(data)} method/benchmark combos")
print(f"Missing: {len(missing)}")
print(f"Warnings: {len(warnings)}")
if missing:
    print("\nRe-run after missing results are available.")
