#!/usr/bin/env python3
"""Generate Seq-CIFAR-100 (50-task) results table for the paper."""
import json
import os

import numpy as np

METHODS = {
    "naive": "Naive",
    "ewc": "EWC",
    "gpm": "GPM",
    "packnet": "PackNet",
    "l2p_vit": "L2P",
    "dualprompt_vit": "DualPrompt",
    "coda_prompt_vit": "CODA-P",
    "latent_shift": "LS (Ours)",
    "latent_shift_tuned": "LS-Tuned (Ours)",
    "der": "DER++",
}
SEEDS = [42, 123, 456]
BENCHMARK = "seq_cifar100"

data = {}
for mk in METHODS:
    accs, fgts, bwts = [], [], []
    for s in SEEDS:
        f = f"results/{mk}_{BENCHMARK}_seed{s}.json"
        if os.path.exists(f):
            d = json.load(open(f))
            m = d["metrics"]
            accs.append(m["average_accuracy"] * 100)
            fgts.append(m["average_forgetting"] * 100)
            bwts.append(m["backward_transfer"] * 100)
    if accs:
        data[mk] = {
            "acc_mean": np.mean(accs), "acc_std": np.std(accs),
            "fgt_mean": np.mean(fgts), "fgt_std": np.std(fgts),
            "bwt_mean": np.mean(bwts), "bwt_std": np.std(bwts),
            "n_seeds": len(accs),
        }

# Console output
print("=== Seq-CIFAR-100 (50 tasks, 2 classes each) ===")
print(f"{'Method':<20} {'Acc (%)':<16} {'Fgt (%)':<16} {'BWT (%)':<16} Seeds")
print("-" * 80)
order = sorted(data.keys(), key=lambda m: -data[m]["acc_mean"])
for mk in order:
    d = data[mk]
    print(f"{METHODS[mk]:<20} "
          f"{d['acc_mean']:5.1f} ± {d['acc_std']:4.1f}   "
          f"{d['fgt_mean']:5.1f} ± {d['fgt_std']:4.1f}   "
          f"{d['bwt_mean']:+5.1f} ± {d['bwt_std']:4.1f}   "
          f"{d['n_seeds']}")

# LaTeX output
display_order = ["der", "packnet", "gpm", "ewc", "naive",
                 "l2p_vit", "dualprompt_vit", "coda_prompt_vit",
                 "latent_shift", "latent_shift_tuned"]

# Find best per column
best_acc = max(data.values(), key=lambda d: d["acc_mean"])["acc_mean"]
best_fgt = min(data.values(), key=lambda d: d["fgt_mean"])["fgt_mean"]
best_bwt = max(data.values(), key=lambda d: d["bwt_mean"])["bwt_mean"]

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Results on Sequential-CIFAR-100 (50 tasks, 2 classes each). "
             r"Mean $\pm$ std over 3 seeds. This long task sequence tests scalability "
             r"of continual learning methods.}")
lines.append(r"\label{tab:seq_cifar100}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lccc}")
lines.append(r"\toprule")
lines.append(r"Method & Acc (\%) $\uparrow$ & Fgt (\%) $\downarrow$ & BWT (\%) $\uparrow$ \\")
lines.append(r"\midrule")

for mk in display_order:
    if mk not in data:
        continue
    d = data[mk]
    mn = METHODS[mk]

    acc_str = f"${d['acc_mean']:.1f} \\pm {d['acc_std']:.1f}$"
    if abs(d["acc_mean"] - best_acc) < 0.05:
        acc_str = f"$\\mathbf{{{d['acc_mean']:.1f}}} \\pm {d['acc_std']:.1f}$"

    fgt_str = f"${d['fgt_mean']:.1f} \\pm {d['fgt_std']:.1f}$"
    if abs(d["fgt_mean"] - best_fgt) < 0.05:
        fgt_str = f"$\\mathbf{{{d['fgt_mean']:.1f}}} \\pm {d['fgt_std']:.1f}$"

    bwt_str = f"${d['bwt_mean']:+.1f} \\pm {d['bwt_std']:.1f}$"
    if abs(d["bwt_mean"] - best_bwt) < 0.05:
        bwt_str = f"$\\mathbf{{{d['bwt_mean']:+.1f}}} \\pm {d['bwt_std']:.1f}$"

    sep = ""
    if mk == "l2p_vit":
        sep = r"\midrule"
    elif mk == "latent_shift":
        sep = r"\midrule"
    if sep:
        lines.append(sep)
    lines.append(f"{mn} & {acc_str} & {fgt_str} & {bwt_str} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tex = "\n".join(lines)
out_path = "paper/figures/seq_cifar100_table.tex"
with open(out_path, "w") as f:
    f.write(tex)
print(f"\nLaTeX saved to {out_path}")
print(tex)
