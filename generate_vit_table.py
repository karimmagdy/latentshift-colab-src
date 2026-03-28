#!/usr/bin/env python3
"""Generate ViT vs ResNet-18 comparison table for Split-CIFAR-100."""
import json
import os

import numpy as np

METHODS = {
    "naive": "Naive",
    "der": "DER++",
    "latent_shift": "LS (Ours)",
    "latent_shift_tuned": "LS-Tuned (Ours)",
}
# Prompt-based methods (ViT-only, no ResNet counterpart)
PROMPT_METHODS = {
    "l2p_vit": "L2P",
    "dualprompt_vit": "DualPrompt",
    "coda_prompt_vit": "CODA-P",
}
SEEDS = [42, 123, 456]
BENCHMARK = "split_cifar100"


def load_results(prefix, seeds):
    accs, fgts, bwts = [], [], []
    for s in seeds:
        f = f"results/{prefix}_seed{s}.json"
        if os.path.exists(f):
            d = json.load(open(f))
            m = d["metrics"]
            accs.append(m["average_accuracy"] * 100)
            fgts.append(m["average_forgetting"] * 100)
            bwts.append(m["backward_transfer"] * 100)
    if not accs:
        return None
    return {
        "acc_mean": np.mean(accs), "acc_std": np.std(accs),
        "fgt_mean": np.mean(fgts), "fgt_std": np.std(fgts),
        "bwt_mean": np.mean(bwts), "bwt_std": np.std(bwts),
        "n": len(accs),
    }


# Collect data for both architectures
print("=== ViT-Tiny vs ResNet-18 on Split-CIFAR-100 ===\n")
print(f"{'Method':<20} {'ResNet-18 Acc':<16} {'ViT-Tiny Acc':<16} {'ResNet-18 BWT':<16} {'ViT-Tiny BWT':<16}")
print("-" * 85)

vit_data = {}
resnet_data = {}

for mk, mn in METHODS.items():
    vit = load_results(f"{mk}_vit_{BENCHMARK}", SEEDS)
    resnet = load_results(f"{mk}_{BENCHMARK}", SEEDS)
    vit_data[mk] = vit
    resnet_data[mk] = resnet

    r_acc = f"{resnet['acc_mean']:.1f} ± {resnet['acc_std']:.1f}" if resnet else "---"
    v_acc = f"{vit['acc_mean']:.1f} ± {vit['acc_std']:.1f}" if vit else "---"
    r_bwt = f"{resnet['bwt_mean']:+.1f} ± {resnet['bwt_std']:.1f}" if resnet else "---"
    v_bwt = f"{vit['bwt_mean']:+.1f} ± {vit['bwt_std']:.1f}" if vit else "---"
    print(f"{mn:<20} {r_acc:<16} {v_acc:<16} {r_bwt:<16} {v_bwt:<16}")

# Prompt-based methods (ViT-only)
prompt_vit_data = {}
print(f"\n{'--- Prompt-based (ViT-only) ---':^85}")
for mk, mn in PROMPT_METHODS.items():
    vit = load_results(f"{mk}_{BENCHMARK}", SEEDS)
    prompt_vit_data[mk] = vit
    v_acc = f"{vit['acc_mean']:.1f} ± {vit['acc_std']:.1f}" if vit else "---"
    v_bwt = f"{vit['bwt_mean']:+.1f} ± {vit['bwt_std']:.1f}" if vit else "---"
    print(f"{mn:<20} {'---':<16} {v_acc:<16} {'---':<16} {v_bwt:<16}")

# LaTeX table
lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Architecture generality: ResNet-18 vs ViT-Tiny on Split-CIFAR-100 "
             r"(mean $\pm$ std over 3 seeds). LatentShift transfers to transformer "
             r"architectures without modification.}")
lines.append(r"\label{tab:vit}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lcccc}")
lines.append(r"\toprule")
lines.append(r" & \multicolumn{2}{c}{Accuracy (\%) $\uparrow$} & \multicolumn{2}{c}{BWT (\%) $\uparrow$} \\")
lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
lines.append(r"Method & ResNet-18 & ViT-Tiny & ResNet-18 & ViT-Tiny \\")
lines.append(r"\midrule")

display_order = ["der", "naive", "latent_shift", "latent_shift_tuned"]
for mk in display_order:
    mn = METHODS[mk]
    r = resnet_data[mk]
    v = vit_data[mk]
    
    r_acc = f"${r['acc_mean']:.1f} \\pm {r['acc_std']:.1f}$" if r else "---"
    v_acc = f"${v['acc_mean']:.1f} \\pm {v['acc_std']:.1f}$" if v else "---"
    r_bwt = f"${r['bwt_mean']:+.1f} \\pm {r['bwt_std']:.1f}$" if r else "---"
    v_bwt = f"${v['bwt_mean']:+.1f} \\pm {v['bwt_std']:.1f}$" if v else "---"

    sep = r"\midrule" if mk == "latent_shift" else ""
    if sep:
        lines.append(sep)
    lines.append(f"{mn} & {r_acc} & {v_acc} & {r_bwt} & {v_bwt} \\\\")

# Prompt-based methods (ViT-only)
lines.append(r"\midrule")
lines.append(r"\multicolumn{5}{l}{\textit{Prompt-based (ViT-Tiny only)}} \\")
prompt_display = ["l2p_vit", "dualprompt_vit", "coda_prompt_vit"]
for mk in prompt_display:
    mn = PROMPT_METHODS[mk]
    v = prompt_vit_data[mk]
    v_acc = f"${v['acc_mean']:.1f} \\pm {v['acc_std']:.1f}$" if v else "---"
    v_bwt = f"${v['bwt_mean']:+.1f} \\pm {v['bwt_std']:.1f}$" if v else "---"
    lines.append(f"{mn} & --- & {v_acc} & --- & {v_bwt} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tex = "\n".join(lines)
out_path = "paper/figures/vit_table.tex"
with open(out_path, "w") as f:
    f.write(tex)
print(f"\nLaTeX saved to {out_path}")
print(tex)
