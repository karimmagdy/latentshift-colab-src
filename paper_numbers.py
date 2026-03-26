#!/usr/bin/env python3
"""Compute mean accuracy and forgetting for paper text."""
import json
from pathlib import Path

methods = {
    "der": "DER++",
    "latent_shift_tuned": "LS-Tuned",
    "latent_shift": "LS",
    "er": "ER",
    "hat": "HAT",
    "gpm": "GPM",
    "gpm_lastlayer": "GPM-LL",
    "trgp": "TRGP",
}
benchmarks = ["split_mnist", "permuted_mnist", "split_cifar10", "split_cifar100", "split_tinyimagenet"]
seeds = [42, 123, 456]

for m, label in methods.items():
    print(f"\n{label}:")
    for b in benchmarks:
        accs, fgts = [], []
        for s in seeds:
            p = Path(f"results/{m}_{b}_seed{s}.json")
            if p.exists():
                d = json.load(open(p))
                met = d.get("metrics", d)
                accs.append(met["average_accuracy"])
                fgts.append(met["average_forgetting"])
        if accs:
            mean_a = sum(accs) / len(accs)
            mean_f = sum(fgts) / len(fgts)
            print(f"  {b}: acc={mean_a:.1%} fgt={mean_f:.1%} (n={len(accs)})")
        else:
            print(f"  {b}: NO DATA")
