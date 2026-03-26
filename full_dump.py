#!/usr/bin/env python3
"""Comprehensive dump of all current results with staleness detection."""
import json
from pathlib import Path
from datetime import datetime

methods = ["naive", "ewc", "gpm", "gpm_lastlayer", "trgp", "packnet", "hat", "er", "der", "latent_shift", "latent_shift_tuned"]
benchmarks = ["split_mnist", "permuted_mnist", "split_cifar10", "split_cifar100", "split_tinyimagenet"]
seeds = [42, 123, 456]

# Cutoff: results from Mar 18 are considered fresh
FRESH_DATE = datetime(2026, 3, 18, 0, 0)

print(f"{'Method':<20} {'Benchmark':<22} {'S42':>8} {'S123':>8} {'S456':>8} {'Mean':>8} {'Stale?'}")
print("-" * 100)

for m in methods:
    for b in benchmarks:
        accs = {}
        stale_seeds = []
        for s in seeds:
            p = Path(f"results/{m}_{b}_seed{s}.json")
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                metrics = d.get("metrics", d)
                acc = metrics.get("average_accuracy")
                if acc is not None:
                    accs[s] = acc
                mod_time = datetime.fromtimestamp(p.stat().st_mtime)
                if mod_time < FRESH_DATE:
                    stale_seeds.append(s)
        
        if not accs:
            continue
            
        s42 = f"{accs.get(42, 0):.1%}" if 42 in accs else "---"
        s123 = f"{accs.get(123, 0):.1%}" if 123 in accs else "---"
        s456 = f"{accs.get(456, 0):.1%}" if 456 in accs else "---"
        mean = f"{sum(accs.values())/len(accs):.1%}" if accs else "---"
        stale = f"STALE:{stale_seeds}" if stale_seeds else ""
        
        print(f"{m:<20} {b:<22} {s42:>8} {s123:>8} {s456:>8} {mean:>8} {stale}")
    print()
