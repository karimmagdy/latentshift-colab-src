#!/usr/bin/env python3
import json
from pathlib import Path

methods = ["naive", "ewc", "gpm", "gpm_lastlayer", "trgp", "packnet", "hat", "er", "der", "latent_shift", "latent_shift_tuned"]
benchmarks = ["split_mnist", "permuted_mnist", "split_cifar10", "split_cifar100", "split_tinyimagenet"]
seeds = [42, 123, 456]

missing = []
total = 0
for m in methods:
    for b in benchmarks:
        for s in seeds:
            total += 1
            p = Path(f"results/{m}_{b}_seed{s}.json")
            if not p.exists():
                missing.append(f"{m}/{b}/s{s}")

print(f"=== MISSING ({len(missing)} / {total}) ===")
for m in missing:
    print(f"  {m}")

# Show TRGP results specifically
print("\n=== TRGP RESULTS ===")
for b in benchmarks:
    for s in seeds:
        p = Path(f"results/trgp_{b}_seed{s}.json")
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            metrics = d.get("metrics", d)
            acc = metrics.get("average_accuracy", "?")
            fgt = metrics.get("average_forgetting", "?")
            if isinstance(acc, float):
                acc = f"{acc:.1%}"
            if isinstance(fgt, float):
                fgt = f"{fgt:.1%}"
            print(f"  trgp/{b}/s{s}: acc={acc}, fgt={fgt}")
        else:
            print(f"  trgp/{b}/s{s}: MISSING")

# Show DER results
print("\n=== DER++ RESULTS ===")
for b in benchmarks:
    for s in seeds:
        p = Path(f"results/der_{b}_seed{s}.json")
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            metrics = d.get("metrics", d)
            acc = metrics.get("average_accuracy", "?")
            if isinstance(acc, float):
                acc = f"{acc:.1%}"
            ts = p.stat().st_mtime
            from datetime import datetime
            mod = datetime.fromtimestamp(ts).strftime("%H:%M")
            print(f"  der/{b}/s{s}: acc={acc} (modified {mod})")
        else:
            print(f"  der/{b}/s{s}: MISSING")

# Show ER results  
print("\n=== ER RESULTS ===")
for b in benchmarks:
    for s in seeds:
        p = Path(f"results/er_{b}_seed{s}.json")
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            metrics = d.get("metrics", d)
            acc = metrics.get("average_accuracy", "?")
            if isinstance(acc, float):
                acc = f"{acc:.1%}"
            ts = p.stat().st_mtime
            from datetime import datetime
            mod = datetime.fromtimestamp(ts).strftime("%H:%M")
            print(f"  er/{b}/s{s}: acc={acc} (modified {mod})")
        else:
            print(f"  er/{b}/s{s}: MISSING")

# Show LST results
print("\n=== LST RESULTS ===")
for b in benchmarks:
    for s in seeds:
        p = Path(f"results/latent_shift_tuned_{b}_seed{s}.json")
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            metrics = d.get("metrics", d)
            acc = metrics.get("average_accuracy", "?")
            if isinstance(acc, float):
                acc = f"{acc:.1%}"
            ts = p.stat().st_mtime
            from datetime import datetime
            mod = datetime.fromtimestamp(ts).strftime("%H:%M")
            print(f"  lst/{b}/s{s}: acc={acc} (modified {mod})")
        else:
            print(f"  lst/{b}/s{s}: MISSING")
