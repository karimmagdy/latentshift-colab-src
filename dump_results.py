#!/usr/bin/env python3
"""Dump key results for paper text."""
import json, os

def get_result(method, bench, seed=42):
    f = "results/%s_%s_seed%d.json" % (method, bench, seed)
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    return {
        "acc": d["metrics"]["average_accuracy"] * 100,
        "forg": d["metrics"]["average_forgetting"] * 100,
        "alpha": d["config"].get("alpha"),
    }

benchmarks = ["split_mnist", "permuted_mnist", "split_cifar10", "split_cifar100", "split_tinyimagenet"]
methods = ["naive", "ewc", "gpm", "gpm_lastlayer", "trgp", "packnet", "hat", "er", "der", "latent_shift", "latent_shift_tuned"]

print("=== DER++ FIXED (seed 42) ===")
for b in benchmarks:
    r = get_result("der", b)
    if r:
        tag = "FIXED" if r["alpha"] == 1.0 else "OLD"
        print("  %-25s acc=%.1f%%  forg=%.1f%%  %s" % (b, r["acc"], r["forg"], tag))

print("\n=== TRGP (seed 42) ===")
for b in benchmarks:
    r = get_result("trgp", b)
    if r:
        print("  %-25s acc=%.1f%%  forg=%.1f%%" % (b, r["acc"], r["forg"]))
    else:
        print("  %-25s MISSING" % b)

print("\n=== LS-Tuned (seed 42) ===")
for b in benchmarks:
    r = get_result("latent_shift_tuned", b)
    if r:
        print("  %-25s acc=%.1f%%  forg=%.1f%%" % (b, r["acc"], r["forg"]))

print("\n=== Rankings per benchmark (seed 42, top methods) ===")
for b in benchmarks:
    results = {}
    for m in methods:
        r = get_result(m, b)
        if r:
            if m == "der" and r["alpha"] != 1.0:
                continue
            results[m] = r["acc"]
    ranked = sorted(results.items(), key=lambda x: -x[1])
    print("  %s:" % b)
    for i, (m, acc) in enumerate(ranked):
        print("    %d. %-22s %.1f%%" % (i+1, m, acc))
