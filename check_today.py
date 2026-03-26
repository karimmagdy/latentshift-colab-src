#!/usr/bin/env python3
"""Check all results that were recently written (today)."""
import json, glob, os
from datetime import datetime

for f in sorted(glob.glob('results/*.json')):
    if '_ci' in f:
        continue
    mtime = os.path.getmtime(f)
    mt = datetime.fromtimestamp(mtime)
    if mt.day >= 18:  # today's results  
        d = json.load(open(f))
        acc = d['metrics']['average_accuracy']
        forg = d['metrics']['average_forgetting']
        method = d['config']['method']
        bench = d['config']['benchmark']
        seed = d.get('seed', '?')
        alpha = d['config'].get('alpha', '-')
        print(f'{method:20s} {bench:20s} s{seed}: acc={acc:.4f} forg={forg:.4f}  alpha={alpha}')
