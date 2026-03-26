#!/usr/bin/env python3
import json

for m in ['er', 'der']:
    for b in ['split_cifar10', 'split_cifar100']:
        for s in [42, 123, 456]:
            f = f'results/{m}_{b}_seed{s}.json'
            try:
                d = json.load(open(f))
                acc = d['metrics']['average_accuracy']
                forg = d['metrics']['average_forgetting']
                print(f'{m:4s} {b:16s} s{s}: acc={acc:.4f} forg={forg:.4f}')
            except Exception as e:
                print(f'{m:4s} {b:16s} s{s}: ERROR {e}')
