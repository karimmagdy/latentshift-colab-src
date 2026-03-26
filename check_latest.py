#!/usr/bin/env python3
import json
results = [
    'results/der_split_cifar10_seed42.json',
    'results/der_split_cifar100_seed42.json',
    'results/er_split_mnist_seed456.json',
    'results/trgp_split_mnist_seed123.json',
    'results/trgp_split_mnist_seed456.json',
]
for f in results:
    try:
        d = json.load(open(f))
        m = d['config']['method']
        b = d['config']['benchmark']
        s = d.get('seed','?')
        acc = d['metrics']['average_accuracy']
        forg = d['metrics']['average_forgetting']
        alpha = d['config'].get('alpha', '-')
        print(f'{m:20s} {b:20s} s{s}: acc={acc:.4f} forg={forg:.4f}')
    except Exception as e:
        print(f'{f}: {e}')
