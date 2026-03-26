#!/usr/bin/env python3
import json
f = 'results/der_split_cifar10_seed42.json'
d = json.load(open(f))
print(f'alpha={d["config"].get("alpha")}, buffer={d["config"].get("buffer_size_per_task")}')
print(f'acc={d["metrics"]["average_accuracy"]:.4f}')
print(f'forg={d["metrics"]["average_forgetting"]:.4f}')
