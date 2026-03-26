# LatentShift: Solving Catastrophic Forgetting via Hilbert-Inspired Latent Shifting

A novel continual learning method that applies Hilbert's Hotel shift operator ($n \to 2n$) to a neural network's latent space. After each task, learned representations are "shifted" into an archived subspace, freeing orthogonal dimensions for new learning — guaranteeing zero forgetting on previous tasks.

## Key Idea

| Hilbert's Hotel | LatentShift |
|----------------|-------------|
| Infinite rooms, all full | Latent space $\mathbb{R}^d$, partially occupied |
| Shift guests: room $n \to 2n$ | Shift representations into archive subspace |
| Odd rooms freed for new guests | Orthogonal "free" dimensions for new task |
| Isometry: no guest is lost | Isometry: no knowledge is lost |

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run LatentShift on Split-MNIST
python experiments/run_experiment.py --config configs/latent_shift_split_mnist.yaml

# Run baselines for comparison
python experiments/run_experiment.py --config configs/naive_split_mnist.yaml
python experiments/run_experiment.py --config configs/ewc_split_mnist.yaml
python experiments/run_experiment.py --config configs/gpm_split_mnist.yaml
```

## Project Structure

```
LatentShift/
├── configs/                   # YAML experiment configs
├── src/
│   ├── models/
│   │   ├── encoder.py         # MLP & ResNet-18 backbones
│   │   ├── shift.py           # SubspaceTracker — core shift operator
│   │   └── decoder.py         # Multi-head task classifier
│   ├── methods/
│   │   ├── base.py            # CL method interface
│   │   ├── latent_shift.py    # LatentShift method
│   │   └── baselines/         # EWC, GPM, Naive fine-tuning
│   ├── training/
│   │   ├── trainer.py         # CL training loop
│   │   └── optimizer.py       # Gradient projection wrapper
│   ├── data/
│   │   └── benchmarks.py      # Split-MNIST, CIFAR-10/100, Permuted-MNIST
│   └── utils/
│       └── metrics.py         # Accuracy, forgetting, transfer metrics
├── experiments/
│   └── run_experiment.py      # CLI entry point
├── tests/
│   └── test_shift.py          # Unit tests for shift operator
└── paper/                     # LaTeX paper source
```

## Method Overview

1. **Train** encoder + task head on the current task
2. **Archive**: Compute SVD of latent activations → identify occupied subspace → merge into cumulative archive via QR
3. **Shift**: For subsequent tasks, project all encoder gradients onto the free subspace ($P = I - AA^T$), ensuring old representations are untouched
4. **Repeat**: Each new task only modifies the remaining orthogonal dimensions

### Theoretical Guarantees

- **Zero-Forgetting Theorem**: If the shift operator is an isometry on the occupied subspace, and gradients are projected onto the orthogonal complement, then outputs on all previous tasks are exactly preserved.
- **Capacity Theorem**: A latent space of dimension $d$ with rank-$r$ representations per task supports $T = \lfloor d/r \rfloor$ tasks with zero forgetting.

## Running Tests

```bash
pytest tests/ -v
```

## Benchmarks

| Benchmark | Tasks | Classes/Task | Backbone |
|-----------|-------|-------------|----------|
| Split-MNIST | 5 | 2 | MLP (400) |
| Permuted-MNIST | 10 | 10 | MLP (400) |
| Split-CIFAR-10 | 5 | 2 | ResNet-18 |
| Split-CIFAR-100 | 10 | 10 | ResNet-18 |

## License

MIT
