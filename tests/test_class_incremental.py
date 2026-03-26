"""Tests for class-incremental evaluation and related components."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.encoder import MLPEncoder
from src.models.decoder import MultiHeadDecoder, SingleHeadDecoder
from src.models.shift import SubspaceTracker
from src.methods.latent_shift import LatentShiftMethod
from src.methods.baselines.naive import NaiveFineTuning
from src.data.benchmarks import SplitMNIST, ClassIncrementalWrapper


def _make_dummy_loader(n=100, input_dim=784, num_classes=2):
    x = torch.randn(n, input_dim)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=32)


def _make_ci_loader(n=100, input_dim=784, num_classes=4):
    """Make a loader with global class labels (class-incremental style)."""
    x = torch.randn(n, input_dim)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=32)


class TestSingleHeadDecoder:

    def test_forward_shape(self):
        dec = SingleHeadDecoder(latent_dim=64, total_classes=10)
        z = torch.randn(8, 64)
        out = dec(z)
        assert out.shape == (8, 10)

    def test_gradients_flow(self):
        dec = SingleHeadDecoder(latent_dim=64, total_classes=10)
        z = torch.randn(8, 64, requires_grad=True)
        out = dec(z)
        out.sum().backward()
        assert z.grad is not None


class TestClassIncrementalWrapper:

    def test_wraps_split_mnist(self):
        base = SplitMNIST(data_root="./data", classes_per_task=2)
        ci = ClassIncrementalWrapper(base)
        assert ci.num_tasks == 5
        assert ci.classes_per_task == 2
        assert ci.total_classes == 10

    def test_global_labels(self):
        base = SplitMNIST(data_root="./data", classes_per_task=2)
        ci = ClassIncrementalWrapper(base)
        _, test_loader = ci.get_task_loaders(0)
        for _, y in test_loader:
            # Task 0 should have labels from {0, 1} (the original MNIST class indices)
            assert y.max() <= 9  # global labels
            break

    def test_cumulative_loader(self):
        base = SplitMNIST(data_root="./data", classes_per_task=2)
        ci = ClassIncrementalWrapper(base)
        cum_loader = ci.get_cumulative_test_loader(1)
        all_labels = set()
        for _, y in cum_loader:
            all_labels.update(y.tolist())
        # Should have labels from tasks 0 and 1 (4 classes)
        assert len(all_labels) == 4


class TestMaxLogitEvaluation:

    def test_max_logit_runs(self):
        """Max-logit class-incremental eval should run and return valid accuracy."""
        encoder = MLPEncoder(input_dim=784, hidden_dim=64, latent_dim=64)
        decoder = MultiHeadDecoder(latent_dim=64, classes_per_task=2)
        device = torch.device("cpu")
        method = NaiveFineTuning(encoder, decoder, device)

        loader = _make_dummy_loader()
        for tid in range(2):
            method.prepare_task(tid, loader)
            method.train_task(tid, loader, epochs=1, lr=0.01)
            method.after_task(tid, loader)

        ci_loader = _make_ci_loader(n=50, num_classes=4)
        acc = method.evaluate_class_incremental(ci_loader, classes_per_task=2)
        assert 0 <= acc <= 1


class TestSubspaceTaskInference:

    def test_latentshift_ci_eval(self):
        """LatentShift subspace-based CI eval should run and return valid accuracy."""
        encoder = MLPEncoder(input_dim=784, hidden_dim=64, latent_dim=64)
        decoder = MultiHeadDecoder(latent_dim=64, classes_per_task=2)
        device = torch.device("cpu")
        method = LatentShiftMethod(
            encoder, decoder, device, latent_dim=64, threshold=0.95, num_samples=50
        )

        loader = _make_dummy_loader()
        for tid in range(2):
            method.prepare_task(tid, loader)
            method.train_task(tid, loader, epochs=1, lr=0.01)
            method.after_task(tid, loader)

        ci_loader = _make_ci_loader(n=50, num_classes=4)
        acc = method.evaluate_class_incremental(ci_loader, classes_per_task=2)
        assert 0 <= acc <= 1
