from __future__ import annotations

"""Unit tests for PackNet and HAT baselines."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.encoder import MLPEncoder, ResNetEncoder
from src.models.decoder import MultiHeadDecoder
from src.methods.baselines.packnet import PackNet
from src.methods.baselines.hat import HAT, HATEncoder
from src.methods.baselines.er import ExperienceReplay
from src.methods.baselines.der import DERPlusPlus


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _make_dummy_loader(n: int = 100, input_dim: int = 784, num_classes: int = 2):
    x = torch.randn(n, input_dim)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=32)


def _make_components(latent_dim=64, hidden_dim=64, input_dim=784, classes_per_task=2):
    encoder = MLPEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = MultiHeadDecoder(latent_dim=latent_dim, classes_per_task=classes_per_task)
    device = torch.device("cpu")
    return encoder, decoder, device


# ---------------------------------------------------------------
# PackNet Tests
# ---------------------------------------------------------------

class TestPackNet:

    def test_initial_state(self):
        encoder, decoder, device = _make_components()
        method = PackNet(encoder, decoder, device, prune_ratio=0.75)
        assert len(method._frozen_mask) == 0
        assert len(method._task_masks) == 0

    def test_prepare_adds_head(self):
        encoder, decoder, device = _make_components()
        method = PackNet(encoder, decoder, device)
        loader = _make_dummy_loader()
        method.prepare_task(0, loader)
        assert "0" in decoder.heads

    def test_train_returns_metrics(self):
        encoder, decoder, device = _make_components()
        method = PackNet(encoder, decoder, device, prune_ratio=0.75)
        loader = _make_dummy_loader()
        method.prepare_task(0, loader)
        metrics = method.train_task(0, loader, epochs=1, lr=0.01)
        assert "train_loss" in metrics
        assert "train_acc" in metrics
        assert 0 <= metrics["train_acc"] <= 1

    def test_after_task_creates_mask(self):
        encoder, decoder, device = _make_components()
        method = PackNet(encoder, decoder, device, prune_ratio=0.75)
        loader = _make_dummy_loader()
        method.prepare_task(0, loader)
        method.train_task(0, loader, epochs=1, lr=0.01)
        method.after_task(0, loader)
        assert 0 in method._task_masks
        assert len(method._frozen_mask) > 0

    def test_frozen_mask_grows(self):
        """After two tasks, more parameters should be frozen."""
        encoder, decoder, device = _make_components()
        method = PackNet(encoder, decoder, device, prune_ratio=0.5)
        loader = _make_dummy_loader()

        # Task 0
        method.prepare_task(0, loader)
        method.train_task(0, loader, epochs=1, lr=0.01)
        method.after_task(0, loader)
        frozen_after_t0 = sum(m.sum().item() for m in method._frozen_mask.values())

        # Task 1
        method.prepare_task(1, loader)
        method.train_task(1, loader, epochs=1, lr=0.01)
        method.after_task(1, loader)
        frozen_after_t1 = sum(m.sum().item() for m in method._frozen_mask.values())

        assert frozen_after_t1 >= frozen_after_t0

    def test_gradient_masking(self):
        """Frozen params should not receive gradient updates."""
        encoder, decoder, device = _make_components()
        method = PackNet(encoder, decoder, device, prune_ratio=0.5)
        loader = _make_dummy_loader()

        # Task 0: train and freeze
        method.prepare_task(0, loader)
        method.train_task(0, loader, epochs=1, lr=0.01)
        method.after_task(0, loader)

        # Snapshot frozen param values
        frozen_snapshots = {}
        for name, param in encoder.named_parameters():
            if name in method._frozen_mask:
                mask = method._frozen_mask[name]
                frozen_snapshots[name] = param.data[mask].clone()

        # Task 1: train (frozen params should not change)
        method.prepare_task(1, loader)
        method.train_task(1, loader, epochs=2, lr=0.01)

        for name, old_vals in frozen_snapshots.items():
            mask = method._frozen_mask[name]
            # Note: after_task for task 0 zeros out free weights, so
            # frozen params from task 0 should remain unchanged
            new_vals = encoder.state_dict()[name][mask]
            # Allow tiny numerical drift
            assert torch.allclose(old_vals, new_vals, atol=1e-5), \
                f"Frozen params changed in {name}"


# ---------------------------------------------------------------
# HAT Tests
# ---------------------------------------------------------------

class TestHAT:

    def test_hat_encoder_wrapping(self):
        base = MLPEncoder(input_dim=784, hidden_dim=64, latent_dim=32)
        hat_enc = HATEncoder(base, num_tasks=5)
        assert hat_enc.latent_dim == 32
        assert len(hat_enc.mask_layers) > 0

    def test_hat_encoder_forward(self):
        base = MLPEncoder(input_dim=784, hidden_dim=64, latent_dim=32)
        hat_enc = HATEncoder(base, num_tasks=5)
        hat_enc.set_task(0, s=1.0)
        x = torch.randn(8, 784)
        z = hat_enc(x)
        assert z.shape == (8, 32)

    def test_mask_values_range(self):
        base = MLPEncoder(input_dim=784, hidden_dim=64, latent_dim=32)
        hat_enc = HATEncoder(base, num_tasks=5)
        masks = hat_enc.get_masks(0, s=100.0)
        for name, gate in masks.items():
            assert gate.min() >= 0.0
            assert gate.max() <= 1.0

    def test_prepare_adds_head(self):
        base = MLPEncoder(input_dim=784, hidden_dim=64, latent_dim=32)
        hat_enc = HATEncoder(base, num_tasks=5)
        decoder = MultiHeadDecoder(latent_dim=32, classes_per_task=2)
        method = HAT(hat_enc, decoder, torch.device("cpu"))
        loader = _make_dummy_loader()
        method.prepare_task(0, loader)
        assert "0" in decoder.heads

    def test_train_returns_metrics(self):
        base = MLPEncoder(input_dim=784, hidden_dim=64, latent_dim=32)
        hat_enc = HATEncoder(base, num_tasks=5)
        decoder = MultiHeadDecoder(latent_dim=32, classes_per_task=2)
        method = HAT(hat_enc, decoder, torch.device("cpu"), s_max=10.0)
        loader = _make_dummy_loader()
        method.prepare_task(0, loader)
        metrics = method.train_task(0, loader, epochs=1, lr=0.01)
        assert "train_loss" in metrics
        assert 0 <= metrics["train_acc"] <= 1

    def test_after_task_accumulates_mask(self):
        base = MLPEncoder(input_dim=784, hidden_dim=64, latent_dim=32)
        hat_enc = HATEncoder(base, num_tasks=5)
        decoder = MultiHeadDecoder(latent_dim=32, classes_per_task=2)
        method = HAT(hat_enc, decoder, torch.device("cpu"), s_max=10.0)
        loader = _make_dummy_loader()

        method.prepare_task(0, loader)
        method.train_task(0, loader, epochs=1, lr=0.01)
        method.after_task(0, loader)
        assert len(method._cumulative_mask) > 0

    def test_evaluate_uses_task_masks(self):
        base = MLPEncoder(input_dim=784, hidden_dim=64, latent_dim=32)
        hat_enc = HATEncoder(base, num_tasks=5)
        decoder = MultiHeadDecoder(latent_dim=32, classes_per_task=2)
        method = HAT(hat_enc, decoder, torch.device("cpu"), s_max=10.0)
        loader = _make_dummy_loader()

        method.prepare_task(0, loader)
        method.train_task(0, loader, epochs=1, lr=0.01)
        method.after_task(0, loader)
        acc = method.evaluate(0, loader)
        assert 0 <= acc <= 1

    def test_multi_task_training(self):
        """Train two tasks and ensure both can be evaluated."""
        base = MLPEncoder(input_dim=784, hidden_dim=64, latent_dim=32)
        hat_enc = HATEncoder(base, num_tasks=5)
        decoder = MultiHeadDecoder(latent_dim=32, classes_per_task=2)
        method = HAT(hat_enc, decoder, torch.device("cpu"), s_max=10.0)
        loader = _make_dummy_loader()

        for tid in range(2):
            method.prepare_task(tid, loader)
            method.train_task(tid, loader, epochs=1, lr=0.01)
            method.after_task(tid, loader)

        for tid in range(2):
            acc = method.evaluate(tid, loader)
            assert 0 <= acc <= 1


# ---------------------------------------------------------------
# Experience Replay Tests
# ---------------------------------------------------------------

class TestER:

    def test_initial_state(self):
        encoder, decoder, device = _make_components()
        method = ExperienceReplay(encoder, decoder, device)
        assert len(method._buffer) == 0
        assert method._tasks_seen == 0

    def test_prepare_adds_head(self):
        encoder, decoder, device = _make_components()
        method = ExperienceReplay(encoder, decoder, device)
        loader = _make_dummy_loader()
        method.prepare_task(0, loader)
        assert "0" in decoder.heads

    def test_train_returns_metrics(self):
        encoder, decoder, device = _make_components()
        method = ExperienceReplay(encoder, decoder, device)
        loader = _make_dummy_loader()
        method.prepare_task(0, loader)
        metrics = method.train_task(0, loader, epochs=1, lr=0.01)
        assert "train_loss" in metrics
        assert 0 <= metrics["train_acc"] <= 1

    def test_after_task_populates_buffer(self):
        encoder, decoder, device = _make_components()
        method = ExperienceReplay(encoder, decoder, device, buffer_size_per_task=50)
        loader = _make_dummy_loader(n=100)
        method.prepare_task(0, loader)
        method.train_task(0, loader, epochs=1, lr=0.01)
        method.after_task(0, loader)
        assert len(method._buffer) == 50
        assert method._tasks_seen == 1

    def test_replay_during_training(self):
        """After task 0, training on task 1 should use replay from buffer."""
        encoder, decoder, device = _make_components()
        method = ExperienceReplay(encoder, decoder, device, buffer_size_per_task=50)
        loader = _make_dummy_loader(n=100)

        method.prepare_task(0, loader)
        method.train_task(0, loader, epochs=1, lr=0.01)
        method.after_task(0, loader)
        assert len(method._buffer) == 50

        method.prepare_task(1, loader)
        metrics = method.train_task(1, loader, epochs=1, lr=0.01)
        assert "train_loss" in metrics

    def test_multi_task(self):
        """Train and evaluate across two tasks."""
        encoder, decoder, device = _make_components()
        method = ExperienceReplay(encoder, decoder, device, buffer_size_per_task=30)
        loader = _make_dummy_loader()

        for tid in range(2):
            method.prepare_task(tid, loader)
            method.train_task(tid, loader, epochs=1, lr=0.01)
            method.after_task(tid, loader)

        for tid in range(2):
            acc = method.evaluate(tid, loader)
            assert 0 <= acc <= 1


# ---------------------------------------------------------------
# HAT ResNet-specific Tests
# ---------------------------------------------------------------

class TestHATResNet:

    def _make_resnet_components(self, latent_dim=32, num_tasks=5):
        base = ResNetEncoder(latent_dim=latent_dim, pretrained=False)
        hat_enc = HATEncoder(base, num_tasks=num_tasks)
        decoder = MultiHeadDecoder(latent_dim=latent_dim, classes_per_task=2)
        device = torch.device("cpu")
        return hat_enc, decoder, device

    def _make_cifar_loader(self, n=32, num_classes=2):
        x = torch.randn(n, 3, 32, 32)
        y = torch.randint(0, num_classes, (n,))
        return DataLoader(TensorDataset(x, y), batch_size=16)

    def test_batchnorm_replaced(self):
        """After wrapping, no BatchNorm should remain in the encoder."""
        hat_enc, _, _ = self._make_resnet_components()
        for name, module in hat_enc.base_encoder.named_modules():
            assert not isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)), \
                f"BatchNorm found at {name} — should be GroupNorm"

    def test_block_level_masking(self):
        """ResNet should have masks for conv1, blocks, and projection."""
        hat_enc, _, _ = self._make_resnet_components()
        assert "conv1" in hat_enc.mask_layers
        assert "projection" in hat_enc.mask_layers
        block_masks = [n for n in hat_enc._layer_order
                       if n.startswith("layer") and "block" in n]
        assert len(block_masks) == 8  # 2 blocks × 4 layers in ResNet-18

    def test_forward_shape(self):
        """Forward pass through HAT-wrapped ResNet should produce correct shape."""
        hat_enc, _, _ = self._make_resnet_components(latent_dim=64)
        hat_enc.set_task(0, s=1.0)
        x = torch.randn(4, 3, 32, 32)
        z = hat_enc(x)
        assert z.shape == (4, 64)

    def test_train_on_cifar(self):
        """Full train cycle on CIFAR-sized data."""
        hat_enc, decoder, device = self._make_resnet_components()
        method = HAT(hat_enc, decoder, device, s_max=10.0)
        loader = self._make_cifar_loader()
        method.prepare_task(0, loader)
        metrics = method.train_task(0, loader, epochs=1, lr=0.01)
        assert "train_loss" in metrics
        method.after_task(0, loader)
        assert len(method._cumulative_mask) > 0

    def test_two_tasks_cifar(self):
        """Train two tasks on CIFAR and evaluate both."""
        hat_enc, decoder, device = self._make_resnet_components()
        method = HAT(hat_enc, decoder, device, s_max=10.0)
        loader = self._make_cifar_loader()

        for tid in range(2):
            method.prepare_task(tid, loader)
            method.train_task(tid, loader, epochs=1, lr=0.01)
            method.after_task(tid, loader)

        for tid in range(2):
            acc = method.evaluate(tid, loader)
            assert 0 <= acc <= 1


# ---------------------------------------------------------------
# DER++ Tests
# ---------------------------------------------------------------

class TestDERPlusPlus:

    def test_initial_state(self):
        encoder, decoder, device = _make_components()
        method = DERPlusPlus(encoder, decoder, device)
        assert len(method._buffer) == 0
        assert method._tasks_seen == 0

    def test_prepare_adds_head(self):
        encoder, decoder, device = _make_components()
        method = DERPlusPlus(encoder, decoder, device)
        loader = _make_dummy_loader()
        method.prepare_task(0, loader)
        assert "0" in decoder.heads

    def test_train_returns_metrics(self):
        encoder, decoder, device = _make_components()
        method = DERPlusPlus(encoder, decoder, device)
        loader = _make_dummy_loader()
        method.prepare_task(0, loader)
        metrics = method.train_task(0, loader, epochs=1, lr=0.01)
        assert "train_loss" in metrics
        assert 0 <= metrics["train_acc"] <= 1

    def test_after_task_stores_logits(self):
        encoder, decoder, device = _make_components()
        method = DERPlusPlus(encoder, decoder, device, buffer_size_per_task=50)
        loader = _make_dummy_loader(n=100)
        method.prepare_task(0, loader)
        method.train_task(0, loader, epochs=1, lr=0.01)
        method.after_task(0, loader)
        assert len(method._buffer) == 50
        # Each buffer entry should have 4 elements: (x, y, task_id, logits)
        assert len(method._buffer[0]) == 4
        assert method._buffer[0][3].dim() == 1  # logits vector

    def test_replay_with_logits(self):
        encoder, decoder, device = _make_components()
        method = DERPlusPlus(encoder, decoder, device, buffer_size_per_task=50)
        loader = _make_dummy_loader(n=100)

        method.prepare_task(0, loader)
        method.train_task(0, loader, epochs=1, lr=0.01)
        method.after_task(0, loader)

        method.prepare_task(1, loader)
        metrics = method.train_task(1, loader, epochs=1, lr=0.01)
        assert "train_loss" in metrics

    def test_multi_task(self):
        encoder, decoder, device = _make_components()
        method = DERPlusPlus(encoder, decoder, device, buffer_size_per_task=30)
        loader = _make_dummy_loader()

        for tid in range(2):
            method.prepare_task(tid, loader)
            method.train_task(tid, loader, epochs=1, lr=0.01)
            method.after_task(tid, loader)

        for tid in range(2):
            acc = method.evaluate(tid, loader)
            assert 0 <= acc <= 1
