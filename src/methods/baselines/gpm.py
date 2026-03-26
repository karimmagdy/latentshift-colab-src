from __future__ import annotations

"""Gradient Projection Memory (GPM) baseline.

Saha et al., "Gradient Projection Memory for Continual Learning" (ICLR 2021).

After each task, stores the space of important gradients (via SVD of
per-layer representations). Future task gradients are projected to be
orthogonal to this stored space — the key comparator for LatentShift.

Key difference from LatentShift: GPM projects per-layer, LatentShift
projects in the holistic latent representation space.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..base import ContinualLearningMethod


class GPM(ContinualLearningMethod):

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
        threshold: float = 0.99,
        num_samples: int = 300,
        last_layer_only: bool = False,
    ):
        super().__init__(encoder, decoder, device)
        self.threshold = threshold
        self.num_samples = num_samples
        self.last_layer_only = last_layer_only

        # Per-layer subspace bases: dict[layer_name -> (feature_dim, rank)]
        self._layer_bases: dict[str, torch.Tensor] = {}
        # Hook handles for capturing activations
        self._activations: dict[str, torch.Tensor] = {}

    def prepare_task(self, task_id: int, train_loader: DataLoader) -> None:
        self.current_task = task_id
        self.decoder.add_task_head(task_id)

    def train_task(
        self, task_id: int, train_loader: DataLoader, epochs: int, lr: float
    ) -> dict[str, float]:
        self.encoder.train()
        self.decoder.train()

        params = list(self.encoder.parameters()) + list(
            self.decoder.heads[str(task_id)].parameters()
        )
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for _ in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                z = self.encoder(x)
                logits = self.decoder(z, task_id)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()

                # Project gradients per layer
                self._project_gradients()

                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_correct += (logits.argmax(1) == y).sum().item()
                total_samples += x.size(0)

        return {
            "train_loss": total_loss / max(total_samples, 1),
            "train_acc": total_correct / max(total_samples, 1),
        }

    def after_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Compute per-layer representation matrices and update subspace bases."""
        self.encoder.eval()
        hooks = []
        self._activations = {}

        # Install forward hooks to capture layer activations
        if self.last_layer_only:
            # Only hook the last Linear layer (latent-space projection)
            last_linear_name, last_linear_mod = None, None
            for name, module in self.encoder.named_modules():
                if isinstance(module, nn.Linear):
                    last_linear_name, last_linear_mod = name, module
            if last_linear_mod is not None:
                hooks.append(
                    last_linear_mod.register_forward_hook(self._make_hook(last_linear_name))
                )
        else:
            for name, module in self.encoder.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    hooks.append(
                        module.register_forward_hook(self._make_hook(name))
                    )

        # Collect activations
        count = 0
        with torch.no_grad():
            for x, _ in train_loader:
                x = x.to(self.device)
                self.encoder(x)
                count += x.size(0)
                if count >= self.num_samples:
                    break

        # Remove hooks
        for h in hooks:
            h.remove()

        # Update per-layer bases via SVD
        for name, acts in self._activations.items():
            acts = acts[: self.num_samples]
            if acts.dim() > 2:
                acts = acts.view(acts.size(0), acts.size(1), -1).mean(dim=2)
            orig_device = acts.device
            acts = acts.cpu()
            acts = acts - acts.mean(dim=0, keepdim=True)

            U, S, _ = torch.linalg.svd(acts, full_matrices=False)
            var = S ** 2
            cumvar = var.cumsum(0) / (var.sum() + 1e-12)
            r = int((cumvar < self.threshold).sum().item()) + 1
            r = min(r, acts.size(1))

            new_basis = torch.linalg.svd(acts.T, full_matrices=False)[0][:, :r]

            if name in self._layer_bases:
                combined = torch.cat([self._layer_bases[name].cpu(), new_basis], dim=1)
                Q, _ = torch.linalg.qr(combined)
                self._layer_bases[name] = Q.to(orig_device)
            else:
                Q, _ = torch.linalg.qr(new_basis)
                self._layer_bases[name] = Q.to(orig_device)

        self._activations = {}

    def _make_hook(self, name: str):
        def hook(module, input, output):
            act = input[0].detach()
            if name in self._activations:
                self._activations[name] = torch.cat([self._activations[name], act], dim=0)
            else:
                self._activations[name] = act
        return hook

    def _project_gradients(self) -> None:
        """Project parameter gradients to be orthogonal to stored per-layer bases."""
        if not self._layer_bases:
            return

        for name, module in self.encoder.named_modules():
            if name not in self._layer_bases:
                continue
            basis = self._layer_bases[name]  # (feat_dim, rank)

            for param in module.parameters():
                if param.grad is None:
                    continue
                g = param.grad.data
                shape = g.shape
                if g.dim() >= 2:
                    g_2d = g.view(g.size(0), -1)
                    if g_2d.size(1) == basis.size(0):
                        proj = g_2d @ basis @ basis.T
                        param.grad.data = (g_2d - proj).view(shape)
                    elif g_2d.size(0) == basis.size(0):
                        proj = basis @ basis.T @ g_2d
                        param.grad.data = (g_2d - proj).view(shape)
