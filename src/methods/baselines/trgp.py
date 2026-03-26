from __future__ import annotations

"""Trust Region Gradient Projection (TRGP) baseline.

Lin et al., "TRGP: Trust Region Gradient Projection for Continual Learning"
(ECCV 2022).

Extends GPM by computing task-relatedness scores between the new task and
previous tasks. Related tasks share gradient subspace rather than projecting
it out entirely, creating a "trust region" that enables beneficial forward
transfer while still preventing catastrophic forgetting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..base import ContinualLearningMethod


class TRGP(ContinualLearningMethod):

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
        threshold: float = 0.99,
        num_samples: int = 300,
        trust_alpha: float = 0.5,
    ):
        super().__init__(encoder, decoder, device)
        self.threshold = threshold
        self.num_samples = num_samples
        self.trust_alpha = trust_alpha

        # Per-layer subspace bases: dict[layer_name -> (feature_dim, rank)]
        self._layer_bases: dict[str, torch.Tensor] = {}
        # Per-task, per-layer bases (for relatedness computation)
        self._task_layer_bases: dict[int, dict[str, torch.Tensor]] = {}
        # Per-layer trust-region scale factors: dict[layer_name -> (rank,)]
        self._layer_scales: dict[str, torch.Tensor] = {}
        # Hook handles for capturing activations
        self._activations: dict[str, torch.Tensor] = {}

    def prepare_task(self, task_id: int, train_loader: DataLoader) -> None:
        self.current_task = task_id
        self.decoder.add_task_head(task_id)

        # Compute trust-region scales for this task using relatedness to
        # previous tasks' subspaces. Only possible after task 0.
        if task_id > 0 and self._layer_bases:
            self._compute_trust_region_scales(task_id, train_loader)

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

                # Project gradients with trust-region scaling
                self._project_gradients_trgp()

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

        for h in hooks:
            h.remove()

        # Store per-task bases and update combined bases
        task_bases = {}
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

            # Store this task's individual basis
            Q_task, _ = torch.linalg.qr(new_basis)
            task_bases[name] = Q_task.to(orig_device)

            # Update combined subspace
            if name in self._layer_bases:
                combined = torch.cat([self._layer_bases[name].cpu(), new_basis], dim=1)
                Q, _ = torch.linalg.qr(combined)
                self._layer_bases[name] = Q.to(orig_device)
            else:
                Q, _ = torch.linalg.qr(new_basis)
                self._layer_bases[name] = Q.to(orig_device)

        self._task_layer_bases[task_id] = task_bases
        self._activations = {}

    def _compute_trust_region_scales(
        self, task_id: int, train_loader: DataLoader
    ) -> None:
        """Compute per-layer trust-region scale factors based on task relatedness.

        For each layer, measures how much the new task's feature space overlaps
        with the stored subspace from each previous task. High overlap means
        the tasks are related, so we allow more gradient flow through those
        subspace directions (scale closer to 1 = less projection).
        """
        self.encoder.eval()
        hooks = []
        self._activations = {}

        for name, module in self.encoder.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(
                    module.register_forward_hook(self._make_hook(name))
                )

        count = 0
        with torch.no_grad():
            for x, _ in train_loader:
                x = x.to(self.device)
                self.encoder(x)
                count += x.size(0)
                if count >= self.num_samples:
                    break

        for h in hooks:
            h.remove()

        self._layer_scales = {}
        for name in self._layer_bases:
            basis = self._layer_bases[name]  # (feat_dim, rank)
            rank = basis.size(1)

            # Get current task's feature representation
            if name not in self._activations:
                self._layer_scales[name] = torch.zeros(rank, device=basis.device)
                continue

            acts = self._activations[name][: self.num_samples]
            if acts.dim() > 2:
                acts = acts.view(acts.size(0), acts.size(1), -1).mean(dim=2)
            acts = acts.cpu()
            acts = acts - acts.mean(dim=0, keepdim=True)

            # SVD of new task features
            U_new, S_new, _ = torch.linalg.svd(acts, full_matrices=False)
            var_new = S_new ** 2
            cumvar = var_new.cumsum(0) / (var_new.sum() + 1e-12)
            r_new = int((cumvar < self.threshold).sum().item()) + 1
            r_new = min(r_new, acts.size(1))
            new_basis = torch.linalg.svd(acts.T, full_matrices=False)[0][:, :r_new].to(
                basis.device
            )

            # Compute relatedness: for each direction in the combined basis,
            # how much does the new task's subspace overlap?
            # overlap = ||P_new @ basis_col||^2 where P_new = new_basis @ new_basis^T
            # This gives a per-direction score in [0, 1].
            overlap = (new_basis.T @ basis) ** 2  # (r_new, rank)
            per_dir_overlap = overlap.sum(dim=0)  # (rank,)
            per_dir_overlap = per_dir_overlap.clamp(0, 1)

            # Trust-region scale: high overlap → scale ≈ 1 (allow gradient)
            # Low overlap → scale ≈ 0 (project out fully, like GPM)
            # Scale by trust_alpha to control the trust-protection trade-off.
            self._layer_scales[name] = per_dir_overlap * self.trust_alpha

        self._activations = {}
        self.encoder.train()

    def _make_hook(self, name: str):
        def hook(module, input, output):
            act = input[0].detach()
            if name in self._activations:
                self._activations[name] = torch.cat(
                    [self._activations[name], act], dim=0
                )
            else:
                self._activations[name] = act

        return hook

    def _project_gradients_trgp(self) -> None:
        """Project gradients with trust-region scaling.

        Instead of fully removing gradient components along stored subspace
        (like GPM), scales the removal by (1 - relatedness), allowing
        gradient flow along directions shared with the new task.
        """
        if not self._layer_bases:
            return

        for name, module in self.encoder.named_modules():
            if name not in self._layer_bases:
                continue
            basis = self._layer_bases[name]  # (feat_dim, rank)

            # Get per-direction scale factors
            if name in self._layer_scales:
                scales = self._layer_scales[name]  # (rank,)
            else:
                # No trust-region info → full projection (like GPM)
                scales = torch.zeros(basis.size(1), device=basis.device)

            for param in module.parameters():
                if param.grad is None:
                    continue
                g = param.grad.data
                shape = g.shape
                if g.dim() >= 2:
                    g_2d = g.view(g.size(0), -1)
                    if g_2d.size(1) == basis.size(0):
                        # g_proj_i = (g @ basis_i) * basis_i^T
                        # Remove (1 - scale_i) fraction of each direction
                        coeffs = g_2d @ basis  # (out, rank)
                        removal = (1 - scales).unsqueeze(0)  # (1, rank)
                        proj = (coeffs * removal) @ basis.T  # (out, feat)
                        param.grad.data = (g_2d - proj).view(shape)
                    elif g_2d.size(0) == basis.size(0):
                        coeffs = basis.T @ g_2d  # (rank, out)
                        removal = (1 - scales).unsqueeze(1)  # (rank, 1)
                        proj = basis @ (coeffs * removal)  # (feat, out)
                        param.grad.data = (g_2d - proj).view(shape)
