from __future__ import annotations

"""PackNet baseline: pruning-based continual learning.

Mallya & Lazebnik, "PackNet: Adding Multiple Tasks to a Single Network
by Iterative Pruning" (CVPR 2018).

After each task, prunes the least-important weights (by magnitude) and
freezes the surviving weights via binary masks. Future tasks can only
use the unpruned (free) capacity.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..base import ContinualLearningMethod


class PackNet(ContinualLearningMethod):

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
        prune_ratio: float = 0.75,
        retrain_epochs: int = 0,
    ):
        super().__init__(encoder, decoder, device)
        self.prune_ratio = prune_ratio
        self.retrain_epochs = retrain_epochs

        # Cumulative binary mask of parameters frozen by previous tasks.
        # 1 = frozen (belongs to an old task), 0 = free.
        self._frozen_mask: dict[str, torch.Tensor] = {}
        # Per-task masks (for diagnostics / potential replay)
        self._task_masks: dict[int, dict[str, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # CL interface
    # ------------------------------------------------------------------

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

                # Zero out gradients for frozen parameters
                self._apply_gradient_mask()

                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_correct += (logits.argmax(1) == y).sum().item()
                total_samples += x.size(0)

        # Optional: retrain after pruning (fine-tune with mask applied)
        if self.retrain_epochs > 0 and task_id > 0:
            for _ in range(self.retrain_epochs):
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    z = self.encoder(x)
                    logits = self.decoder(z, task_id)
                    loss = criterion(logits, y)
                    optimizer.zero_grad()
                    loss.backward()
                    self._apply_gradient_mask()
                    optimizer.step()

        return {
            "train_loss": total_loss / max(total_samples, 1),
            "train_acc": total_correct / max(total_samples, 1),
        }

    def after_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Prune the current task's weights and update the frozen mask."""
        task_mask = self._compute_prune_mask()
        self._task_masks[task_id] = task_mask

        # Merge into cumulative frozen mask
        for name, mask in task_mask.items():
            if name in self._frozen_mask:
                # Union: frozen if either old-frozen or newly-pruned-as-important
                self._frozen_mask[name] = self._frozen_mask[name] | mask
            else:
                self._frozen_mask[name] = mask.clone()

        # Zero out pruned weights so they don't affect future tasks
        with torch.no_grad():
            for name, param in self.encoder.named_parameters():
                if name in self._frozen_mask:
                    free = ~self._frozen_mask[name]
                    param.data[free] *= 0  # clear freed weights for next task

        total_frozen = sum(m.sum().item() for m in self._frozen_mask.values())
        total_params = sum(p.numel() for p in self.encoder.parameters())
        print(
            f"  PackNet task {task_id}: "
            f"frozen={int(total_frozen)}/{total_params} "
            f"({100 * total_frozen / total_params:.1f}%), "
            f"free={total_params - int(total_frozen)}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_prune_mask(self) -> dict[str, torch.Tensor]:
        """Identify the top-(1-prune_ratio) weights by magnitude as important.

        Returns a dict of boolean tensors: True = important (to be frozen).
        Only considers currently-free parameters.
        """
        # Collect all free weight magnitudes
        magnitudes = []
        param_info = []  # (name, shape, free_mask)
        for name, param in self.encoder.named_parameters():
            free = ~self._frozen_mask[name] if name in self._frozen_mask else torch.ones_like(param, dtype=torch.bool)
            free_mags = param.data.abs()[free]
            magnitudes.append(free_mags)
            param_info.append((name, param.shape, free))

        if not magnitudes:
            return {}

        all_mags = torch.cat(magnitudes)
        if all_mags.numel() == 0:
            return {}

        # Keep top (1 - prune_ratio) fraction of free weights
        keep_ratio = 1.0 - self.prune_ratio
        k = max(1, int(keep_ratio * all_mags.numel()))
        threshold = torch.topk(all_mags, k, largest=True).values[-1]

        task_mask = {}
        for name, param in self.encoder.named_parameters():
            free = ~self._frozen_mask[name] if name in self._frozen_mask else torch.ones_like(param, dtype=torch.bool)
            # Important = free AND above threshold
            important = free & (param.data.abs() >= threshold)
            task_mask[name] = important

        return task_mask

    def _apply_gradient_mask(self) -> None:
        """Zero out gradients for parameters frozen by previous tasks."""
        if not self._frozen_mask:
            return
        for name, param in self.encoder.named_parameters():
            if param.grad is not None and name in self._frozen_mask:
                param.grad.data[self._frozen_mask[name]] = 0.0
