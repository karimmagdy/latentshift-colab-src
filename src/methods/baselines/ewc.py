from __future__ import annotations

"""Elastic Weight Consolidation (EWC) baseline.

Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (2017).

After each task, computes the Fisher Information Matrix (diagonal approximation)
and penalizes changes to parameters that are important for previous tasks.
"""

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..base import ContinualLearningMethod


class EWC(ContinualLearningMethod):

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
        ewc_lambda: float = 400.0,
        num_fisher_samples: int = 200,
    ):
        super().__init__(encoder, decoder, device)
        self.ewc_lambda = ewc_lambda
        self.num_fisher_samples = num_fisher_samples

        # Stores (fisher_dict, params_dict) for each completed task
        self._consolidation_data: list[tuple[dict, dict]] = []

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
                loss = criterion(logits, y) + self._ewc_penalty()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_correct += (logits.argmax(1) == y).sum().item()
                total_samples += x.size(0)

        return {
            "train_loss": total_loss / max(total_samples, 1),
            "train_acc": total_correct / max(total_samples, 1),
        }

    def after_task(self, task_id: int, train_loader: DataLoader) -> None:
        fisher = self._compute_fisher(task_id, train_loader)
        params_snapshot = {
            n: p.clone().detach() for n, p in self.encoder.named_parameters()
        }
        self._consolidation_data.append((fisher, params_snapshot))

    def _compute_fisher(self, task_id: int, loader: DataLoader) -> dict[str, torch.Tensor]:
        """Compute diagonal Fisher Information Matrix approximation."""
        self.encoder.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.encoder.named_parameters()}
        criterion = nn.CrossEntropyLoss()
        count = 0

        for x, y in loader:
            if count >= self.num_fisher_samples:
                break
            x, y = x.to(self.device), y.to(self.device)
            self.encoder.zero_grad()
            z = self.encoder(x)
            logits = self.decoder(z, task_id)
            loss = criterion(logits, y)
            loss.backward()

            for n, p in self.encoder.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) * x.size(0)
            count += x.size(0)

        for n in fisher:
            fisher[n] /= max(count, 1)
        return fisher

    def _ewc_penalty(self) -> torch.Tensor:
        """Compute EWC penalty: sum over old tasks of F * (θ - θ*)^2."""
        penalty = torch.tensor(0.0, device=self.device)
        for fisher, old_params in self._consolidation_data:
            for n, p in self.encoder.named_parameters():
                penalty += (fisher[n] * (p - old_params[n]).pow(2)).sum()
        return 0.5 * self.ewc_lambda * penalty
