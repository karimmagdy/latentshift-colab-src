from __future__ import annotations

"""Naive fine-tuning baseline: no continual learning mechanism.

Simply trains on each new task with SGD, overwriting previous knowledge.
This establishes the lower bound (worst forgetting).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..base import ContinualLearningMethod


class NaiveFineTuning(ContinualLearningMethod):

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
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_correct += (logits.argmax(1) == y).sum().item()
                total_samples += x.size(0)

        return {
            "train_loss": total_loss / max(total_samples, 1),
            "train_acc": total_correct / max(total_samples, 1),
        }

    def after_task(self, task_id: int, train_loader: DataLoader) -> None:
        pass  # No CL mechanism
