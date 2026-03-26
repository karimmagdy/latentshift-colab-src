from __future__ import annotations

"""Experience Replay (ER) baseline.

Maintains a fixed-size replay buffer of past task samples. During training on
each new task, interleaves current-task batches with replay batches from the
buffer. After each task, reservoir-samples data into the buffer.
"""

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..base import ContinualLearningMethod


class ExperienceReplay(ContinualLearningMethod):
    """Experience Replay continual learning baseline.

    Args:
        buffer_size_per_task: Number of samples to store per task.
        replay_batch_size: Batch size for replay sampling.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
        buffer_size_per_task: int = 200,
        replay_batch_size: int = 64,
    ):
        super().__init__(encoder, decoder, device)
        self.buffer_size_per_task = buffer_size_per_task
        self.replay_batch_size = replay_batch_size
        # Buffer: list of (x_cpu, y_cpu, task_id) tuples
        self._buffer: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        self._tasks_seen: int = 0

    def prepare_task(self, task_id: int, train_loader: DataLoader) -> None:
        self.current_task = task_id
        self.decoder.add_task_head(task_id)

    def train_task(
        self, task_id: int, train_loader: DataLoader, epochs: int, lr: float
    ) -> dict[str, float]:
        self.encoder.train()
        self.decoder.train()

        # Optimizer over all decoder heads seen so far + encoder
        head_params = []
        for tid in range(task_id + 1):
            key = str(tid)
            if key in self.decoder.heads:
                head_params.extend(self.decoder.heads[key].parameters())
        params = list(self.encoder.parameters()) + head_params
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for _ in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Current task loss
                z = self.encoder(x)
                logits = self.decoder(z, task_id)
                loss = criterion(logits, y)

                # Replay loss (if buffer is non-empty)
                if self._buffer:
                    rx, ry, rt = self._sample_replay()
                    rx, ry = rx.to(self.device), ry.to(self.device)
                    rz = self.encoder(rx)
                    # Group by task_id for correct decoder head
                    replay_loss = torch.tensor(0.0, device=self.device)
                    unique_tasks = rt.unique()
                    for t in unique_tasks:
                        mask = rt == t
                        r_logits = self.decoder(rz[mask], t.item())
                        replay_loss = replay_loss + criterion(r_logits, ry[mask])
                    replay_loss = replay_loss / max(len(unique_tasks), 1)
                    loss = loss + replay_loss

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
        """Reservoir-sample current task data into the buffer."""
        self._tasks_seen += 1
        # Collect all samples from this task
        all_x, all_y = [], []
        for x, y in train_loader:
            all_x.append(x)
            all_y.append(y)
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)

        # Randomly select up to buffer_size_per_task samples
        n = all_x.size(0)
        k = min(n, self.buffer_size_per_task)
        indices = torch.randperm(n)[:k]
        for i in indices:
            self._buffer.append((all_x[i].cpu(), all_y[i].cpu(), task_id))

        # If total buffer exceeds limit, downsample uniformly
        max_total = self.buffer_size_per_task * self._tasks_seen
        if len(self._buffer) > max_total:
            self._buffer = random.sample(self._buffer, max_total)

        print(f"  ER buffer: {len(self._buffer)} samples ({self._tasks_seen} tasks)")

    def _sample_replay(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch from the replay buffer."""
        k = min(len(self._buffer), self.replay_batch_size)
        samples = random.sample(self._buffer, k)
        xs = torch.stack([s[0] for s in samples])
        ys = torch.stack([s[1] for s in samples])
        ts = torch.tensor([s[2] for s in samples])
        return xs, ys, ts
