from __future__ import annotations

"""Dark Experience Replay++ (DER++) baseline.

Buzzega et al., "Dark Experience for General Continual Learning: a Strong,
Simple Baseline" (NeurIPS 2020).

Extends ER by also storing and replaying the model's logits at storage time.
Loss = CE(current) + alpha * MSE(replay_logits, stored_logits) + beta * CE(replay, labels)
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base import ContinualLearningMethod


class DERPlusPlus(ContinualLearningMethod):
    """DER++ continual learning baseline.

    Args:
        buffer_size_per_task: Number of samples to store per task.
        replay_batch_size: Batch size for replay sampling.
        alpha: Weight for logit distillation loss (MSE on stored logits).
        beta: Weight for replay classification loss.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
        buffer_size_per_task: int = 200,
        replay_batch_size: int = 64,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        super().__init__(encoder, decoder, device)
        self.buffer_size_per_task = buffer_size_per_task
        self.replay_batch_size = replay_batch_size
        self.alpha = alpha
        self.beta = beta
        # Buffer: list of (x_cpu, y_cpu, task_id, logits_cpu) tuples
        self._buffer: list[tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]] = []
        self._tasks_seen: int = 0

    def prepare_task(self, task_id: int, train_loader: DataLoader) -> None:
        self.current_task = task_id
        self.decoder.add_task_head(task_id)

    def train_task(
        self, task_id: int, train_loader: DataLoader, epochs: int, lr: float
    ) -> dict[str, float]:
        self.encoder.train()
        self.decoder.train()

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

                z = self.encoder(x)
                logits = self.decoder(z, task_id)
                loss = criterion(logits, y)

                if self._buffer:
                    rx, ry, rt, r_stored_logits = self._sample_replay()
                    rx = rx.to(self.device)
                    ry = ry.to(self.device)
                    r_stored_logits = r_stored_logits.to(self.device)

                    rz = self.encoder(rx)

                    # Logit distillation loss (alpha term)
                    distill_loss = torch.tensor(0.0, device=self.device)
                    # Classification replay loss (beta term)
                    replay_ce_loss = torch.tensor(0.0, device=self.device)

                    unique_tasks = rt.unique()
                    num_replay_tasks = len(unique_tasks)
                    for t in unique_tasks:
                        mask = rt == t
                        t_int = t.item()
                        r_logits_now = self.decoder(rz[mask], t_int)
                        # Soft distillation: MSE on softmax outputs to avoid
                        # raw-logit magnitude issues with small per-task heads
                        distill_loss = distill_loss + F.mse_loss(
                            F.softmax(r_logits_now, dim=1),
                            r_stored_logits[mask],
                        )
                        # CE on replay labels
                        replay_ce_loss = replay_ce_loss + criterion(
                            r_logits_now, ry[mask]
                        )

                    # Normalize by number of replay tasks
                    distill_loss = distill_loss / max(num_replay_tasks, 1)
                    replay_ce_loss = replay_ce_loss / max(num_replay_tasks, 1)

                    loss = loss + self.alpha * distill_loss + self.beta * replay_ce_loss

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
        """Store current-task data with logits into the buffer."""
        self._tasks_seen += 1
        self.encoder.eval()
        self.decoder.eval()

        all_x, all_y, all_probs = [], [], []
        with torch.no_grad():
            for x, y in train_loader:
                x_dev = x.to(self.device)
                z = self.encoder(x_dev)
                logits = self.decoder(z, task_id)
                all_x.append(x)
                all_y.append(y)
                all_probs.append(F.softmax(logits, dim=1).cpu())

        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        all_probs = torch.cat(all_probs, dim=0)

        n = all_x.size(0)
        k = min(n, self.buffer_size_per_task)
        indices = torch.randperm(n)[:k]
        for i in indices:
            self._buffer.append(
                (all_x[i].cpu(), all_y[i].cpu(), task_id, all_probs[i].cpu())
            )

        max_total = self.buffer_size_per_task * self._tasks_seen
        if len(self._buffer) > max_total:
            self._buffer = random.sample(self._buffer, max_total)

        self.encoder.train()
        print(f"  DER++ buffer: {len(self._buffer)} samples ({self._tasks_seen} tasks)")

    def _refresh_stored_logits(self) -> None:
        """Re-compute stored softmax probabilities with the current encoder.

        This prevents the distillation targets from becoming stale as
        the encoder evolves across tasks."""
        if not self._buffer:
            return
        self.encoder.eval()
        self.decoder.eval()
        new_buffer = []
        with torch.no_grad():
            for x_cpu, y_cpu, tid, _ in self._buffer:
                x_dev = x_cpu.unsqueeze(0).to(self.device)
                z = self.encoder(x_dev)
                probs = F.softmax(self.decoder(z, tid), dim=1).squeeze(0).cpu()
                new_buffer.append((x_cpu, y_cpu, tid, probs))
        self._buffer = new_buffer
        self.encoder.train()
        self.decoder.train()

    def _sample_replay(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch from the replay buffer."""
        k = min(len(self._buffer), self.replay_batch_size)
        samples = random.sample(self._buffer, k)
        xs = torch.stack([s[0] for s in samples])
        ys = torch.stack([s[1] for s in samples])
        ts = torch.tensor([s[2] for s in samples])
        logits = torch.stack([s[3] for s in samples])
        return xs, ys, ts, logits
