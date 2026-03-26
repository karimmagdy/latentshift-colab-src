from __future__ import annotations

"""Base class for continual learning methods."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ContinualLearningMethod(ABC):
    """Abstract base class that all CL methods must implement."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.current_task: int = -1

    def supports_checkpointing(self) -> bool:
        """Whether this method can be safely checkpointed and resumed."""
        return False

    def state_dict(self) -> dict:
        """Return serializable method state for checkpointing."""
        decoder_head_ids: list[int] = []
        if hasattr(self.decoder, "heads"):
            decoder_head_ids = sorted(int(task_id) for task_id in self.decoder.heads.keys())
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "decoder_head_ids": decoder_head_ids,
            "current_task": self.current_task,
            "extra_state": self._extra_state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore method state from a checkpoint."""
        for task_id in state.get("decoder_head_ids", []):
            if hasattr(self.decoder, "add_task_head"):
                self.decoder.add_task_head(int(task_id))
        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])
        self.current_task = state.get("current_task", -1)
        self._load_extra_state_dict(state.get("extra_state", {}))

    def _extra_state_dict(self) -> dict:
        return {}

    def _load_extra_state_dict(self, state: dict) -> None:
        del state

    @abstractmethod
    def prepare_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Called before training on a new task. Set up any task-specific state."""

    @abstractmethod
    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        epochs: int,
        lr: float,
    ) -> dict[str, float]:
        """Train on a single task. Returns a dict of training metrics."""

    @abstractmethod
    def after_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Called after training on a task. Update any continual learning state
        (e.g., compute Fisher for EWC, update subspace for LatentShift)."""

    def evaluate(self, task_id: int, test_loader: DataLoader) -> float:
        """Evaluate accuracy on a specific task."""
        self.encoder.eval()
        self.decoder.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                z = self.encoder(x)
                logits = self.decoder(z, task_id)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    def evaluate_class_incremental(
        self, test_loader: DataLoader, classes_per_task: int
    ) -> float:
        """Evaluate class-incremental accuracy using max-logit task inference.

        Runs all task heads, concatenates logits, and picks the global argmax.
        Labels in test_loader must be global class indices.
        """
        self.encoder.eval()
        self.decoder.eval()
        correct = 0
        total = 0
        num_heads = self.decoder.num_tasks
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                z = self.encoder(x)
                all_logits = self.decoder.forward_all(z)
                # Concatenate logits in task order → (N, num_heads * classes_per_task)
                cat_logits = torch.cat(
                    [all_logits[t] for t in range(num_heads)], dim=1
                )
                preds = cat_logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0
