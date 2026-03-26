from __future__ import annotations

"""Task-specific decoder heads for continual learning."""

import torch
import torch.nn as nn


class MultiHeadDecoder(nn.Module):
    """Multi-head classifier: one linear head per task.

    Used in task-incremental learning where the task ID is known at test time.
    New heads are added dynamically as tasks arrive.
    """

    def __init__(self, latent_dim: int, classes_per_task: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.classes_per_task = classes_per_task
        self.heads = nn.ModuleDict()
        # Dummy param to track device placement
        self.register_buffer("_device_tracker", torch.tensor(0))

    @property
    def num_tasks(self) -> int:
        return len(self.heads)

    def add_task_head(self, task_id: int) -> None:
        key = str(task_id)
        if key not in self.heads:
            self.heads[key] = nn.Linear(self.latent_dim, self.classes_per_task).to(
                self._device_tracker.device
            )

    def forward(self, z: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.heads[str(task_id)](z)

    def forward_all(self, z: torch.Tensor) -> dict[int, torch.Tensor]:
        """Run all task heads on the same latent representation."""
        return {int(k): head(z) for k, head in self.heads.items()}


class SingleHeadDecoder(nn.Module):
    """Single-head classifier for class-incremental learning.

    A single linear layer maps from latent space to all classes seen so far.
    No task ID required at test time.
    """

    def __init__(self, latent_dim: int, total_classes: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.total_classes = total_classes
        self.head = nn.Linear(latent_dim, total_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)
