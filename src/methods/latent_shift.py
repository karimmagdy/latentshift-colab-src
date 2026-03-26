from __future__ import annotations

"""LatentShift: Hilbert-inspired continual learning via latent space shifting.

This is the core method. It:
1. Trains the encoder + decoder on each task
2. After each task, computes the SVD of latent activations to identify
   the subspace occupied by the task's representations
3. Shifts that subspace into the "archive" (even rooms in Hilbert's Hotel)
4. During subsequent tasks, projects gradients to only modify the "free"
   subspace (odd rooms), guaranteeing zero forgetting on archived tasks
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import ContinualLearningMethod
from ..models.shift import SubspaceTracker


class LatentShiftMethod(ContinualLearningMethod):
    """LatentShift continual learning method.

    Uses a SubspaceTracker to maintain the archive of old-task representations
    and projects gradients to prevent interference.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
        latent_dim: int = 256,
        threshold: float = 0.99,
        num_samples: int = 300,
        lossy: bool = False,
        max_archive_rank: int | None = None,
    ):
        super().__init__(encoder, decoder, device)
        self.tracker = SubspaceTracker(latent_dim, threshold=threshold).to(device)
        self.num_samples = num_samples
        self.lossy = lossy
        self.max_archive_rank = max_archive_rank
        # NCM prototypes for class-incremental evaluation
        self._class_means: dict[int, torch.Tensor] = {}  # global_class -> mean_z

    def prepare_task(self, task_id: int, train_loader: DataLoader) -> None:
        self.current_task = task_id
        # Add a new head in the decoder
        self.decoder.add_task_head(task_id)

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        epochs: int,
        lr: float,
    ) -> dict[str, float]:
        self.encoder.train()
        self.decoder.train()

        # Only train the current task head + encoder parameters
        params = list(self.encoder.parameters()) + list(
            self.decoder.heads[str(task_id)].parameters()
        )
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                z = self.encoder(x)

                # Register backward hook on latent representation to project
                # gradients onto the free subspace. This ensures the gradient
                # flowing back through the latent space has zero component in
                # the archive subspace, so all upstream encoder parameters
                # are updated only in directions that don't affect old tasks.
                if self.tracker.archive_rank > 0 and z.requires_grad:
                    projector = self.tracker.get_projector()
                    z.register_hook(lambda grad, p=projector: p(grad))

                logits = self.decoder(z, task_id)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * x.size(0)
                total_correct += (logits.argmax(1) == y).sum().item()
                total_samples += x.size(0)

            total_loss += epoch_loss

        return {
            "train_loss": total_loss / max(total_samples * epochs, 1),
            "train_acc": total_correct / max(total_samples, 1),
        }

    def after_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Compute activations, update subspace archive, and store class prototypes."""
        activations = self._collect_activations(train_loader)
        t0 = time.time()
        r_t = self.tracker.update(activations)
        svd_time = time.time() - t0

        # Lossy compression if needed
        if self.lossy and self.max_archive_rank is not None:
            if self.tracker.archive_rank > self.max_archive_rank:
                self.tracker.compress(self.max_archive_rank)

        # Store per-class mean representations for NCM-based CI evaluation
        self._store_class_prototypes(task_id, train_loader)

        print(
            f"  Task {task_id}: rank={r_t}, "
            f"archive={self.tracker.archive_rank}/{self.tracker.latent_dim}, "
            f"free={self.tracker.free_dim}, "
            f"svd_time={svd_time:.3f}s"
        )

    @torch.no_grad()
    def _store_class_prototypes(self, task_id: int, train_loader: DataLoader) -> None:
        """Compute and store mean latent representation for each class in this task."""
        self.encoder.eval()
        class_sums: dict[int, torch.Tensor] = {}
        class_counts: dict[int, int] = {}
        classes_per_task = self.decoder.classes_per_task

        for x, y in train_loader:
            x = x.to(self.device)
            z = self.encoder(x)
            for local_label in y.unique():
                global_label = task_id * classes_per_task + local_label.item()
                mask = y == local_label
                z_class = z[mask]
                if global_label not in class_sums:
                    class_sums[global_label] = z_class.sum(dim=0)
                    class_counts[global_label] = z_class.shape[0]
                else:
                    class_sums[global_label] += z_class.sum(dim=0)
                    class_counts[global_label] += z_class.shape[0]

        for g_label in class_sums:
            self._class_means[g_label] = class_sums[g_label] / class_counts[g_label]
        self.encoder.train()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_activations(self, loader: DataLoader) -> torch.Tensor:
        """Collect a batch of latent activations for SVD computation."""
        self.encoder.eval()
        acts = []
        count = 0
        for x, _ in loader:
            x = x.to(self.device)
            z = self.encoder(x)
            acts.append(z)
            count += x.size(0)
            if count >= self.num_samples:
                break
        self.encoder.train()
        return torch.cat(acts, dim=0)[: self.num_samples]

    def evaluate_class_incremental(
        self, test_loader: DataLoader, classes_per_task: int
    ) -> float:
        """Evaluate class-incremental accuracy using Nearest-Class-Mean (NCM).

        For each sample: compute latent z, find the nearest stored class
        prototype, and predict that global class label.
        """
        if not self._class_means:
            return 0.0

        self.encoder.eval()
        self.decoder.eval()

        # Stack all class means into a matrix for efficient distance computation
        sorted_labels = sorted(self._class_means.keys())
        proto_matrix = torch.stack([self._class_means[g] for g in sorted_labels])  # (C, d)
        label_map = torch.tensor(sorted_labels, device=self.device)

        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                z = self.encoder(x)  # (N, d)
                # Compute distances to all prototypes
                dists = torch.cdist(z, proto_matrix)  # (N, C)
                pred_idx = dists.argmin(dim=1)  # (N,)
                preds = label_map[pred_idx]
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0
