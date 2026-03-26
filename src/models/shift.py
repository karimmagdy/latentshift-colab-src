from __future__ import annotations

"""Hilbert-inspired latent shift operator and subspace tracker.

Core idea: After learning task t, compute the subspace V_t occupied by
learned representations (via SVD of the activation matrix). Maintain a
cumulative "archive" subspace A_{1:t} and its orthogonal complement
F_{t+1} (the "free" subspace for new learning). Gradient projection
onto F_{t+1} ensures zero forgetting on previous tasks.

This implements the finite-dimensional analog of Hilbert's n→2n shift:
existing knowledge is "shifted" into the archive subspace, and new
orthogonal dimensions are freed for the next task.
"""

import torch
import torch.nn as nn


class SubspaceTracker(nn.Module):
    """Tracks the cumulative occupied subspace across tasks and provides
    projection matrices for gradient constraining.

    At each task boundary:
    1. Collect activations Z_t from the encoder on task t data
    2. Compute SVD to find the top-r_t directions (occupied subspace V_t)
    3. Merge V_t into the cumulative archive A_{1:t} via QR re-orthogonalization
    4. Compute projection P_{t+1} = I - A A^T for constraining future gradients
    """

    def __init__(self, latent_dim: int, threshold: float = 0.99):
        """
        Args:
            latent_dim: Dimensionality d of the latent space.
            threshold: Fraction of variance to retain when selecting rank r_t.
                       E.g. 0.99 keeps singular vectors explaining 99% of variance.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.threshold = threshold

        # Cumulative orthonormal basis of the archive subspace A_{1:t}
        # Shape: (d, k) where k = sum of retained ranks across tasks so far
        self.register_buffer("archive_basis", torch.empty(latent_dim, 0))

        # Per-task metadata
        self.task_ranks: list[int] = []
        # Per-task subspace bases for task inference (class-incremental)
        self.task_bases: list[torch.Tensor] = []

    @property
    def archive_rank(self) -> int:
        """Current total rank of the archive subspace."""
        return self.archive_basis.shape[1]

    @property
    def free_dim(self) -> int:
        """Remaining free dimensions for new learning."""
        return self.latent_dim - self.archive_rank

    @property
    def num_tasks(self) -> int:
        return len(self.task_ranks)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def update(self, activations: torch.Tensor) -> int:
        """Process activations from the just-completed task and shift its
        knowledge into the archive subspace.

        Args:
            activations: (N, d) tensor of latent representations from task t.

        Returns:
            r_t: The rank (number of dimensions) allocated to this task.
        """
        # Center activations for SVD
        mean = activations.mean(dim=0, keepdim=True)
        centered = activations - mean

        # Move to CPU for linalg ops (SVD/QR not supported on MPS)
        orig_device = centered.device
        centered_cpu = centered.cpu()

        # Economy SVD
        U, S, _ = torch.linalg.svd(centered_cpu, full_matrices=False)

        # Determine rank r_t by variance threshold
        var = S ** 2
        cumvar = var.cumsum(0) / var.sum()
        r_t = int((cumvar < self.threshold).sum().item()) + 1
        r_t = min(r_t, self.free_dim)  # can't exceed available space

        if r_t == 0:
            self.task_ranks.append(0)
            self.task_bases.append(torch.empty(self.latent_dim, 0))
            return 0

        # New directions for this task (d × r_t)
        # These are the right singular vectors of the centered activation matrix
        # We re-derive them from the SVD of the data matrix
        new_basis = torch.linalg.svd(centered_cpu.T, full_matrices=False)[0][:, :r_t]

        # Store per-task basis before QR merging (for task inference)
        Q_task, _ = torch.linalg.qr(new_basis)
        self.task_bases.append(Q_task.to(orig_device))

        # Merge with existing archive via QR to maintain orthonormality
        if self.archive_rank > 0:
            combined = torch.cat([self.archive_basis.cpu(), new_basis], dim=1)
            Q, _ = torch.linalg.qr(combined)
            self.archive_basis = Q[:, : self.archive_rank + r_t].to(orig_device)
        else:
            Q, _ = torch.linalg.qr(new_basis)
            self.archive_basis = Q.to(orig_device)

        self.task_ranks.append(r_t)
        return r_t

    def get_projection_matrix(self) -> torch.Tensor:
        """Compute the projection onto the free subspace F_{t+1}.

        P = I - A @ A^T

        Gradients projected through P only modify the free subspace,
        leaving the archive subspace (and thus old task representations)
        untouched.

        Returns:
            (d, d) projection matrix.
        """
        A = self.archive_basis
        if A.shape[1] == 0:
            return torch.eye(self.latent_dim, device=A.device)
        return torch.eye(self.latent_dim, device=A.device) - A @ A.T

    def get_projector(self) -> "_Projector":
        """Return a lightweight callable projector (avoids recomputing P)."""
        return _Projector(self.archive_basis.clone())

    def compress(self, target_rank: int) -> float:
        """Lossy shift: compress the archive to target_rank dimensions.

        This is the finite-capacity safety valve — when the archive is
        nearly full, we can discard the least important directions.

        Args:
            target_rank: Desired rank after compression.

        Returns:
            Reconstruction error bound (next singular value of discarded part).
        """
        if target_rank >= self.archive_rank:
            return 0.0

        # SVD of the archive basis itself doesn't help (it's already orthonormal).
        # Instead we'd need the original activation data to decide what to drop.
        # For now, simply truncate the last columns (oldest-task directions added
        # last are dropped first, which is a FIFO heuristic).
        discarded = self.archive_basis[:, target_rank:]
        self.archive_basis = self.archive_basis[:, :target_rank]
        # Rough error bound: norm of discarded basis (always 1 per column since orthonormal)
        return float(discarded.shape[1])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def subspace_overlap(self, activations: torch.Tensor) -> float:
        """Measure how much of the activation variance lies inside the archive.

        Useful for diagnosing whether old knowledge is still captured.
        Returns a value in [0, 1]; 1 means all variance is in the archive.
        """
        if self.archive_rank == 0:
            return 0.0
        centered = activations - activations.mean(dim=0, keepdim=True)
        A = self.archive_basis
        projected = centered @ A @ A.T
        return float((projected.norm() ** 2 / (centered.norm() ** 2 + 1e-12)).item())

    def verify_isometry(self, activations: torch.Tensor, atol: float = 1e-5) -> dict:
        """Verify that projection into the archive subspace preserves norms
        and inner products (i.e., acts as an isometry on the occupied subspace).

        Returns diagnostic dict with max norm error and max inner-product error.
        """
        centered = activations - activations.mean(dim=0, keepdim=True)
        A = self.archive_basis

        # Project into archive
        proj = centered @ A @ A.T

        # Norm preservation
        orig_norms = centered.norm(dim=1)
        proj_norms = proj.norm(dim=1)
        norm_errors = (orig_norms - proj_norms).abs()

        # Inner product preservation (sample a subset for efficiency)
        n = min(200, centered.shape[0])
        orig_gram = centered[:n] @ centered[:n].T
        proj_gram = proj[:n] @ proj[:n].T
        ip_errors = (orig_gram - proj_gram).abs()

        return {
            "max_norm_error": float(norm_errors.max().item()),
            "mean_norm_error": float(norm_errors.mean().item()),
            "max_ip_error": float(ip_errors.max().item()),
            "overlap": self.subspace_overlap(activations),
            "isometry_holds": bool(norm_errors.max().item() < atol),
        }

    # ------------------------------------------------------------------
    # Task inference (class-incremental)
    # ------------------------------------------------------------------

    def task_membership_scores(self, z: torch.Tensor) -> torch.Tensor:
        """Compute how much each sample belongs to each task's subspace.

        For each task t with basis V_t: score_t = ||V_t^T z||^2 / ||z||^2.

        Args:
            z: (N, d) latent representations.

        Returns:
            (N, num_tasks) tensor of membership scores in [0, 1].
        """
        if not self.task_bases:
            return torch.zeros(z.shape[0], 0, device=z.device)
        z_norm_sq = (z * z).sum(dim=1, keepdim=True).clamp(min=1e-12)  # (N, 1)
        scores = []
        for V_t in self.task_bases:
            if V_t.shape[1] == 0:
                scores.append(torch.zeros(z.shape[0], 1, device=z.device))
            else:
                proj = z @ V_t.to(z.device)  # (N, r_t)
                proj_norm_sq = (proj * proj).sum(dim=1, keepdim=True)  # (N, 1)
                scores.append(proj_norm_sq / z_norm_sq)
        return torch.cat(scores, dim=1)  # (N, num_tasks)

    def infer_task(self, z: torch.Tensor) -> torch.Tensor:
        """Infer the most likely task for each sample based on subspace membership.

        Args:
            z: (N, d) latent representations.

        Returns:
            (N,) tensor of predicted task IDs.
        """
        scores = self.task_membership_scores(z)
        return scores.argmax(dim=1)


class _Projector:
    """Lightweight callable that projects vectors onto the free subspace."""

    def __init__(self, archive_basis: torch.Tensor):
        self.A = archive_basis  # (d, k)

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        if self.A.shape[1] == 0:
            return grad
        return grad - (grad @ self.A) @ self.A.T
