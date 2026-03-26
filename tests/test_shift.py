"""Tests for the SubspaceTracker (shift operator) — the core of LatentShift."""

import torch
import pytest

from src.models.shift import SubspaceTracker


@pytest.fixture
def tracker():
    return SubspaceTracker(latent_dim=64, threshold=0.95)


def make_task_activations(n: int, d: int, rank: int, seed: int = 0) -> torch.Tensor:
    """Generate activations that lie in a rank-r subspace of R^d."""
    rng = torch.Generator().manual_seed(seed)
    basis = torch.randn(d, rank, generator=rng)
    Q, _ = torch.linalg.qr(basis)
    coeffs = torch.randn(n, rank, generator=rng)
    return coeffs @ Q[:, :rank].T


class TestSubspaceTracker:

    def test_initial_state(self, tracker):
        assert tracker.archive_rank == 0
        assert tracker.free_dim == 64
        assert tracker.num_tasks == 0

    def test_single_task_update(self, tracker):
        acts = make_task_activations(200, 64, rank=5, seed=42)
        r = tracker.update(acts)
        assert r > 0
        assert tracker.archive_rank == r
        assert tracker.free_dim == 64 - r
        assert tracker.num_tasks == 1

    def test_archive_basis_is_orthonormal(self, tracker):
        acts = make_task_activations(200, 64, rank=8, seed=1)
        tracker.update(acts)
        A = tracker.archive_basis
        gram = A.T @ A
        eye = torch.eye(gram.size(0))
        assert torch.allclose(gram, eye, atol=1e-5), "Archive basis is not orthonormal"

    def test_projection_orthogonality(self, tracker):
        """After archiving task 1, the projection should be orthogonal to the archive."""
        acts1 = make_task_activations(200, 64, rank=5, seed=10)
        tracker.update(acts1)

        P = tracker.get_projection_matrix()
        A = tracker.archive_basis

        # P @ A should be ~0 (projecting archive directions gives zero)
        residual = P @ A
        assert residual.abs().max() < 1e-5, "Projection is not orthogonal to archive"

    def test_zero_forgetting_property(self, tracker):
        """Core property: after archiving task 1 and projecting new gradients,
        the archived representations should be unaffected."""
        acts1 = make_task_activations(200, 64, rank=5, seed=20)
        tracker.update(acts1)

        projector = tracker.get_projector()

        # Simulate a gradient in the latent space
        grad = torch.randn(64)
        projected_grad = projector(grad.unsqueeze(0)).squeeze(0)

        # The projected gradient should have zero component in the archive
        A = tracker.archive_basis
        archive_component = A.T @ projected_grad
        assert archive_component.abs().max() < 1e-5, \
            "Projected gradient has nonzero component in archive subspace"

    def test_multi_task_accumulation(self, tracker):
        """Archive grows correctly across multiple tasks."""
        total_rank = 0
        for t in range(4):
            acts = make_task_activations(200, 64, rank=5, seed=t * 100)
            r = tracker.update(acts)
            total_rank += r
            assert tracker.archive_rank <= total_rank  # QR may merge overlapping directions
            assert tracker.num_tasks == t + 1

    def test_capacity_limit(self):
        """When archive is full, no more dimensions can be allocated."""
        tracker = SubspaceTracker(latent_dim=16, threshold=0.999)
        for t in range(10):
            acts = make_task_activations(200, 16, rank=5, seed=t * 100)
            tracker.update(acts)
        assert tracker.archive_rank <= 16
        assert tracker.free_dim >= 0

    def test_isometry_verification(self, tracker):
        """verify_isometry should show near-perfect overlap for in-subspace data."""
        acts = make_task_activations(200, 64, rank=5, seed=30)
        tracker.update(acts)
        result = tracker.verify_isometry(acts)
        assert result["overlap"] > 0.9, f"Overlap too low: {result['overlap']}"

    def test_lossy_compression(self):
        tracker = SubspaceTracker(latent_dim=32, threshold=0.95)
        acts = make_task_activations(200, 32, rank=15, seed=40)
        tracker.update(acts)
        old_rank = tracker.archive_rank

        error = tracker.compress(target_rank=5)
        assert tracker.archive_rank == 5
        assert error > 0

    def test_projector_preserves_free_directions(self, tracker):
        """Vectors in the free subspace should pass through the projector unchanged."""
        acts = make_task_activations(200, 64, rank=5, seed=50)
        tracker.update(acts)

        P = tracker.get_projection_matrix()
        # Generate a random vector in the free subspace
        free_vec = P @ torch.randn(64)
        projector = tracker.get_projector()
        result = projector(free_vec.unsqueeze(0)).squeeze(0)
        assert torch.allclose(result, free_vec, atol=1e-5), \
            "Projector should not modify vectors in the free subspace"


class TestProjector:

    def test_empty_archive(self):
        projector = SubspaceTracker(latent_dim=32, threshold=0.95).get_projector()
        v = torch.randn(1, 32)
        result = projector(v)
        assert torch.allclose(result, v), "Empty archive projector should be identity"

    def test_idempotent(self):
        tracker = SubspaceTracker(latent_dim=32, threshold=0.95)
        acts = make_task_activations(100, 32, rank=5, seed=60)
        tracker.update(acts)

        projector = tracker.get_projector()
        v = torch.randn(1, 32)
        p1 = projector(v)
        p2 = projector(p1)
        assert torch.allclose(p1, p2, atol=1e-5), "Projection should be idempotent"


class TestTaskInference:
    """Tests for per-task bases and task inference (class-incremental support)."""

    def test_task_bases_stored(self):
        tracker = SubspaceTracker(latent_dim=64, threshold=0.95)
        acts1 = make_task_activations(200, 64, rank=5, seed=10)
        acts2 = make_task_activations(200, 64, rank=5, seed=20)
        tracker.update(acts1)
        tracker.update(acts2)
        assert len(tracker.task_bases) == 2
        for V in tracker.task_bases:
            assert V.shape[0] == 64
            assert V.shape[1] > 0

    def test_task_bases_orthonormal(self):
        tracker = SubspaceTracker(latent_dim=64, threshold=0.95)
        acts = make_task_activations(200, 64, rank=5, seed=30)
        tracker.update(acts)
        V = tracker.task_bases[0]
        gram = V.T @ V
        assert torch.allclose(gram, torch.eye(gram.size(0)), atol=1e-5)

    def test_membership_scores_shape(self):
        tracker = SubspaceTracker(latent_dim=64, threshold=0.95)
        for t in range(3):
            acts = make_task_activations(200, 64, rank=5, seed=t * 100)
            tracker.update(acts)
        z = torch.randn(10, 64)
        scores = tracker.task_membership_scores(z)
        assert scores.shape == (10, 3)
        assert (scores >= 0).all()
        assert (scores <= 1 + 1e-5).all()

    def test_infer_task_shape(self):
        tracker = SubspaceTracker(latent_dim=64, threshold=0.95)
        for t in range(3):
            acts = make_task_activations(200, 64, rank=5, seed=t * 100)
            tracker.update(acts)
        z = torch.randn(10, 64)
        pred = tracker.infer_task(z)
        assert pred.shape == (10,)
        assert pred.min() >= 0
        assert pred.max() <= 2

    def test_infer_task_correct_for_in_subspace_data(self):
        """Data generated in task 0's subspace should be inferred as task 0."""
        tracker = SubspaceTracker(latent_dim=64, threshold=0.999)
        acts0 = make_task_activations(200, 64, rank=5, seed=10)
        acts1 = make_task_activations(200, 64, rank=5, seed=20)
        tracker.update(acts0)
        tracker.update(acts1)

        # Generate test data purely in task 0's subspace
        V0 = tracker.task_bases[0]
        test_z = torch.randn(50, V0.shape[1]) @ V0.T
        preds = tracker.infer_task(test_z)
        # Most should be task 0
        accuracy = (preds == 0).float().mean()
        assert accuracy > 0.7, f"Task inference accuracy too low: {accuracy}"

    def test_empty_tracker_membership(self):
        tracker = SubspaceTracker(latent_dim=32, threshold=0.95)
        z = torch.randn(5, 32)
        scores = tracker.task_membership_scores(z)
        assert scores.shape == (5, 0)
