from __future__ import annotations

"""Gradient projection optimizer wrapper for LatentShift.

Wraps any standard optimizer and applies gradient projection onto the
free subspace after each backward pass, before the optimizer step.
"""

import torch
from torch.optim import Optimizer

from ..models.shift import SubspaceTracker, _Projector


class ProjectedOptimizer:
    """Wraps a standard optimizer and projects gradients onto the free subspace
    before each step.

    Usage:
        base_opt = torch.optim.SGD(params, lr=0.01)
        opt = ProjectedOptimizer(base_opt, tracker, encoder)
        ...
        loss.backward()
        opt.step()  # automatically projects gradients before stepping
        opt.zero_grad()
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        tracker: SubspaceTracker,
        encoder: torch.nn.Module,
    ):
        self.base_optimizer = base_optimizer
        self.tracker = tracker
        self.encoder = encoder

    def step(self) -> None:
        """Project encoder gradients, then step the base optimizer."""
        if self.tracker.archive_rank > 0:
            projector = self.tracker.get_projector()
            for param in self.encoder.parameters():
                if param.grad is not None:
                    g = param.grad.data
                    shape = g.shape
                    if g.dim() >= 2:
                        g_2d = g.view(g.size(0), -1)
                        param.grad.data = projector(g_2d).view(shape)
                    else:
                        param.grad.data = projector(g.unsqueeze(0)).squeeze(0)
        self.base_optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups
