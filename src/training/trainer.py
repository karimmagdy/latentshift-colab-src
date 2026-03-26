from __future__ import annotations

"""Continual learning training loop."""

import time
from typing import Callable, Protocol

from torch.utils.data import DataLoader

from ..methods.base import ContinualLearningMethod
from ..utils.metrics import ContinualMetrics


class Benchmark(Protocol):
    num_tasks: int
    classes_per_task: int
    def get_task_loaders(self, task_id: int) -> tuple[DataLoader, DataLoader]: ...


def run_continual_learning(
    method: ContinualLearningMethod,
    benchmark: Benchmark,
    epochs_per_task: int = 10,
    lr: float = 0.01,
    verbose: bool = True,
    start_task: int = 0,
    metrics: ContinualMetrics | None = None,
    checkpoint_callback: Callable[[int, ContinualMetrics], None] | None = None,
) -> ContinualMetrics:
    """Run the full continual learning loop.

    For each task:
      1. prepare_task  — set up task-specific state
      2. train_task    — train the model
      3. after_task    — update CL state (e.g., subspace archive)
      4. evaluate      — evaluate on all tasks seen so far

    Returns a ContinualMetrics object with the full accuracy matrix.
    """
    metrics = metrics or ContinualMetrics()

    for task_id in range(start_task, benchmark.num_tasks):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Task {task_id}")
            print(f"{'='*60}")

        train_loader, test_loader = benchmark.get_task_loaders(task_id)

        # 1. Prepare
        method.prepare_task(task_id, train_loader)

        # 2. Train
        t0 = time.time()
        train_metrics = method.train_task(task_id, train_loader, epochs_per_task, lr)
        train_elapsed = time.time() - t0
        if verbose:
            print(f"  Train: loss={train_metrics.get('train_loss', 0):.4f} "
                  f"acc={train_metrics.get('train_acc', 0):.4f} ({train_elapsed:.1f}s)")

        # 3. After task
        t_after = time.time()
        method.after_task(task_id, train_loader)
        after_elapsed = time.time() - t_after

        # Track method-specific state (e.g., archive rank for LatentShift)
        extra_kwargs: dict[str, float] = {
            "wall_clock_train": train_elapsed,
            "wall_clock_after_task": after_elapsed,
        }
        if hasattr(method, "tracker"):
            extra_kwargs["archive_rank"] = method.tracker.archive_rank
            extra_kwargs["free_dim"] = method.tracker.free_dim
        metrics.log_extra(task_id, **extra_kwargs)

        # 4. Evaluate on all tasks (including future ones for FWT)
        if verbose:
            print("  Eval:", end="")
        for t_eval in range(benchmark.num_tasks):
            _, t_test_loader = benchmark.get_task_loaders(t_eval)
            # For future tasks, the decoder head may not exist yet
            if hasattr(method.decoder, 'heads') and str(t_eval) not in method.decoder.heads:
                continue
            try:
                acc = method.evaluate(t_eval, t_test_loader)
            except (KeyError, RuntimeError):
                continue
            metrics.log(current_task=task_id, eval_task=t_eval, accuracy=acc)
            if verbose:
                print(f"  T{t_eval}={acc:.4f}", end="")
        if verbose:
            print()

        if checkpoint_callback is not None:
            checkpoint_callback(task_id, metrics)

    if verbose:
        print(f"\n{'='*60}")
        print("Final Results")
        print(f"{'='*60}")
        metrics.print_matrix()
        summary = metrics.summary()
        print(f"\nAverage Accuracy:   {summary['average_accuracy']:.4f}")
        print(f"Average Forgetting: {summary['average_forgetting']:.4f}")
        print(f"Backward Transfer:  {summary['backward_transfer']:.4f}")

    return metrics
