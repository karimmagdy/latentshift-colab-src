from __future__ import annotations

"""Continual learning evaluation metrics."""

import numpy as np


class ContinualMetrics:
    """Tracks per-task accuracy across the learning sequence and computes
    standard CL metrics: average accuracy, forgetting, forward/backward transfer.

    Usage:
        metrics = ContinualMetrics()
        # After training task t, evaluate on all tasks seen so far:
        for t_eval in range(t + 1):
            acc = method.evaluate(t_eval, test_loaders[t_eval])
            metrics.log(current_task=t, eval_task=t_eval, accuracy=acc)
        # After all tasks:
        print(metrics.summary())
    """

    def __init__(self):
        # acc_matrix[i][j] = accuracy on task j after training on task i
        self.acc_matrix: dict[int, dict[int, float]] = {}
        # Per-task extra info (e.g., archive_rank for LatentShift)
        self.extras: dict[int, dict[str, float]] = {}

    def log(self, current_task: int, eval_task: int, accuracy: float) -> None:
        if current_task not in self.acc_matrix:
            self.acc_matrix[current_task] = {}
        self.acc_matrix[current_task][eval_task] = accuracy

    def log_extra(self, task_id: int, **kwargs: float) -> None:
        """Log extra per-task metadata (e.g., archive_rank, free_dim)."""
        self.extras[task_id] = kwargs

    def get_accuracy_matrix(self) -> np.ndarray:
        """Return the full T×T accuracy matrix as a numpy array."""
        tasks = sorted(self.acc_matrix.keys())
        T = len(tasks)
        mat = np.zeros((T, T))
        for i, t_train in enumerate(tasks):
            for j, t_eval in enumerate(tasks):
                mat[i, j] = self.acc_matrix[t_train].get(t_eval, 0.0)
        return mat

    def average_accuracy(self) -> float:
        """Average accuracy after the last task, across all tasks."""
        tasks = sorted(self.acc_matrix.keys())
        if not tasks:
            return 0.0
        last = tasks[-1]
        accs = list(self.acc_matrix[last].values())
        return float(np.mean(accs))

    def average_forgetting(self) -> float:
        """Average forgetting: mean drop in accuracy for each task between its
        peak performance and final performance.

        F = (1/T-1) * sum_{j=0}^{T-2} [max_{t>=j} a_{t,j} - a_{T-1,j}]
        """
        tasks = sorted(self.acc_matrix.keys())
        T = len(tasks)
        if T <= 1:
            return 0.0

        mat = self.get_accuracy_matrix()
        forgetting = []
        for j in range(T - 1):
            peak = max(mat[i, j] for i in range(j, T))
            final = mat[T - 1, j]
            forgetting.append(peak - final)
        return float(np.mean(forgetting))

    def backward_transfer(self) -> float:
        """BWT = (1/T-1) * sum_{j=0}^{T-2} [a_{T-1,j} - a_{j,j}]"""
        tasks = sorted(self.acc_matrix.keys())
        T = len(tasks)
        if T <= 1:
            return 0.0
        mat = self.get_accuracy_matrix()
        bwt = [mat[T - 1, j] - mat[j, j] for j in range(T - 1)]
        return float(np.mean(bwt))

    def forward_transfer(self) -> float:
        """FWT = (1/T-1) * sum_{j=1}^{T-1} [a_{j-1,j} - random_baseline]
        We use 0 as the random baseline (no prior knowledge)."""
        tasks = sorted(self.acc_matrix.keys())
        T = len(tasks)
        if T <= 1:
            return 0.0
        mat = self.get_accuracy_matrix()
        # a_{j-1, j} is accuracy on task j right before training on it
        # (after training on task j-1). This requires evaluating on unseen tasks.
        # If not logged, return 0.
        fwt = []
        for j in range(1, T):
            if j in self.acc_matrix.get(j - 1, {}):
                fwt.append(mat[j - 1, j])
        return float(np.mean(fwt)) if fwt else 0.0

    def summary(self) -> dict:
        result = {
            "average_accuracy": self.average_accuracy(),
            "average_forgetting": self.average_forgetting(),
            "backward_transfer": self.backward_transfer(),
            "forward_transfer": self.forward_transfer(),
        }
        if self.extras:
            result["per_task_extras"] = {
                str(k): v for k, v in self.extras.items()
            }
        return result

    def to_dict(self) -> dict:
        return {
            "acc_matrix": {
                str(task_id): {str(eval_task): acc for eval_task, acc in row.items()}
                for task_id, row in self.acc_matrix.items()
            },
            "extras": {str(task_id): values for task_id, values in self.extras.items()},
        }

    @classmethod
    def from_dict(cls, state: dict) -> "ContinualMetrics":
        metrics = cls()
        for task_id, row in state.get("acc_matrix", {}).items():
            metrics.acc_matrix[int(task_id)] = {
                int(eval_task): float(acc) for eval_task, acc in row.items()
            }
        for task_id, values in state.get("extras", {}).items():
            metrics.extras[int(task_id)] = values
        return metrics

    def print_matrix(self) -> None:
        mat = self.get_accuracy_matrix()
        T = mat.shape[0]
        header = "       " + "".join(f"Task {j:>2}  " for j in range(T))
        print(header)
        for i in range(T):
            row = f"After {i}: " + "".join(f" {mat[i,j]:.4f} " for j in range(T))
            print(row)
