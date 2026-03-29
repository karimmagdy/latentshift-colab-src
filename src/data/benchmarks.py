from __future__ import annotations

"""Standard continual learning benchmarks."""

import os
from typing import Sequence

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, TensorDataset
from PIL import Image

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _class_split(dataset, classes_per_task: int) -> list[list[int]]:
    """Group dataset indices by class, then chunk into tasks."""
    targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    all_classes = sorted(targets.unique().tolist())
    tasks = []
    for i in range(0, len(all_classes), classes_per_task):
        task_classes = all_classes[i : i + classes_per_task]
        mask = sum(targets == c for c in task_classes).bool()
        indices = mask.nonzero(as_tuple=False).squeeze(1).tolist()
        tasks.append(indices)
    return tasks


def _relabel(targets: torch.Tensor, classes: Sequence[int]) -> torch.Tensor:
    """Map original class labels to 0..len(classes)-1 within a task."""
    mapping = {c: i for i, c in enumerate(sorted(classes))}
    return torch.tensor([mapping[int(t)] for t in targets])


# ---------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------

class _SplitBenchmark:
    """Base class for split-dataset benchmarks."""

    def __init__(
        self,
        dataset_cls,
        transform,
        classes_per_task: int,
        data_root: str = "./data",
        batch_size: int = 64,
    ):
        self.batch_size = batch_size
        self.classes_per_task = classes_per_task

        train_ds = dataset_cls(root=data_root, train=True, download=True, transform=transform)
        test_ds = dataset_cls(root=data_root, train=False, download=True, transform=transform)

        self.train_splits = _class_split(train_ds, classes_per_task)
        self.test_splits = _class_split(test_ds, classes_per_task)

        self._train_ds = train_ds
        self._test_ds = test_ds
        self._loader_cache: dict[int, tuple[DataLoader, DataLoader]] = {}

    @property
    def num_tasks(self) -> int:
        return len(self.train_splits)

    def get_task_loaders(self, task_id: int) -> tuple[DataLoader, DataLoader]:
        """Return (train_loader, test_loader) for a given task.

        Labels are remapped to 0..classes_per_task-1 within each task.
        Results are cached to avoid repeated tensor materialization.
        """
        if task_id in self._loader_cache:
            return self._loader_cache[task_id]

        tr_indices = self.train_splits[task_id]
        te_indices = self.test_splits[task_id]

        # Gather data and relabel
        tr_x = torch.stack([self._train_ds[i][0] for i in tr_indices])
        tr_y_orig = torch.tensor([self._train_ds[i][1] for i in tr_indices])
        task_classes = sorted(tr_y_orig.unique().tolist())
        tr_y = _relabel(tr_y_orig, task_classes)

        te_x = torch.stack([self._test_ds[i][0] for i in te_indices])
        te_y_orig = torch.tensor([self._test_ds[i][1] for i in te_indices])
        te_y = _relabel(te_y_orig, task_classes)

        train_loader = DataLoader(
            TensorDataset(tr_x, tr_y), batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(te_x, te_y), batch_size=self.batch_size, shuffle=False
        )
        self._loader_cache[task_id] = (train_loader, test_loader)
        return train_loader, test_loader


class SplitMNIST(_SplitBenchmark):
    def __init__(self, data_root: str = "./data", batch_size: int = 64, classes_per_task: int = 2):
        transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        super().__init__(torchvision.datasets.MNIST, transform, classes_per_task, data_root, batch_size)


class SplitCIFAR10(_SplitBenchmark):
    def __init__(self, data_root: str = "./data", batch_size: int = 64, classes_per_task: int = 2):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        super().__init__(torchvision.datasets.CIFAR10, transform, classes_per_task, data_root, batch_size)


class SplitCIFAR100(_SplitBenchmark):
    def __init__(self, data_root: str = "./data", batch_size: int = 64, classes_per_task: int = 10):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        super().__init__(torchvision.datasets.CIFAR100, transform, classes_per_task, data_root, batch_size)


class PermutedMNIST:
    """Permuted-MNIST benchmark: each task applies a fixed random permutation to pixels."""

    def __init__(
        self,
        num_tasks: int = 10,
        data_root: str = "./data",
        batch_size: int = 64,
        seed: int = 42,
    ):
        self._num_tasks = num_tasks
        self.batch_size = batch_size
        self.classes_per_task = 10  # all 10 digits each time

        transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        self._train_ds = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        self._test_ds = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

        # Pre-generate permutations
        rng = torch.Generator().manual_seed(seed)
        self.permutations = [torch.randperm(784, generator=rng) for _ in range(num_tasks)]
        self._loader_cache: dict[int, tuple[DataLoader, DataLoader]] = {}

        # Pre-materialize base data once
        self._train_x = torch.stack([self._train_ds[i][0] for i in range(len(self._train_ds))]).view(len(self._train_ds), -1)
        self._train_y = torch.tensor([self._train_ds[i][1] for i in range(len(self._train_ds))])
        self._test_x = torch.stack([self._test_ds[i][0] for i in range(len(self._test_ds))]).view(len(self._test_ds), -1)
        self._test_y = torch.tensor([self._test_ds[i][1] for i in range(len(self._test_ds))])

    @property
    def num_tasks(self) -> int:
        return self._num_tasks

    def get_task_loaders(self, task_id: int) -> tuple[DataLoader, DataLoader]:
        if task_id in self._loader_cache:
            return self._loader_cache[task_id]
        perm = self.permutations[task_id]
        train_loader = DataLoader(
            TensorDataset(self._train_x[:, perm], self._train_y),
            batch_size=self.batch_size, shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(self._test_x[:, perm], self._test_y),
            batch_size=self.batch_size, shuffle=False,
        )
        self._loader_cache[task_id] = (train_loader, test_loader)
        return train_loader, test_loader


class ClassIncrementalWrapper:
    """Wraps a _SplitBenchmark to expose global class labels instead of per-task labels.

    In class-incremental learning, the model must classify among all classes
    seen so far without knowing the task ID. Labels are kept as global indices
    (e.g., 0-9 for CIFAR-10 rather than 0-1 per task).
    """

    def __init__(self, base_benchmark: _SplitBenchmark):
        self._base = base_benchmark

    @property
    def num_tasks(self) -> int:
        return self._base.num_tasks

    @property
    def classes_per_task(self) -> int:
        return self._base.classes_per_task

    @property
    def total_classes(self) -> int:
        return self.num_tasks * self.classes_per_task

    def get_task_loaders(self, task_id: int) -> tuple[DataLoader, DataLoader]:
        """Return (train_loader, test_loader) with global class labels."""
        tr_indices = self._base.train_splits[task_id]
        te_indices = self._base.test_splits[task_id]

        tr_x = torch.stack([self._base._train_ds[i][0] for i in tr_indices])
        tr_y = torch.tensor([self._base._train_ds[i][1] for i in tr_indices])

        te_x = torch.stack([self._base._test_ds[i][0] for i in te_indices])
        te_y = torch.tensor([self._base._test_ds[i][1] for i in te_indices])

        train_loader = DataLoader(
            TensorDataset(tr_x, tr_y), batch_size=self._base.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(te_x, te_y), batch_size=self._base.batch_size, shuffle=False
        )
        return train_loader, test_loader

    def get_cumulative_test_loader(self, up_to_task: int) -> DataLoader:
        """Return a test loader over all classes from tasks 0..up_to_task."""
        all_x, all_y = [], []
        for t in range(up_to_task + 1):
            te_indices = self._base.test_splits[t]
            te_x = torch.stack([self._base._test_ds[i][0] for i in te_indices])
            te_y = torch.tensor([self._base._test_ds[i][1] for i in te_indices])
            all_x.append(te_x)
            all_y.append(te_y)
        return DataLoader(
            TensorDataset(torch.cat(all_x), torch.cat(all_y)),
            batch_size=self._base.batch_size,
            shuffle=False,
        )


class _TinyImageNetDataset:
    """Minimal TinyImageNet dataset loader.

    Downloads from http://cs231n.stanford.edu/tiny-imagenet-200.zip if needed.
    Provides indexing: dataset[i] -> (image_tensor, label_int).
    """

    def __init__(self, root: str, train: bool, transform, download: bool = True):
        self.root = os.path.join(root, "tiny-imagenet-200")
        self.transform = transform
        self.train = train
        self._images: list[str] = []
        self._labels: list[int] = []

        if download and not os.path.isdir(self.root):
            self._download(root)

        self._load_class_map()
        if train:
            self._load_train()
        else:
            self._load_val()

    def _download(self, root: str) -> None:
        import urllib.request
        import zipfile
        os.makedirs(root, exist_ok=True)
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = os.path.join(root, "tiny-imagenet-200.zip")
        if not os.path.exists(zip_path):
            print(f"Downloading TinyImageNet to {zip_path}...")
            urllib.request.urlretrieve(url, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)

    def _load_class_map(self) -> None:
        wnids_file = os.path.join(self.root, "wnids.txt")
        with open(wnids_file) as f:
            self._wnids = [line.strip() for line in f if line.strip()]
        self._wnid_to_idx = {w: i for i, w in enumerate(sorted(self._wnids))}

    def _load_train(self) -> None:
        train_dir = os.path.join(self.root, "train")
        for wnid in self._wnids:
            img_dir = os.path.join(train_dir, wnid, "images")
            label = self._wnid_to_idx[wnid]
            for fname in sorted(os.listdir(img_dir)):
                if fname.endswith(".JPEG"):
                    self._images.append(os.path.join(img_dir, fname))
                    self._labels.append(label)

    def _load_val(self) -> None:
        val_dir = os.path.join(self.root, "val")
        ann_file = os.path.join(val_dir, "val_annotations.txt")
        fname_to_wnid: dict[str, str] = {}
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                fname_to_wnid[parts[0]] = parts[1]
        img_dir = os.path.join(val_dir, "images")
        for fname in sorted(os.listdir(img_dir)):
            if fname.endswith(".JPEG") and fname in fname_to_wnid:
                wnid = fname_to_wnid[fname]
                self._images.append(os.path.join(img_dir, fname))
                self._labels.append(self._wnid_to_idx[wnid])

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        img = Image.open(self._images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self._labels[idx]


class SplitTinyImageNet(_SplitBenchmark):
    """Split Tiny-ImageNet: 200 classes, 10 tasks x 20 classes."""

    def __init__(
        self,
        data_root: str = "./data",
        batch_size: int = 64,
        classes_per_task: int = 20,
    ):
        self.batch_size = batch_size
        self.classes_per_task = classes_per_task

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])

        train_ds = _TinyImageNetDataset(root=data_root, train=True, transform=transform)
        test_ds = _TinyImageNetDataset(root=data_root, train=False, transform=transform)

        self.train_splits = _class_split(train_ds, classes_per_task)
        self.test_splits = _class_split(test_ds, classes_per_task)

        self._train_ds = train_ds
        self._test_ds = test_ds
        self._loader_cache: dict[int, tuple[DataLoader, DataLoader]] = {}
