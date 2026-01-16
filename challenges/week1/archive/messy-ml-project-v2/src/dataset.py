#!/usr/bin/env python3
"""
Dataset utilities
Custom datasets and data handling
Overlaps with load_data in train_final_REAL.py
"""
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms
from typing import Tuple, List, Optional, Callable
import numpy as np
import os

# Constants - duplicated
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
DEFAULT_BATCH_SIZE = 64


class MNISTDataset(Dataset):
    """Custom MNIST dataset wrapper - mostly redundant"""

    def __init__(self, root: str = 'data/raw', train: bool = True,
                 transform: Callable = None, download: bool = True):
        self.dataset = datasets.MNIST(root, train=train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class CachedDataset(Dataset):
    """Dataset that caches transformed samples - memory intensive"""

    def __init__(self, base_dataset: Dataset, transform: Callable = None):
        self.base_dataset = base_dataset
        self.transform = transform
        self.cache = {}

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        if idx not in self.cache:
            img, label = self.base_dataset[idx]
            if self.transform:
                img = self.transform(img)
            self.cache[idx] = (img, label)
        return self.cache[idx]

    def clear_cache(self):
        self.cache = {}


class SubsetDataset(Dataset):
    """Subset of a dataset - just use torch.utils.data.Subset instead"""

    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def get_dataloaders(data_dir: str = 'data/raw', batch_size: int = DEFAULT_BATCH_SIZE,
                    num_workers: int = 0, val_split: float = 0.1,
                    augment: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get train, val, test dataloaders - similar to load_data"""

    # Transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])

    # Datasets
    full_train = datasets.MNIST(data_dir, train=True, download=True,
                                  transform=train_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                   transform=test_transform)

    # Split train/val
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_train_val_loaders(data_dir: str = 'data/raw', batch_size: int = DEFAULT_BATCH_SIZE):
    """Get just train and val loaders - duplicate of load_data"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                    transform=transform)
    val_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def create_subset(dataset: Dataset, fraction: float = 0.1,
                  seed: int = 42) -> Dataset:
    """Create a smaller subset of dataset for debugging"""
    np.random.seed(seed)
    n = len(dataset)
    indices = np.random.choice(n, int(n * fraction), replace=False)
    return Subset(dataset, indices)


class DatasetStats:
    """Compute dataset statistics - rarely used"""

    @staticmethod
    def compute_mean_std(dataset: Dataset) -> Tuple[float, float]:
        """Compute mean and std of dataset"""
        loader = DataLoader(dataset, batch_size=1000, shuffle=False)
        mean = 0.
        std = 0.
        total = 0

        for data, _ in loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, -1)
            mean += data.mean(1).sum()
            std += data.std(1).sum()
            total += batch_samples

        return mean / total, std / total


# Legacy functions
def load_data_old(batch_size):
    """Old data loading function - deprecated"""
    pass


def deprecated_dataset_func():
    """Deprecated"""
    pass


class OldDataset:
    """Old dataset class - don't use"""

    def __init__(self):
        self._deprecated = True
