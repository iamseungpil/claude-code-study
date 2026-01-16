#!/usr/bin/env python3
"""
Data Samplers
Custom samplers for data loading
"""
import torch
from torch.utils.data import Sampler, Dataset
from typing import Iterator, List, Optional
import numpy as np
import random


class BalancedSampler(Sampler):
    """Balanced class sampler - ensures equal class representation"""

    def __init__(self, dataset: Dataset, num_samples: int = None):
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)

        # Build class indices
        self.class_indices = {}
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        self.num_classes = len(self.class_indices)

    def __iter__(self) -> Iterator[int]:
        indices = []
        samples_per_class = self.num_samples // self.num_classes

        for class_idx in self.class_indices.values():
            sampled = np.random.choice(class_idx, samples_per_class, replace=True)
            indices.extend(sampled.tolist())

        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


class WeightedRandomSampler(Sampler):
    """Weighted random sampler - duplicate of torch.utils.data.WeightedRandomSampler"""

    def __init__(self, weights: List[float], num_samples: int, replacement: bool = True):
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(self.weights, self.num_samples, self.replacement)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


class StratifiedSampler(Sampler):
    """Stratified sampler for train/val split"""

    def __init__(self, labels: List[int], batch_size: int):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.num_classes = len(np.unique(self.labels))

        # Group indices by label
        self.class_indices = {
            i: np.where(self.labels == i)[0].tolist()
            for i in range(self.num_classes)
        }

    def __iter__(self) -> Iterator[int]:
        batches = []
        indices_per_class = {k: v.copy() for k, v in self.class_indices.items()}

        for class_indices in indices_per_class.values():
            random.shuffle(class_indices)

        while any(len(v) > 0 for v in indices_per_class.values()):
            batch = []
            for class_idx, indices in indices_per_class.items():
                if indices:
                    batch.append(indices.pop())
            if batch:
                batches.extend(batch)

        return iter(batches)

    def __len__(self) -> int:
        return len(self.labels)


class SubsetSampler(Sampler):
    """Sample a subset of indices - simpler version"""

    def __init__(self, indices: List[int], shuffle: bool = False):
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            return iter(np.random.permutation(self.indices))
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class InfiniteSampler(Sampler):
    """Infinite sampler - keeps yielding indices"""

    def __init__(self, dataset: Dataset, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        order = list(range(len(self.dataset)))
        while True:
            if self.shuffle:
                np.random.shuffle(order)
            for idx in order:
                yield idx


class BatchBalancedSampler(Sampler):
    """Ensure each batch has balanced classes"""

    def __init__(self, labels: List[int], batch_size: int):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.classes = np.unique(self.labels)
        self.class_indices = {
            c: np.where(self.labels == c)[0].tolist()
            for c in self.classes
        }

    def __iter__(self) -> Iterator[int]:
        indices = []
        samples_per_class = self.batch_size // len(self.classes)

        # Shuffle within classes
        for class_idx in self.class_indices.values():
            random.shuffle(class_idx)

        # Create balanced batches
        while all(len(v) >= samples_per_class for v in self.class_indices.values()):
            batch = []
            for class_indices in self.class_indices.values():
                batch.extend(class_indices[:samples_per_class])
                del class_indices[:samples_per_class]
            random.shuffle(batch)
            indices.extend(batch)

        return iter(indices)

    def __len__(self) -> int:
        return len(self.labels)


# Legacy samplers
class OldSampler:
    """Old sampler - deprecated"""

    def __init__(self, data):
        self.data = data

    def sample(self, n):
        return random.sample(range(len(self.data)), n)


def deprecated_sampler():
    """Deprecated"""
    pass
