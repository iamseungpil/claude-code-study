#!/usr/bin/env python3
"""
Old Sampler implementations - DEPRECATED
DO NOT USE - kept for reference only
"""
import random
import numpy as np
from typing import List, Iterator

# WARNING: These samplers are outdated and may not work


class LegacySampler:
    """Very old sampler from v1.0"""

    def __init__(self, data_source, batch_size=32):
        self.data_source = data_source
        self.batch_size = batch_size
        self._deprecated = True

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.data_source)


class OldRandomSampler:
    """Old random sampler - use torch.utils.data.RandomSampler instead"""

    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source
        self.num_samples = num_samples or len(data_source)

    def __iter__(self):
        n = len(self.data_source)
        return iter(random.sample(range(n), min(self.num_samples, n)))

    def __len__(self):
        return self.num_samples


class OldSequentialSampler:
    """Old sequential sampler - redundant"""

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class VeryOldSampler:
    """Even older sampler - from v0.5"""

    def __init__(self, n):
        self.n = n
        self.indices = list(range(n))

    def shuffle(self):
        random.shuffle(self.indices)
        return self.indices

    def get_batch(self, batch_size):
        self.shuffle()
        return self.indices[:batch_size]


class BrokenSampler:
    """This sampler has bugs - don't use"""

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        # BUG: this doesn't work properly
        return iter(self.data)


def old_sample_batch(data, batch_size):
    """Old batch sampling function - deprecated"""
    indices = random.sample(range(len(data)), batch_size)
    return [data[i] for i in indices]


def old_split_data(data, ratio=0.8):
    """Old train/val split - deprecated"""
    n = len(data)
    split_idx = int(n * ratio)
    indices = list(range(n))
    random.shuffle(indices)
    return indices[:split_idx], indices[split_idx:]


def deprecated_sampler_factory(name):
    """Old sampler factory - deprecated"""
    samplers = {
        'random': OldRandomSampler,
        'sequential': OldSequentialSampler,
    }
    return samplers.get(name)


# More dead code
class NeverUsedSampler:
    """This class was never used"""
    pass


def _internal_sample_helper():
    """Internal helper - unused"""
    pass


if False:
    # Dead code
    class DeadSampler:
        pass
