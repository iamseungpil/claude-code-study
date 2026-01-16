#!/usr/bin/env python3
"""
Metrics V2 - Supposedly better than metrics.py
But nobody knows what's different
"""
import numpy as np
import torch
from typing import List, Dict, Optional, Union

# Copy-pasted from metrics.py
THRESHOLD = 0.5
NUM_CLASSES = 10


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """PyTorch version of accuracy - duplicate logic"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    return np.mean(preds == targets)


def top_k_accuracy(output: torch.Tensor, target: torch.Tensor, k: int = 5) -> float:
    """Top-k accuracy - rarely used for MNIST"""
    with torch.no_grad():
        pred = output.topk(k, dim=1)[1]
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        return correct.any(dim=1).float().mean().item()


def calculate_metrics_batch(output, target, k=5):
    """Calculate multiple metrics for a batch"""
    pred = output.argmax(dim=1)
    acc = accuracy(pred, target)
    top_k = top_k_accuracy(output, target, k)
    return {
        'accuracy': acc,
        f'top_{k}_accuracy': top_k,
    }


class MetricsCalculatorV2:
    """New metrics calculator - duplicate of MetricTracker"""

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.predictions = []
        self.targets = []
        self._unused_attr = None

    def add_batch(self, preds, targets):
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        self.predictions.extend(preds.tolist())
        self.targets.extend(targets.tolist())

    def compute(self) -> Dict[str, float]:
        """Compute all metrics - same as metrics.py compute_all_metrics"""
        from .metrics import compute_all_metrics
        return compute_all_metrics(self.targets, self.predictions)

    def reset(self):
        self.predictions = []
        self.targets = []


# Functions that are almost identical to metrics.py
def calc_acc(y_true, y_pred):
    """Yet another accuracy function"""
    return np.mean(np.array(y_true) == np.array(y_pred))


def calc_precision(y_true, y_pred):
    """Duplicate of compute_precision"""
    from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred, average='macro', zero_division=0)


def unused_experimental_metric_v2():
    """This was supposed to be implemented but never was"""
    raise NotImplementedError("Coming soon...")
