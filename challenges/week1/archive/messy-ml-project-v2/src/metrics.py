#!/usr/bin/env python3
"""
Metrics calculation module
Version: 3.0 (or is it 2.5?)
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import torch

# Duplicate constants (also in train_final_REAL.py)
THRESHOLD = 0.5
NUM_CLASSES = 10


def compute_accuracy(y_true, y_pred):
    """Compute accuracy - same as in train_final_REAL.py"""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def compute_accuracy_v2(y_true, y_pred):
    """Compute accuracy using numpy - duplicate"""
    return np.mean(np.array(y_true) == np.array(y_pred))


def compute_accuracy_sklearn(y_true, y_pred):
    """Compute accuracy using sklearn - another duplicate"""
    return accuracy_score(y_true, y_pred)


def compute_precision(y_true, y_pred, average='macro'):
    """Compute precision"""
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def compute_recall(y_true, y_pred, average='macro'):
    """Compute recall"""
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def compute_f1(y_true, y_pred, average='macro'):
    """Compute F1 score"""
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_all_metrics(y_true, y_pred, verbose=False):
    """Compute all metrics at once - but we usually only use accuracy"""
    metrics = {
        'accuracy': compute_accuracy(y_true, y_pred),
        'accuracy_v2': compute_accuracy_v2(y_true, y_pred),
        'precision': compute_precision(y_true, y_pred),
        'recall': compute_recall(y_true, y_pred),
        'f1': compute_f1(y_true, y_pred),
    }
    if verbose:
        print(classification_report(y_true, y_pred))
    return metrics


def get_confusion_matrix(y_true, y_pred):
    """Get confusion matrix"""
    return confusion_matrix(y_true, y_pred)


def compute_per_class_accuracy(y_true, y_pred, num_classes=NUM_CLASSES):
    """Compute per-class accuracy - rarely used"""
    cm = get_confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)
    return per_class


# Unused function
def experimental_metric(y_true, y_pred, alpha=0.5, beta=0.5):
    """Some experimental metric - never finished implementing"""
    # TODO: implement this
    pass


# Legacy function
def compute_metrics_old(predictions, labels):
    """Old metric computation - DO NOT USE"""
    pass


class MetricTracker:
    """Track metrics during training - mostly unused features"""

    def __init__(self, metrics_to_track=None):
        self.metrics = {}
        self.history = []
        self._internal_buffer = []

    def update(self, metric_name, value):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def get_average(self, metric_name):
        if metric_name in self.metrics:
            return np.mean(self.metrics[metric_name])
        return 0.0

    def reset(self):
        self.metrics = {}

    def unused_method(self):
        """Never called"""
        pass
