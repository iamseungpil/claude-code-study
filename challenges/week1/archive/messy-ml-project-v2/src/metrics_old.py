#!/usr/bin/env python3
"""
OLD METRICS - DO NOT USE
Kept for backwards compatibility (nobody knows with what)
"""
import numpy as np

# WARNING: These functions may not work correctly

def calculate_accuracy_legacy(predictions, labels):
    """Legacy accuracy calculation - probably broken"""
    total = len(predictions)
    if total == 0:
        return 0.0
    correct = 0
    for i in range(total):
        if predictions[i] == labels[i]:
            correct += 1
    return correct / total


def calculate_precision_legacy(predictions, labels, class_id):
    """Old precision - single class only"""
    tp = 0
    fp = 0
    for p, l in zip(predictions, labels):
        if p == class_id:
            if l == class_id:
                tp += 1
            else:
                fp += 1
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def calculate_recall_legacy(predictions, labels, class_id):
    """Old recall - single class only"""
    tp = 0
    fn = 0
    for p, l in zip(predictions, labels):
        if l == class_id:
            if p == class_id:
                tp += 1
            else:
                fn += 1
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


class OldMetricCalculator:
    """Really old metric calculator - from version 1.0"""

    def __init__(self):
        self.results = {}
        self._deprecated = True

    def calculate(self, preds, labels):
        """Calculate old-style metrics"""
        self.results['acc'] = calculate_accuracy_legacy(preds, labels)
        return self.results

    def get_result(self, key):
        return self.results.get(key, None)


# Dead code below - never executed
if False:
    def never_runs():
        print("This will never print")
        return None


def _internal_helper():
    """Was used internally, now unused"""
    pass


# More unused functions
def deprecated_metric_1():
    pass

def deprecated_metric_2():
    pass

def deprecated_metric_3():
    pass
