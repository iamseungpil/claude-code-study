#!/usr/bin/env python3
"""
Analysis module
Analyze training results, model performance, etc.
"""
import numpy as np
import torch
import json
import os
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Duplicate imports from metrics
from .metrics import compute_all_metrics, get_confusion_matrix


def analyze_training_run(log_path: str) -> Dict[str, Any]:
    """Analyze a training run from log file"""
    with open(log_path, 'r') as f:
        log = json.load(f)

    analysis = {}

    if 'train_losses' in log:
        losses = log['train_losses']
        analysis['loss_analysis'] = {
            'initial': losses[0],
            'final': losses[-1],
            'min': min(losses),
            'max': max(losses),
            'reduction': losses[0] - losses[-1],
        }

    if 'val_accuracies' in log:
        accs = log['val_accuracies']
        analysis['accuracy_analysis'] = {
            'initial': accs[0],
            'final': accs[-1],
            'best': max(accs),
            'best_epoch': accs.index(max(accs)),
        }

    return analysis


def analyze_model_predictions(model, dataloader, device='cpu'):
    """Analyze model predictions - duplicate logic from validate()"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Get metrics
    metrics = compute_all_metrics(all_targets, all_preds)
    cm = get_confusion_matrix(all_targets, all_preds)

    # Analyze confusion matrix
    analysis = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'per_class_accuracy': (cm.diagonal() / cm.sum(axis=1)).tolist(),
        'most_confused_pairs': find_confused_pairs(cm),
    }

    return analysis


def find_confused_pairs(cm: np.ndarray, top_k: int = 5) -> List[Dict]:
    """Find most confused class pairs"""
    confused = []
    n = len(cm)

    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                confused.append({
                    'true': i,
                    'predicted': j,
                    'count': int(cm[i, j]),
                })

    confused.sort(key=lambda x: x['count'], reverse=True)
    return confused[:top_k]


def analyze_loss_curve(losses: List[float]) -> Dict[str, Any]:
    """Analyze loss curve for patterns - duplicate of analyze_training_run"""
    return {
        'converged': losses[-1] < losses[0] * 0.1,
        'plateau_detected': detect_plateau(losses),
        'oscillating': detect_oscillation(losses),
    }


def detect_plateau(values: List[float], threshold: float = 0.01,
                   window: int = 5) -> bool:
    """Detect if values have plateaued"""
    if len(values) < window:
        return False
    recent = values[-window:]
    return (max(recent) - min(recent)) < threshold


def detect_oscillation(values: List[float], threshold: float = 0.1) -> bool:
    """Detect oscillation in values"""
    if len(values) < 3:
        return False
    diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
    sign_changes = sum(1 for i in range(len(diffs)-1)
                       if diffs[i] * diffs[i+1] < 0)
    return sign_changes > len(diffs) * 0.5


def compute_convergence_rate(losses: List[float]) -> float:
    """Compute convergence rate - experimental"""
    if len(losses) < 2:
        return 0.0
    rates = [(losses[i] - losses[i+1]) / losses[i]
             for i in range(len(losses)-1) if losses[i] != 0]
    return np.mean(rates) if rates else 0.0


class TrainingAnalyzer:
    """Class for analyzing training - mostly unused"""

    def __init__(self):
        self.cache = {}

    def analyze(self, log_path: str) -> Dict[str, Any]:
        if log_path in self.cache:
            return self.cache[log_path]
        result = analyze_training_run(log_path)
        self.cache[log_path] = result
        return result

    def clear_cache(self):
        self.cache = {}


# Unused functions
def old_analysis():
    """Old analysis function"""
    pass


def experimental_analysis():
    """Work in progress"""
    pass


def _helper():
    """Internal helper"""
    pass
