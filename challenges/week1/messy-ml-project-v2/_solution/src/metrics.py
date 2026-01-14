"""
Evaluation metrics
"""
import numpy as np
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }


def get_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """Get confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def per_class_accuracy(y_true: List[int], y_pred: List[int], num_classes: int = 10) -> Dict[int, float]:
    """Compute per-class accuracy."""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    per_class = {}
    for i in range(num_classes):
        total = cm[i].sum()
        per_class[i] = cm[i, i] / total if total > 0 else 0.0
    return per_class
