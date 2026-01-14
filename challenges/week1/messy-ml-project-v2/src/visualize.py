#!/usr/bin/env python3
"""
Visualization utilities
Duplicate of plot functions in train_final_REAL.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Optional
import os

# Constants - also defined elsewhere
FIGURE_DPI = 100
DEFAULT_FIGSIZE = (10, 6)
STYLE = 'seaborn-v0_8-whitegrid'


def plot_loss_curve(losses: List[float], save_path: str = None, title: str = 'Loss'):
    """Plot single loss curve - simple version"""
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI)
    plt.close()


def plot_loss_curves(train_losses: List[float], val_losses: List[float],
                     save_path: str = None):
    """Plot train and val loss - duplicate of train_final_REAL.py"""
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI)
    plt.close()


def plot_accuracy_curve(accuracies: List[float], save_path: str = None):
    """Plot accuracy curve"""
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI)
    plt.close()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """DUPLICATE of train_final_REAL.py plot_training_curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train')
    ax1.plot(val_losses, label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Curves')

    ax2.plot(train_accs, label='Train')
    ax2.plot(val_accs, label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy Curves')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm, class_names=None, save_path=None,
                          normalize=False, cmap='Blues'):
    """Plot confusion matrix with seaborn"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()


def plot_samples(images, labels, predictions=None, num_samples=16, save_path=None):
    """Plot sample images with labels"""
    n = min(num_samples, len(images))
    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        title = f'Label: {labels[i]}'
        if predictions is not None:
            title += f'\nPred: {predictions[i]}'
        axes[i].set_title(title, fontsize=8)
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI)
    plt.close()


def plot_lr_schedule(lr_values: List[float], save_path: str = None):
    """Plot learning rate schedule"""
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(lr_values)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI)
    plt.close()


# Unused plotting functions
def plot_experimental():
    """Experimental plot - never finished"""
    pass


def plot_old_style():
    """Old plotting style - deprecated"""
    pass


class Visualizer:
    """Visualizer class - mostly unused"""

    def __init__(self, output_dir: str = 'outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot(self, data, name):
        """Generic plot"""
        save_path = os.path.join(self.output_dir, f'{name}.png')
        plt.figure()
        plt.plot(data)
        plt.savefig(save_path)
        plt.close()
        return save_path
