#!/usr/bin/env python3
"""
Result Visualization - Another visualization module
Why do we have two? Nobody knows
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# Import from visualize.py - circular dependency risk
from .visualize import plot_confusion_matrix, FIGURE_DPI


def load_training_log(log_path: str):
    """Load training log from JSON file"""
    with open(log_path, 'r') as f:
        return json.load(f)


def visualize_training_results(log_path: str, output_dir: str = 'outputs/figures'):
    """Visualize training results from log file"""
    os.makedirs(output_dir, exist_ok=True)

    log = load_training_log(log_path)

    # Plot losses
    if 'train_losses' in log and 'val_losses' in log:
        from .visualize import plot_loss_curves
        plot_loss_curves(
            log['train_losses'],
            log['val_losses'],
            os.path.join(output_dir, 'losses.png')
        )

    return True


def create_report_figures(results_dict: dict, output_dir: str):
    """Create all figures for report - duplicate logic"""
    os.makedirs(output_dir, exist_ok=True)

    # This function duplicates visualize.py functionality
    if 'losses' in results_dict:
        plt.figure(figsize=(10, 6))
        plt.plot(results_dict['losses'])
        plt.savefig(os.path.join(output_dir, 'losses_report.png'))
        plt.close()

    if 'accuracies' in results_dict:
        plt.figure(figsize=(10, 6))
        plt.plot(results_dict['accuracies'])
        plt.savefig(os.path.join(output_dir, 'accuracies_report.png'))
        plt.close()


def compare_experiments(exp_dirs: list, metric: str = 'accuracy'):
    """Compare multiple experiments - mostly unused"""
    results = []
    for exp_dir in exp_dirs:
        log_path = os.path.join(exp_dir, 'training_log.json')
        if os.path.exists(log_path):
            with open(log_path) as f:
                log = json.load(f)
                if metric in log:
                    results.append({
                        'name': os.path.basename(exp_dir),
                        'values': log[metric]
                    })
    return results


def plot_comparison(results: list, metric: str, save_path: str = None):
    """Plot comparison of multiple experiments"""
    plt.figure(figsize=(12, 6))
    for r in results:
        plt.plot(r['values'], label=r['name'])
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.title(f'{metric.capitalize()} Comparison')
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI)
    plt.close()


# More duplicate/unused functions
def plot_results_old():
    """Old result plotting - deprecated"""
    pass


def generate_latex_table(results: dict):
    """Generate LaTeX table - never used"""
    pass


def export_to_csv(results: dict, path: str):
    """Export results to CSV - rarely used"""
    pass


class ResultVisualizer:
    """Yet another visualizer class"""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self._cache = {}

    def plot_all(self):
        """Plot everything"""
        # TODO: implement
        pass

    def unused_method(self):
        pass
