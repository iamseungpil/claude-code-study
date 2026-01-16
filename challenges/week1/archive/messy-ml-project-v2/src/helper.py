"""Helper functions - mostly duplicates of utils.py"""
import torch
import numpy as np
import random


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the device to use for training."""
    # Same as utils_old.get_device()
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_params(model):
    """Count model parameters - same as utils.count_parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    """Print model summary."""
    print(model)
    print(f"Total parameters: {count_params(model):,}")


def normalize_data(data, mean=0.1307, std=0.3081):
    """Normalize data - duplicate of transform in load_data"""
    return (data - mean) / std


def denormalize_data(data, mean=0.1307, std=0.3081):
    """Denormalize data."""
    return data * std + mean


def compute_accuracy(outputs, targets):
    """Compute accuracy - same as in train_final_REAL.py"""
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()


def save_to_json(data, path):
    """Save data to JSON file."""
    import json
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_from_json(path):
    """Load data from JSON file."""
    import json
    with open(path, 'r') as f:
        return json.load(f)
