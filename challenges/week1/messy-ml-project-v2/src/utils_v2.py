"""Utility functions v2 - with some improvements"""
import os
import logging
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def setup_logging(log_file=None):
    """Configure logging for training."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_data(batch_size=64, num_workers=4):
    """Load MNIST dataset with multiprocessing."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        'data/raw', train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        'data/raw', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader


def save_model(model, path, metadata=None):
    """Save model checkpoint with optional metadata."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    checkpoint = {'state_dict': model.state_dict()}
    if metadata:
        checkpoint['metadata'] = metadata
    torch.save(checkpoint, path)
    logging.info(f"Model saved to {path}")


def load_model(model, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    logging.info(f"Model loaded from {path}")
    return model


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)
