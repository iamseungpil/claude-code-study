"""Utility functions for MNIST training"""
import os
import logging
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def setup_logging():
    """Configure logging for training."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_data(batch_size=64):
    """Load MNIST dataset."""
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def save_model(model, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logging.info(f"Model saved to {path}")


def load_model(model, path):
    """Load model checkpoint."""
    model.load_state_dict(torch.load(path))
    logging.info(f"Model loaded from {path}")
    return model
