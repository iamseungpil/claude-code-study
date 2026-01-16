"""Data loader utilities - BACKUP
Created before the refactoring
"""
from torch.utils.data import DataLoader
from torchvision import datasets
from preprocess import get_train_transforms, get_val_transforms


def create_loaders(data_dir='data/raw', batch_size=64, num_workers=0):
    """Create train and validation data loaders."""
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True,
        transform=get_train_transforms()
    )
    val_dataset = datasets.MNIST(
        data_dir, train=False, download=True,
        transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
