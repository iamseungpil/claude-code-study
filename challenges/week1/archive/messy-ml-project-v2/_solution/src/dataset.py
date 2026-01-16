"""
Dataset utilities for MNIST
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Get train and test transforms."""
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return train_transform, test_transform


def create_dataloaders(
    data_dir: str = './data',
    batch_size: int = 64,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders."""
    train_transform, test_transform = get_transforms()

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader
