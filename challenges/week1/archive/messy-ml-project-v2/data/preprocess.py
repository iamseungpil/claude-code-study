"""Data preprocessing - current version"""
import torch
from torchvision import transforms

# Magic numbers - same as train_final_REAL.py
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_train_transforms():
    """Get training transforms."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])


def get_val_transforms():
    """Get validation transforms."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])


def get_augmented_transforms():
    """Get augmented training transforms."""
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])


def normalize(data):
    """Normalize data - duplicate of helper.normalize_data"""
    return (data - MNIST_MEAN) / MNIST_STD


def denormalize(data):
    """Denormalize data - duplicate of helper.denormalize_data"""
    return data * MNIST_STD + MNIST_MEAN
