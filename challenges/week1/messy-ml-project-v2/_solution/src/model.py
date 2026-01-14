"""
Clean CNN Model for MNIST Classification
"""
import torch
import torch.nn as nn
from typing import Tuple


class MNISTClassifier(nn.Module):
    """Simple CNN for MNIST digit classification."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(num_classes: int = 10, dropout_rate: float = 0.5) -> MNISTClassifier:
    """Factory function to create model."""
    return MNISTClassifier(num_classes=num_classes, dropout_rate=dropout_rate)
