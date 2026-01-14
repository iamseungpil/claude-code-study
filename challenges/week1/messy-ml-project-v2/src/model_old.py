"""Old model - simple linear classifier
DEPRECATED: Use model.py instead
"""
import torch
import torch.nn as nn


class OldModel(nn.Module):
    """Simple linear model - too simple, doesn't work well"""

    def __init__(self):
        super(OldModel, self).__init__()
        self.layer = nn.Linear(784, 10)

    def forward(self, x):
        return self.layer(x.view(-1, 784))


class EvenOlderModel(nn.Module):
    """The first model we tried - terrible performance"""

    def __init__(self):
        super(EvenOlderModel, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.flatten(1)
        return self.fc(x)
