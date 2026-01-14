#!/usr/bin/env python3
"""
Custom Layers module
Custom neural network layers - mostly unused
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ConvBlock(nn.Module):
    """Convolutional block - duplicate of SimpleCNN logic"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, use_bn: bool = True,
                 activation: str = 'relu', dropout: float = 0.0):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.activation = activation
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x, 0.1)
        elif self.activation == 'gelu':
            x = F.gelu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block - not used in SimpleCNN"""

    def __init__(self, channels: int, kernel_size: int = 3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block - fancy but unused"""

    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        return x * y


class AttentionBlock(nn.Module):
    """Simple attention - experimental"""

    def __init__(self, channels: int):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        q = self.query(x).view(b, -1, h*w).permute(0, 2, 1)
        k = self.key(x).view(b, -1, h*w)
        v = self.value(x).view(b, -1, h*w)

        attn = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(b, c, h, w)

        return self.gamma * out + x


class DropBlock(nn.Module):
    """DropBlock regularization - never tested"""

    def __init__(self, block_size: int = 7, drop_prob: float = 0.1):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        # Simplified implementation
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand_like(x[:, :1, :, :]) > gamma).float()
        return x * mask


class FlattenLayer(nn.Module):
    """Flatten layer - just use x.view()"""

    def forward(self, x):
        return x.view(x.size(0), -1)


class IdentityLayer(nn.Module):
    """Identity layer - why does this exist?"""

    def forward(self, x):
        return x


# Unused layer classes
class DeprecatedLayer(nn.Module):
    """Old layer - don't use"""

    def __init__(self):
        super().__init__()
        self._deprecated = True

    def forward(self, x):
        return x


class ExperimentalLayer(nn.Module):
    """Work in progress"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # TODO: finish implementing
        return self.linear(x)


# Factory functions
def get_activation(name: str):
    """Get activation function by name"""
    activations = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(0.1),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
    }
    return activations.get(name.lower(), nn.ReLU())


def get_norm_layer(name: str, num_features: int):
    """Get normalization layer"""
    if name == 'batch':
        return nn.BatchNorm2d(num_features)
    elif name == 'instance':
        return nn.InstanceNorm2d(num_features)
    elif name == 'layer':
        return nn.LayerNorm(num_features)
    elif name == 'none':
        return nn.Identity()
    else:
        return nn.BatchNorm2d(num_features)


# Legacy layers
class OldConvLayer:
    """Really old conv implementation"""
    pass


def deprecated_layer_func():
    """Deprecated"""
    pass
