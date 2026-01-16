#!/usr/bin/env python3
"""
Optimizer utilities
Configuration and helpers for optimizers
"""
import torch
import torch.optim as optim
from typing import Dict, Any, Optional, List
import math

# Constants - duplicate of train_final_REAL.py
DEFAULT_LR = 0.001
LR_DECAY_FACTOR = 0.1
WEIGHT_DECAY = 1e-4


def get_optimizer(model, name: str = 'adam', lr: float = DEFAULT_LR,
                  weight_decay: float = WEIGHT_DECAY, momentum: float = 0.9,
                  betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                  nesterov: bool = False, **kwargs) -> optim.Optimizer:
    """Get optimizer by name - too many unused parameters"""

    params = model.parameters()

    if name.lower() == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay,
                          betas=betas, eps=eps)
    elif name.lower() == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                           betas=betas, eps=eps)
    elif name.lower() == 'sgd':
        return optim.SGD(params, lr=lr, momentum=momentum,
                         weight_decay=weight_decay, nesterov=nesterov)
    elif name.lower() == 'rmsprop':
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def get_optimizer_simple(model, lr: float = DEFAULT_LR) -> optim.Optimizer:
    """Simple optimizer getter - duplicate but simpler"""
    return optim.Adam(model.parameters(), lr=lr)


def configure_optimizer(model, config: Dict[str, Any]) -> optim.Optimizer:
    """Configure optimizer from dict - duplicate of get_optimizer"""
    name = config.get('name', 'adam')
    lr = config.get('lr', DEFAULT_LR)
    weight_decay = config.get('weight_decay', 0)

    return get_optimizer(model, name=name, lr=lr, weight_decay=weight_decay)


class OptimizerBuilder:
    """Builder pattern for optimizer - overcomplicated"""

    def __init__(self, model):
        self.model = model
        self.config = {
            'name': 'adam',
            'lr': DEFAULT_LR,
            'weight_decay': 0,
        }

    def with_lr(self, lr: float):
        self.config['lr'] = lr
        return self

    def with_weight_decay(self, wd: float):
        self.config['weight_decay'] = wd
        return self

    def with_name(self, name: str):
        self.config['name'] = name
        return self

    def build(self) -> optim.Optimizer:
        return configure_optimizer(self.model, self.config)


def get_param_groups(model, lr: float, lr_multiplier: Dict[str, float] = None):
    """Get parameter groups with different learning rates - rarely used"""
    if lr_multiplier is None:
        return model.parameters()

    param_groups = []
    for name, param in model.named_parameters():
        group_lr = lr
        for pattern, mult in lr_multiplier.items():
            if pattern in name:
                group_lr = lr * mult
                break
        param_groups.append({'params': [param], 'lr': group_lr})

    return param_groups


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_epochs=[30, 60, 90],
                         decay_factor=LR_DECAY_FACTOR):
    """Adjust learning rate by epoch - old-school method"""
    lr = initial_lr
    for milestone in decay_epochs:
        if epoch >= milestone:
            lr *= decay_factor

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def warmup_lr(optimizer, epoch, warmup_epochs, initial_lr):
    """Warmup learning rate - duplicate of scheduler functionality"""
    if epoch < warmup_epochs:
        lr = initial_lr * (epoch + 1) / warmup_epochs
    else:
        lr = initial_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


# Old/unused functions
def get_optimizer_old(model, lr):
    """Old optimizer getter - DO NOT USE"""
    pass


def deprecated_optimizer_config():
    """Deprecated"""
    pass


class LegacyOptimizer:
    """Legacy optimizer wrapper - unused"""

    def __init__(self, model, lr):
        self.optimizer = optim.SGD(model.parameters(), lr=lr)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
