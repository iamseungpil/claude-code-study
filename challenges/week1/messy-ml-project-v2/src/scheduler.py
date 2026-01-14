#!/usr/bin/env python3
"""
Learning Rate Scheduler utilities
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, OneCycleLR, LambdaLR
)
from typing import Optional, List, Dict, Any
import math

# Constants - duplicated
DEFAULT_STEP_SIZE = 10
DEFAULT_GAMMA = 0.1
DEFAULT_MILESTONES = [30, 60, 90]


def get_scheduler(optimizer, name: str = 'step', **kwargs):
    """Get scheduler by name - too many options"""

    if name.lower() == 'step':
        return StepLR(optimizer,
                      step_size=kwargs.get('step_size', DEFAULT_STEP_SIZE),
                      gamma=kwargs.get('gamma', DEFAULT_GAMMA))

    elif name.lower() == 'multistep':
        return MultiStepLR(optimizer,
                           milestones=kwargs.get('milestones', DEFAULT_MILESTONES),
                           gamma=kwargs.get('gamma', DEFAULT_GAMMA))

    elif name.lower() == 'exponential':
        return ExponentialLR(optimizer,
                             gamma=kwargs.get('gamma', 0.95))

    elif name.lower() == 'cosine':
        return CosineAnnealingLR(optimizer,
                                 T_max=kwargs.get('T_max', 100),
                                 eta_min=kwargs.get('eta_min', 1e-6))

    elif name.lower() == 'plateau':
        return ReduceLROnPlateau(optimizer,
                                 mode=kwargs.get('mode', 'min'),
                                 factor=kwargs.get('factor', 0.1),
                                 patience=kwargs.get('patience', 10))

    elif name.lower() == 'onecycle':
        return OneCycleLR(optimizer,
                          max_lr=kwargs.get('max_lr', 0.01),
                          total_steps=kwargs.get('total_steps', 1000))

    elif name.lower() == 'none' or name.lower() is None:
        return None

    else:
        raise ValueError(f"Unknown scheduler: {name}")


def get_scheduler_simple(optimizer, epochs: int = 10):
    """Simple scheduler getter - duplicate"""
    return StepLR(optimizer, step_size=epochs // 3, gamma=0.1)


def create_warmup_scheduler(optimizer, warmup_steps: int,
                            total_steps: int, min_lr: float = 1e-7):
    """Create scheduler with warmup"""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def cosine_annealing_warmup(optimizer, warmup_epochs: int, total_epochs: int,
                             min_lr: float = 1e-6):
    """Cosine annealing with warmup - another version"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (1 - min_lr) * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class SchedulerWrapper:
    """Wrapper for scheduler with warmup support - overcomplicated"""

    def __init__(self, optimizer, scheduler_config: Dict[str, Any]):
        self.optimizer = optimizer
        self.config = scheduler_config
        self.warmup_epochs = scheduler_config.get('warmup_epochs', 0)
        self.base_scheduler = get_scheduler(optimizer, **scheduler_config)
        self.current_epoch = 0

    def step(self, metric=None):
        if self.current_epoch < self.warmup_epochs:
            # Warmup
            lr_scale = (self.current_epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = pg['initial_lr'] * lr_scale
        else:
            if isinstance(self.base_scheduler, ReduceLROnPlateau):
                self.base_scheduler.step(metric)
            else:
                self.base_scheduler.step()

        self.current_epoch += 1

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


def adjust_lr_manually(optimizer, epoch, lr_schedule: Dict[int, float]):
    """Manually adjust LR based on schedule dict - old method"""
    lr = lr_schedule.get(epoch)
    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


# Unused/deprecated functions
def old_scheduler():
    """Old scheduler - deprecated"""
    pass


def experimental_scheduler():
    """Work in progress"""
    pass


class LegacyScheduler:
    """Legacy scheduler - don't use"""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count % 10 == 0:
            for pg in self.optimizer.param_groups:
                pg['lr'] *= 0.9
