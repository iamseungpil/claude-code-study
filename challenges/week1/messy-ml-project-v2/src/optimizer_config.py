#!/usr/bin/env python3
"""
Optimizer Configuration
Separate config file for optimizer settings - probably overkill
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import json
import os

# Default configs - duplicated from optimizer.py
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MOMENTUM = 0.9
DEFAULT_BETAS = (0.9, 0.999)


@dataclass
class OptimizerConfig:
    """Optimizer configuration dataclass"""
    name: str = 'adam'
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    momentum: float = DEFAULT_MOMENTUM
    betas: Tuple[float, float] = DEFAULT_BETAS
    eps: float = 1e-8
    nesterov: bool = False

    # Unused fields
    warmup_epochs: int = 0
    warmup_lr: float = 1e-6
    min_lr: float = 1e-8
    gradient_clip: Optional[float] = None


@dataclass
class SchedulerConfig:
    """Scheduler configuration - maybe should be in scheduler.py"""
    name: str = 'step'
    step_size: int = 10
    gamma: float = 0.1
    milestones: List[int] = field(default_factory=lambda: [30, 60, 90])
    T_max: int = 100
    eta_min: float = 1e-6


@dataclass
class TrainingOptimConfig:
    """Combined optimizer and scheduler config"""
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    use_amp: bool = False
    accumulation_steps: int = 1


def load_optimizer_config(path: str) -> OptimizerConfig:
    """Load optimizer config from JSON"""
    with open(path, 'r') as f:
        data = json.load(f)
    return OptimizerConfig(**data)


def save_optimizer_config(config: OptimizerConfig, path: str):
    """Save optimizer config to JSON"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)


def get_default_config(model_type: str = 'cnn') -> OptimizerConfig:
    """Get default config based on model type - mostly unused"""
    if model_type == 'cnn':
        return OptimizerConfig(name='adam', lr=0.001)
    elif model_type == 'transformer':
        return OptimizerConfig(name='adamw', lr=1e-4, weight_decay=0.01)
    else:
        return OptimizerConfig()


def merge_configs(base: OptimizerConfig, override: Dict[str, Any]) -> OptimizerConfig:
    """Merge configs - utility function"""
    config_dict = base.__dict__.copy()
    config_dict.update(override)
    return OptimizerConfig(**{k: v for k, v in config_dict.items()
                              if k in OptimizerConfig.__dataclass_fields__})


# Preset configurations
ADAM_CONFIG = OptimizerConfig(name='adam', lr=0.001)
SGD_CONFIG = OptimizerConfig(name='sgd', lr=0.01, momentum=0.9)
ADAMW_CONFIG = OptimizerConfig(name='adamw', lr=1e-4, weight_decay=0.01)

# Legacy configs - shouldn't use these
LEGACY_CONFIG_1 = {'lr': 0.1, 'momentum': 0.9}
LEGACY_CONFIG_2 = {'lr': 0.001}


def old_config_loader(path):
    """Old config loader - deprecated"""
    pass


class ConfigManager:
    """Config manager - overkill for this project"""

    def __init__(self, config_dir: str = 'config'):
        self.config_dir = config_dir
        self._cache = {}

    def get(self, name: str) -> OptimizerConfig:
        if name in self._cache:
            return self._cache[name]
        path = os.path.join(self.config_dir, f'{name}.json')
        config = load_optimizer_config(path)
        self._cache[name] = config
        return config

    def set(self, name: str, config: OptimizerConfig):
        self._cache[name] = config
        path = os.path.join(self.config_dir, f'{name}.json')
        save_optimizer_config(config, path)
