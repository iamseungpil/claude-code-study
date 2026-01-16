#!/usr/bin/env python3
"""
Checkpoint utilities
Save and load model checkpoints
Duplicates save_model in train_final_REAL.py
"""
import torch
import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import logging

# Constants
CHECKPOINT_DIR = 'checkpoints'
BEST_MODEL_NAME = 'best_model.pt'
LATEST_MODEL_NAME = 'latest_model.pt'


def save_checkpoint(model, optimizer, epoch: int, loss: float,
                    accuracy: float, path: str, **kwargs):
    """Save training checkpoint - similar to train_final_REAL.py save_model"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat(),
    }
    checkpoint.update(kwargs)

    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, model=None, optimizer=None, device='cpu'):
    """Load checkpoint"""
    checkpoint = torch.load(path, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def save_model_only(model, path: str):
    """Save just the model weights - duplicate of save_model_simple"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model_only(model, path: str, device='cpu'):
    """Load just the model weights"""
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


class CheckpointManager:
    """Manage multiple checkpoints - overcomplicated for this project"""

    def __init__(self, checkpoint_dir: str = CHECKPOINT_DIR,
                 max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def save(self, model, optimizer, epoch: int, metrics: Dict[str, float],
             is_best: bool = False):
        """Save checkpoint and manage old ones"""
        filename = f'checkpoint_epoch_{epoch}.pt'
        path = self.checkpoint_dir / filename

        save_checkpoint(model, optimizer, epoch, metrics.get('loss', 0),
                        metrics.get('accuracy', 0), str(path))

        self.checkpoints.append(path)

        # Remove old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old = self.checkpoints.pop(0)
            if old.exists():
                old.unlink()

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / BEST_MODEL_NAME
            shutil.copy(path, best_path)

        # Always save latest
        latest_path = self.checkpoint_dir / LATEST_MODEL_NAME
        shutil.copy(path, latest_path)

    def load_best(self, model, optimizer=None, device='cpu'):
        """Load best checkpoint"""
        best_path = self.checkpoint_dir / BEST_MODEL_NAME
        if best_path.exists():
            return load_checkpoint(str(best_path), model, optimizer, device)
        return None

    def load_latest(self, model, optimizer=None, device='cpu'):
        """Load latest checkpoint"""
        latest_path = self.checkpoint_dir / LATEST_MODEL_NAME
        if latest_path.exists():
            return load_checkpoint(str(latest_path), model, optimizer, device)
        return None

    def list_checkpoints(self):
        """List all checkpoints"""
        return list(self.checkpoint_dir.glob('checkpoint_*.pt'))


def get_best_checkpoint(checkpoint_dir: str):
    """Find best checkpoint in directory"""
    path = Path(checkpoint_dir) / BEST_MODEL_NAME
    return str(path) if path.exists() else None


def cleanup_checkpoints(checkpoint_dir: str, keep: int = 3):
    """Remove old checkpoints, keeping only recent ones"""
    checkpoints = sorted(Path(checkpoint_dir).glob('checkpoint_*.pt'),
                         key=lambda x: x.stat().st_mtime)
    for ckpt in checkpoints[:-keep]:
        ckpt.unlink()


# Legacy functions
def save_checkpoint_old(model, path):
    """Old save function - deprecated"""
    torch.save(model.state_dict(), path)


def load_checkpoint_old(model, path):
    """Old load function - deprecated"""
    model.load_state_dict(torch.load(path))


class LegacyCheckpointManager:
    """Old checkpoint manager - don't use"""

    def __init__(self):
        self._deprecated = True

    def save(self, model, path):
        save_checkpoint_old(model, path)
