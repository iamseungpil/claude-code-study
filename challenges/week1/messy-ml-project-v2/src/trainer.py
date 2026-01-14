#!/usr/bin/env python3
"""
Trainer module
Encapsulates training logic - but train_final_REAL.py does the same thing
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, List
import logging
import time
import os

from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .metrics import compute_accuracy


class Trainer:
    """Trainer class - duplicates train_final_REAL.py logic"""

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: Dict[str, Any]):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device(config.get('device', 'cpu'))
        self.model.to(self.device)

        self.optimizer = get_optimizer(
            model,
            name=config.get('optimizer', 'adam'),
            lr=config.get('lr', 0.001),
        )
        self.criterion = nn.CrossEntropyLoss()

        scheduler_config = config.get('scheduler', {})
        self.scheduler = get_scheduler(self.optimizer, **scheduler_config) if scheduler_config else None

        self.epochs = config.get('epochs', 10)
        self.log_interval = config.get('log_interval', 100)

        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_acc = 0

        # Unused attributes
        self._internal_state = {}
        self._callbacks = []

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch - duplicate of train_final_REAL.py train_one_epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % self.log_interval == 0:
                logging.info(f'Epoch {self.current_epoch} [{batch_idx}/{len(self.train_loader)}] '
                             f'Loss: {loss.item():.6f}')

        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': correct / total,
        }

    def validate(self) -> Dict[str, float]:
        """Validate - duplicate of train_final_REAL.py validate"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': correct / total,
        }

    def train(self) -> Dict[str, Any]:
        """Full training loop - same as train_final_REAL.py main()"""
        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.train_accs.append(train_metrics['accuracy'])
            self.val_accs.append(val_metrics['accuracy'])

            if self.scheduler:
                self.scheduler.step()

            if val_metrics['accuracy'] > self.best_acc:
                self.best_acc = val_metrics['accuracy']
                self.save_checkpoint('best_model.pt')

            logging.info(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                         f"Acc={train_metrics['accuracy']:.4f} | "
                         f"Val Loss={val_metrics['loss']:.4f}, "
                         f"Acc={val_metrics['accuracy']:.4f}")

        return self.get_history()

    def save_checkpoint(self, path: str):
        """Save checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_acc': self.best_acc,
        }, path)

    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_acc = checkpoint.get('best_acc', 0)

    def get_history(self) -> Dict[str, List[float]]:
        """Get training history"""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }

    # Unused methods
    def unused_method(self):
        pass

    def experimental_train(self):
        """Experimental training - WIP"""
        pass


def train_model(model, train_loader, val_loader, config):
    """Convenience function - just creates Trainer and calls train()"""
    trainer = Trainer(model, train_loader, val_loader, config)
    return trainer.train()


# Legacy trainer
class LegacyTrainer:
    """Old trainer - deprecated"""

    def __init__(self, model, data):
        self.model = model
        self.data = data

    def run(self):
        pass
