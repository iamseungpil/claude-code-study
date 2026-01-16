#!/usr/bin/env python3
"""
Legacy Trainer - OLD VERSION
DO NOT USE - kept for "backwards compatibility"
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

# Old constants
OLD_LR = 0.1
OLD_MOMENTUM = 0.9
OLD_BATCH_SIZE = 128


class OldTrainer:
    """Original trainer implementation from v1.0"""

    def __init__(self, model, train_data, test_data, lr=OLD_LR):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.lr = lr

        # Old-style initialization
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=OLD_MOMENTUM)
        self.criterion = nn.CrossEntropyLoss()

        self.history = {'loss': [], 'acc': []}
        self._deprecated = True

    def train_one_epoch(self):
        """Old training loop - inefficient"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(self.train_data):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(self.train_data)
        epoch_acc = correct / total

        self.history['loss'].append(epoch_loss)
        self.history['acc'].append(epoch_acc)

        return epoch_loss, epoch_acc

    def evaluate(self):
        """Old evaluation"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_data:
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return correct / total

    def run(self, epochs):
        """Run training"""
        for e in range(epochs):
            loss, acc = self.train_one_epoch()
            val_acc = self.evaluate()
            print(f"Epoch {e+1}: Loss={loss:.4f}, Acc={acc:.4f}, Val Acc={val_acc:.4f}")

        return self.history


class VeryOldTrainer:
    """Even older trainer - from v0.5"""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.optimizer = optim.SGD(model.parameters(), lr=0.01)

    def train(self, epochs=10):
        for e in range(epochs):
            for x, y in self.data:
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = nn.functional.cross_entropy(out, y)
                loss.backward()
                self.optimizer.step()


def old_train_function(model, data, epochs):
    """Very old training function - don't use"""
    trainer = VeryOldTrainer(model, data)
    trainer.train(epochs)


def deprecated_training_loop(model, data, epochs, lr):
    """Another deprecated function"""
    pass


class ExperimentalTrainer:
    """Experimental trainer - never finished"""

    def __init__(self):
        self._not_implemented = True

    def train(self):
        raise NotImplementedError("This trainer was never completed")


# More dead code
def _internal_train_helper():
    pass


def _another_unused_function():
    pass


if False:
    # This code never runs
    class NeverUsedTrainer:
        pass
