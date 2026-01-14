#!/usr/bin/env python3
"""
MNIST Training Script - THE REAL ONE (probably)
Author: Someone who left the company
Last modified: a long time ago
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys
import json
import logging
import numpy as np
from datetime import datetime
import pickle
import time
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TODO: refactor this later
# FIXME: memory leak somewhere
# HACK: temporary fix for the deadline
# XXX: don't touch this, it works somehow

# Magic numbers - don't change these!!
MAGIC_NUMBER_1 = 0.1307
MAGIC_NUMBER_2 = 0.3081
SOME_THRESHOLD = 0.01
BATCH_SIZE_DEFAULT = 64
LR_DECAY_FACTOR = 0.1
MYSTERIOUS_CONSTANT = 42

# Global variables (bad practice but deadline was tight)
global_model = None
global_optimizer = None
global_loss_history = []
current_epoch = 0
is_training = False
debug_mode = True


class SimpleCNN(nn.Module):
    """Simple CNN - copy pasted from stackoverflow"""

    def __init__(self, num_classes=10, dropout_rate=0.25, use_batchnorm=True,
                 hidden_size=128, kernel_size=3, pool_size=2, activation='relu',
                 use_bias=True, init_weights=True, extra_param_1=None, extra_param_2=None):
        super(SimpleCNN, self).__init__()
        # Too many parameters but we might need them later
        self.conv1 = nn.Conv2d(1, 32, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size, padding=1)
        self.pool = nn.MaxPool2d(pool_size, pool_size)
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm2d(32) if use_batchnorm else None
        self.bn2 = nn.BatchNorm2d(64) if use_batchnorm else None

        # Unused attributes
        self.extra_param_1 = extra_param_1
        self.extra_param_2 = extra_param_2
        self._unused_buffer = None
        self._legacy_mode = False

    def forward(self, x):
        # This could be simplified but it works
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def unused_method_1(self):
        """This method is never called"""
        print("This should never print")
        return None

    def unused_method_2(self, x, y, z):
        """Another unused method from old requirements"""
        temp = x + y + z
        return temp * 2

    def deprecated_forward(self, x):
        """Old forward method - DO NOT USE"""
        # Keep for backwards compatibility (?)
        return self.forward(x)


class OldModel(nn.Module):
    """Old model architecture - keeping for reference"""
    def __init__(self):
        super(OldModel, self).__init__()
        self.layer = nn.Linear(784, 10)

    def forward(self, x):
        return self.layer(x.view(-1, 784))


def setup_logging_old():
    """Old logging setup - deprecated"""
    pass


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_data(batch_size=64, data_dir='data/raw', augment=False,
              normalize=True, shuffle=True, num_workers=0, pin_memory=False,
              drop_last=False, validation_split=0.1, seed=42):
    """Load MNIST dataset - way too many parameters"""

    # Duplicate normalization logic (also in preprocess.py)
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(
            transforms.Normalize((MAGIC_NUMBER_1,), (MAGIC_NUMBER_2,))
        )
    if augment:
        # Never actually tested this
        transform_list.insert(0, transforms.RandomRotation(10))
        transform_list.insert(0, transforms.RandomAffine(0, translate=(0.1, 0.1)))

    transform = transforms.Compose(transform_list)

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader


def load_data_old(batch_size):
    """Old data loading function - DO NOT USE"""
    # This doesn't work anymore
    pass


def save_model(model, path, save_optimizer=False, optimizer=None,
               epoch=None, loss=None, accuracy=None, config=None):
    """Save model checkpoint - overcomplicated"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }

    if save_optimizer and optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    if accuracy is not None:
        checkpoint['accuracy'] = accuracy
    if config is not None:
        checkpoint['config'] = config

    torch.save(checkpoint, path)

    # Also save as pickle for some reason
    pickle_path = path.replace('.pt', '.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump({'path': path, 'timestamp': checkpoint['timestamp']}, f)

    logging.info(f"Model saved to {path}")


def save_model_simple(model, path):
    """Simple save - duplicate of save_model"""
    torch.save(model.state_dict(), path)


def load_model_checkpoint(model, path):
    """Load model from checkpoint"""
    checkpoint = torch.load(path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model


def train_one_epoch(model, train_loader, optimizer, criterion, device,
                    epoch, log_interval=100, use_amp=False, scaler=None,
                    gradient_clip=None, warmup_steps=0, scheduler=None):
    """Train for one epoch - too many parameters"""
    global global_loss_history, current_epoch, is_training

    model.train()
    is_training = True
    current_epoch = epoch
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        # Logging
        if batch_idx % log_interval == 0:
            if debug_mode:
                print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    global_loss_history.append(avg_loss)

    return avg_loss, accuracy


def train_one_epoch_old(model, data, optimizer, device):
    """Old training function - keeping for reference"""
    # This was the old implementation
    pass


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def validate_old(model, data, device):
    """Old validation - deprecated"""
    pass


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training curves - hardcoded style"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train')
    ax1.plot(val_losses, label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Curves')

    ax2.plot(train_accs, label='Train')
    ax2.plot(val_accs, label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy Curves')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def helper_function_1():
    """Some helper - not sure what it does"""
    return True


def helper_function_2():
    """Another helper - duplicate in utils.py"""
    return False


def process_data(x):
    """Process data - same as utils.preprocess"""
    return x


def do_something(a, b, c, d, e, f, g):
    """Function with too many parameters and unclear name"""
    result = a + b
    result = result * c
    result = result / d if d != 0 else 0
    result = result - e
    result = result ** f
    return result + g


def compute_metrics(y_true, y_pred, threshold=0.5, average='macro',
                   include_confusion=False, class_names=None):
    """Compute metrics - mostly unused options"""
    # Just compute accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def main():
    """Main training function"""
    global global_model, global_optimizer

    parser = argparse.ArgumentParser(description='Train MNIST CNN')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', type=str, default='outputs/model_final.pt')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=MYSTERIOUS_CONSTANT)
    parser.add_argument('--debug', action='store_true')
    # Unused arguments
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    setup_logging()

    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader = load_data(args.batch_size)

    # Initialize model
    model = SimpleCNN().to(device)
    global_model = model

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    global_optimizer = optimizer

    criterion = nn.CrossEntropyLoss()

    # Training
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, args.output)
            print(f"  Saved best model (acc={best_acc:.4f})")

    # Plot
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        'outputs/training_curves.png'
    )

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")


def main_old():
    """Old main function - for reference only"""
    pass


def test_function():
    """Test function that was never moved to tests/"""
    model = SimpleCNN()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)
    print("Test passed!")


if __name__ == '__main__':
    main()
