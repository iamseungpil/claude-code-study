#!/usr/bin/env python3
"""
MNIST Training Script - Clean Version
"""
import argparse
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.model import create_model
from src.dataset import create_dataloaders
from src.trainer import Trainer
from src.utils import get_device, set_seed, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description='Train MNIST Classifier')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )

    # Model
    model = create_model()
    print(f"Model parameters: {count_parameters(model):,}")

    # Training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=Path(args.checkpoint_dir)
    )

    history = trainer.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs
    )

    # Final evaluation
    _, final_acc = trainer.evaluate(test_loader)
    print(f"\nFinal Test Accuracy: {final_acc:.4f}")


if __name__ == '__main__':
    main()
