#!/usr/bin/env python3
"""Version 2 of training script"""
import torch
import torch.nn as nn
from model import SimpleCNN
from utils import load_data, save_model

def train():
    model = SimpleCNN()
    train_data, val_data = load_data()

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        # training loop
        pass

    save_model(model, "model.pt")

if __name__ == '__main__':
    train()
