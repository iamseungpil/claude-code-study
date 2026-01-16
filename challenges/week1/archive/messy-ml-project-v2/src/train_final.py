#!/usr/bin/env python3
"""Final version - but it's not actually final"""
import torch
import torch.nn as nn
from model import SimpleCNN
from utils import load_data, save_model

def train():
    model = SimpleCNN()
    train_data, val_data = load_data()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for batch in train_data:
            optimizer.zero_grad()
            # ...
            pass

    save_model(model, "outputs/model_final.pt")

if __name__ == '__main__':
    train()
