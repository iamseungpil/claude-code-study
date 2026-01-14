#!/usr/bin/env python3
"""Final version 2 - made some improvements"""
import torch
import torch.nn as nn
from model import SimpleCNN
from utils import load_data, save_model

def train():
    model = SimpleCNN()
    train_data, val_data = load_data()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

    for epoch in range(10):
        model.train()
        for batch in train_data:
            optimizer.zero_grad()
            pass
        scheduler.step()

    save_model(model, "outputs/model_final_v2.pt")

if __name__ == '__main__':
    train()
