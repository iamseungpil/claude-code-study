#!/usr/bin/env python3
"""
DO NOT USE THIS FILE
It has bugs and will corrupt your model

Keeping for historical reasons only
"""
import torch
from model_old import OldModel

def train_old():
    # This code has a memory leak
    model = OldModel()
    while True:  # BUG: infinite loop if not careful
        data = torch.randn(1000, 784)
        output = model(data)
        # Missing: loss calculation, optimization
        break  # Added to prevent infinite loop

    torch.save(model, 'model_broken.pt')  # Wrong save format

if __name__ == '__main__':
    print("WARNING: This script is deprecated!")
    train_old()
