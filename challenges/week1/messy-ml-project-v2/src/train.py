#!/usr/bin/env python3
"""Old training script - DO NOT USE"""
import torch
from model import SimpleCNN
from utils import load_data

# This is the first version - incomplete
def train():
    model = SimpleCNN()
    data = load_data()
    # TODO: finish this
    pass

if __name__ == '__main__':
    train()
