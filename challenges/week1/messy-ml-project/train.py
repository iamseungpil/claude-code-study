import torch
import torch.nn as nn
from model import SimpleCNN
from utils import load_data

# Old training script - DO NOT USE
def train():
    model = SimpleCNN()
    data = load_data()
    # incomplete...
