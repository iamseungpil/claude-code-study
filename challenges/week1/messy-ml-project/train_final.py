import torch
import torch.nn as nn
from model import SimpleCNN
from utils import load_data, save_model

# Final version - but not actually final
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
