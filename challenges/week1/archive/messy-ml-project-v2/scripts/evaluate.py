"""Evaluation script - needs cleanup"""
import torch
import sys
sys.path.append('../src')

from model import SimpleCNN
from utils import load_data


def evaluate(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    _, val_loader = load_data(batch_size=64)

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    print(f"Accuracy: {100 * correct / total:.2f}%")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        evaluate(sys.argv[1])
    else:
        evaluate('../outputs/model_final.pt')
