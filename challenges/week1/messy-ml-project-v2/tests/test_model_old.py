"""Old model tests - DEPRECATED"""
import torch

def test_old_model():
    from model_old import OldModel
    model = OldModel()
    x = torch.randn(1, 784)
    y = model(x)
    assert y.shape == (1, 10)

if __name__ == '__main__':
    test_old_model()
