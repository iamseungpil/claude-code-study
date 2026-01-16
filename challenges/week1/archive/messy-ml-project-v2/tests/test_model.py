"""Tests for model - OUTDATED
These tests haven't been run in months
"""
import torch
import sys
sys.path.append('../src')

from model import SimpleCNN


def test_forward():
    model = SimpleCNN()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)
    print("Forward test passed")


def test_backward():
    model = SimpleCNN()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    loss = y.sum()
    loss.backward()
    print("Backward test passed")


def test_parameter_count():
    model = SimpleCNN()
    params = sum(p.numel() for p in model.parameters())
    # This number might be wrong
    assert params > 10000
    print(f"Parameter count: {params}")


# TODO: Add more tests
# FIXME: test_training is broken

def test_training():
    """This test doesn't actually test training"""
    pass


if __name__ == '__main__':
    test_forward()
    test_backward()
    test_parameter_count()
    print("All tests passed!")
