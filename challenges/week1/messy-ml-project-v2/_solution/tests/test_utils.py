"""Tests for utils module"""
import pytest
import torch
import torch.nn as nn
from src.utils import get_device, count_parameters, set_seed


def test_get_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_count_parameters():
    model = nn.Linear(10, 5)
    params = count_parameters(model)
    assert params == 10 * 5 + 5  # weights + bias


def test_set_seed():
    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    assert torch.equal(a, b)
