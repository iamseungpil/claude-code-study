"""Tests for model module"""
import pytest
import torch
from src.model import MNISTClassifier, create_model


class TestMNISTClassifier:
    def test_output_shape(self):
        model = MNISTClassifier()
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        assert output.shape == (4, 10)

    def test_custom_num_classes(self):
        model = MNISTClassifier(num_classes=5)
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        assert output.shape == (2, 5)

    def test_factory_function(self):
        model = create_model(num_classes=10, dropout_rate=0.3)
        assert isinstance(model, MNISTClassifier)
