#!/usr/bin/env python3
"""
Utility function tests
Tests for various utility functions
"""
import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestHelperFunctions:
    """Test helper functions"""

    def test_helper_function_1(self):
        """Test helper function 1"""
        from train_final_REAL import helper_function_1
        result = helper_function_1()
        assert result == True

    def test_helper_function_2(self):
        """Test helper function 2"""
        from train_final_REAL import helper_function_2
        result = helper_function_2()
        assert result == False

    def test_process_data(self):
        """Test process_data - just returns input"""
        from train_final_REAL import process_data
        x = [1, 2, 3]
        assert process_data(x) == x


class TestDoSomething:
    """Test the mysterious do_something function"""

    def test_do_something_basic(self):
        """Test do_something with basic inputs"""
        from train_final_REAL import do_something
        result = do_something(1, 2, 2, 2, 1, 1, 0)
        # (1 + 2) * 2 / 2 - 1 = 2, 2^1 = 2, 2 + 0 = 2
        assert result == 2.0

    def test_do_something_zero_division(self):
        """Test do_something handles zero division"""
        from train_final_REAL import do_something
        result = do_something(1, 1, 1, 0, 0, 1, 0)
        assert result == 0.0  # Division by zero returns 0


class TestMetricsModule:
    """Test metrics module"""

    def test_compute_accuracy(self):
        """Test compute_accuracy function"""
        from metrics import compute_accuracy
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 2, 1, 1]  # 4/5 correct
        acc = compute_accuracy(y_true, y_pred)
        assert abs(acc - 0.8) < 0.01

    def test_compute_accuracy_v2(self):
        """Test compute_accuracy_v2 function"""
        from metrics import compute_accuracy_v2
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 2, 1, 1]
        acc = compute_accuracy_v2(y_true, y_pred)
        assert abs(acc - 0.8) < 0.01

    def test_accuracies_match(self):
        """Test that both accuracy functions give same result"""
        from metrics import compute_accuracy, compute_accuracy_v2, compute_accuracy_sklearn
        y_true = [0, 1, 2, 0, 1, 2, 0, 1]
        y_pred = [0, 1, 2, 0, 0, 2, 1, 1]

        acc1 = compute_accuracy(y_true, y_pred)
        acc2 = compute_accuracy_v2(y_true, y_pred)
        acc3 = compute_accuracy_sklearn(y_true, y_pred)

        assert abs(acc1 - acc2) < 0.01
        assert abs(acc2 - acc3) < 0.01


class TestOptimizer:
    """Test optimizer utilities"""

    def test_get_optimizer(self):
        """Test get_optimizer function"""
        from optimizer import get_optimizer
        from train_final_REAL import SimpleCNN

        model = SimpleCNN()
        optimizer = get_optimizer(model, name='adam', lr=0.001)
        assert optimizer is not None

    def test_get_optimizer_sgd(self):
        """Test SGD optimizer"""
        from optimizer import get_optimizer
        from train_final_REAL import SimpleCNN

        model = SimpleCNN()
        optimizer = get_optimizer(model, name='sgd', lr=0.01, momentum=0.9)
        assert optimizer is not None


class TestScheduler:
    """Test scheduler utilities"""

    def test_get_scheduler(self):
        """Test get_scheduler function"""
        from scheduler import get_scheduler
        from optimizer import get_optimizer
        from train_final_REAL import SimpleCNN

        model = SimpleCNN()
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(optimizer, name='step', step_size=5)
        assert scheduler is not None


class TestLayers:
    """Test custom layers"""

    def test_conv_block(self):
        """Test ConvBlock"""
        from layers import ConvBlock

        block = ConvBlock(1, 32)
        x = torch.randn(2, 1, 28, 28)
        out = block(x)
        assert out.shape == (2, 32, 28, 28)

    def test_residual_block(self):
        """Test ResidualBlock"""
        from layers import ResidualBlock

        block = ResidualBlock(32)
        x = torch.randn(2, 32, 14, 14)
        out = block(x)
        assert out.shape == x.shape


# Legacy tests
def test_old_utils():
    """Old utility test - skip"""
    pytest.skip("Deprecated")


def test_placeholder():
    """Placeholder"""
    assert True
