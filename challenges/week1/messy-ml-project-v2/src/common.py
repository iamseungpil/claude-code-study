"""Common utilities - WHY DO WE HAVE THIS AND helper.py???"""
import torch
import numpy as np


def get_device():
    """Get device - third copy of this function!"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def set_random_seed(seed):
    """Set seed - duplicate of helper.set_seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_accuracy(predictions, labels):
    """Calculate accuracy - similar to helper.compute_accuracy"""
    correct = (predictions.argmax(1) == labels).sum()
    return correct / len(labels)


def format_number(num):
    """Format large numbers with commas."""
    return f"{num:,}"


def get_timestamp():
    """Get current timestamp."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Constants - should be in config
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
RANDOM_SEED = 42
