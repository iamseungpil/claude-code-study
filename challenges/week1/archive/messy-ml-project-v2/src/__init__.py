#!/usr/bin/env python3
"""
Messy ML Project - Source Package
This package contains training, model, and utility modules.
Some modules are deprecated but kept for backwards compatibility.
"""

# These imports might fail due to circular dependencies
try:
    from .train_final_REAL import SimpleCNN, load_data, main
except ImportError:
    pass

try:
    from .model import SimpleCNN as ModelCNN  # Might conflict
except ImportError:
    pass

# Version - probably outdated
__version__ = '2.0.0-beta'

# Deprecated exports
__all__ = [
    'SimpleCNN',
    'load_data',
    'main',
]
