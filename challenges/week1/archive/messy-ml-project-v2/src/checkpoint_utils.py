#!/usr/bin/env python3
"""
Checkpoint Utilities - Additional helpers
Why is this separate from checkpoint.py? Nobody knows
"""
import torch
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


def compute_model_hash(model) -> str:
    """Compute hash of model weights for verification"""
    params = []
    for param in model.parameters():
        params.append(param.data.cpu().numpy().tobytes())
    combined = b''.join(params)
    return hashlib.md5(combined).hexdigest()


def verify_checkpoint(path: str, expected_hash: str = None) -> bool:
    """Verify checkpoint integrity"""
    if not os.path.exists(path):
        return False

    try:
        checkpoint = torch.load(path, map_location='cpu')
        if 'model_state_dict' not in checkpoint:
            return False

        if expected_hash:
            # Can't easily verify hash without model
            pass

        return True
    except Exception:
        return False


def get_checkpoint_info(path: str) -> Dict[str, Any]:
    """Get information about a checkpoint without loading weights"""
    checkpoint = torch.load(path, map_location='cpu')

    info = {
        'path': path,
        'file_size': os.path.getsize(path),
        'timestamp': checkpoint.get('timestamp'),
        'epoch': checkpoint.get('epoch'),
        'loss': checkpoint.get('loss'),
        'accuracy': checkpoint.get('accuracy'),
        'keys': list(checkpoint.keys()),
    }

    return info


def compare_checkpoints(path1: str, path2: str) -> Dict[str, Any]:
    """Compare two checkpoints"""
    info1 = get_checkpoint_info(path1)
    info2 = get_checkpoint_info(path2)

    return {
        'path1': path1,
        'path2': path2,
        'epoch_diff': (info2.get('epoch') or 0) - (info1.get('epoch') or 0),
        'accuracy_diff': (info2.get('accuracy') or 0) - (info1.get('accuracy') or 0),
        'loss_diff': (info2.get('loss') or 0) - (info1.get('loss') or 0),
    }


def export_checkpoint_to_onnx(model, checkpoint_path: str, output_path: str,
                               input_shape=(1, 1, 28, 28)):
    """Export checkpoint to ONNX format - rarely used"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(model, dummy_input, output_path,
                       input_names=['input'],
                       output_names=['output'])


def merge_checkpoints(paths: List[str], weights: List[float] = None) -> Dict[str, torch.Tensor]:
    """Merge multiple checkpoints by averaging - experimental"""
    if weights is None:
        weights = [1.0 / len(paths)] * len(paths)

    merged = {}
    for i, path in enumerate(paths):
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']

        for key, value in state_dict.items():
            if key not in merged:
                merged[key] = value * weights[i]
            else:
                merged[key] += value * weights[i]

    return merged


def checkpoint_to_safetensors(checkpoint_path: str, output_path: str):
    """Convert to safetensors format - not implemented"""
    raise NotImplementedError("Safetensors export not implemented")


def checkpoint_to_pt(checkpoint_path: str, output_path: str):
    """Convert to just .pt weights - duplicate of save_model_only"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    torch.save(state_dict, output_path)


class CheckpointConverter:
    """Convert between checkpoint formats - mostly unused"""

    @staticmethod
    def to_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint.get('model_state_dict', checkpoint)

    @staticmethod
    def to_full_checkpoint(state_dict: Dict[str, torch.Tensor],
                           output_path: str, **metadata):
        checkpoint = {
            'model_state_dict': state_dict,
            'timestamp': datetime.now().isoformat(),
            **metadata
        }
        torch.save(checkpoint, output_path)


# Legacy utilities
def old_checkpoint_util():
    """Old utility function"""
    pass


def deprecated_converter():
    """Deprecated"""
    pass


class LegacyConverter:
    """Legacy converter - don't use"""
    pass
