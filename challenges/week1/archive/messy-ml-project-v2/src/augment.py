#!/usr/bin/env python3
"""
Data Augmentation module
Augmentation transforms for training
Overlaps with load_data in train_final_REAL.py
"""
import torch
import torchvision.transforms as T
from typing import List, Optional, Tuple
import numpy as np
import random

# Constants - duplicated from train_final_REAL.py
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_basic_transform(normalize: bool = True):
    """Get basic transform - duplicate of load_data logic"""
    transforms = [T.ToTensor()]
    if normalize:
        transforms.append(T.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    return T.Compose(transforms)


def get_train_transform(augment: bool = True, normalize: bool = True):
    """Get training transform with augmentation"""
    transforms = []

    if augment:
        transforms.extend([
            T.RandomRotation(10),
            T.RandomAffine(0, translate=(0.1, 0.1)),
        ])

    transforms.append(T.ToTensor())

    if normalize:
        transforms.append(T.Normalize((MNIST_MEAN,), (MNIST_STD,)))

    return T.Compose(transforms)


def get_val_transform(normalize: bool = True):
    """Get validation transform - same as basic"""
    return get_basic_transform(normalize)


def get_test_transform(normalize: bool = True):
    """Get test transform - duplicate of get_val_transform"""
    return get_basic_transform(normalize)


class RandomErasing:
    """Random erasing augmentation"""

    def __init__(self, p: float = 0.5, scale: Tuple[float, float] = (0.02, 0.33),
                 ratio: Tuple[float, float] = (0.3, 3.3), value: float = 0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        if random.random() > self.p:
            return img

        area = img.size(1) * img.size(2)
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if h < img.size(1) and w < img.size(2):
                x1 = random.randint(0, img.size(1) - h)
                y1 = random.randint(0, img.size(2) - w)
                img[:, x1:x1+h, y1:y1+w] = self.value
                return img

        return img


class Cutout:
    """Cutout augmentation - simple version"""

    def __init__(self, size: int = 8):
        self.size = size

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        y = random.randint(0, h - self.size)
        x = random.randint(0, w - self.size)
        img[:, y:y+self.size, x:x+self.size] = 0
        return img


class MixUp:
    """MixUp augmentation - for batch processing"""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, batch_x, batch_y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size)
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]
        y_a, y_b = batch_y, batch_y[index]
        return mixed_x, y_a, y_b, lam


class CutMix:
    """CutMix augmentation"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, batch_x, batch_y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size, _, h, w = batch_x.size()
        index = torch.randperm(batch_size)

        # Get cutout box
        r = np.sqrt(1 - lam)
        cut_h = int(h * r)
        cut_w = int(w * r)
        cy = random.randint(0, h - cut_h)
        cx = random.randint(0, w - cut_w)

        batch_x[:, :, cy:cy+cut_h, cx:cx+cut_w] = batch_x[index, :, cy:cy+cut_h, cx:cx+cut_w]

        lam = 1 - (cut_h * cut_w / (h * w))
        y_a, y_b = batch_y, batch_y[index]
        return batch_x, y_a, y_b, lam


def get_augmentation_pipeline(config: dict):
    """Get augmentation pipeline from config - factory function"""
    transforms = [T.ToTensor()]

    if config.get('rotation', False):
        transforms.insert(0, T.RandomRotation(config.get('rotation_degrees', 10)))

    if config.get('translate', False):
        transforms.insert(0, T.RandomAffine(0, translate=(0.1, 0.1)))

    if config.get('normalize', True):
        transforms.append(T.Normalize((MNIST_MEAN,), (MNIST_STD,)))

    if config.get('random_erasing', False):
        transforms.append(RandomErasing())

    return T.Compose(transforms)


# Legacy augmentation functions
def old_augment(img):
    """Old augmentation - deprecated"""
    pass


def deprecated_transform():
    """Deprecated transform"""
    pass


class LegacyAugmentor:
    """Legacy augmentation class"""

    def __init__(self):
        self._deprecated = True

    def __call__(self, x):
        return x
