#!/usr/bin/env python3
"""
Augmentation V2 - Improved augmentation pipeline
But basically same as augment.py
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import random
from typing import Tuple, List, Optional

# Same constants as augment.py
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class AugmentationPipeline:
    """Augmentation pipeline - v2 style"""

    def __init__(self, augmentations: List[str] = None):
        self.augmentations = augmentations or ['rotation', 'translate']
        self._build_pipeline()

    def _build_pipeline(self):
        self.transforms = []

        if 'rotation' in self.augmentations:
            self.transforms.append(T.RandomRotation(15))

        if 'translate' in self.augmentations:
            self.transforms.append(T.RandomAffine(0, translate=(0.15, 0.15)))

        if 'flip' in self.augmentations:
            self.transforms.append(T.RandomHorizontalFlip())

        if 'perspective' in self.augmentations:
            self.transforms.append(T.RandomPerspective(0.2, p=0.5))

        self.transforms.append(T.ToTensor())
        self.transforms.append(T.Normalize((MNIST_MEAN,), (MNIST_STD,)))

        self.pipeline = T.Compose(self.transforms)

    def __call__(self, img):
        return self.pipeline(img)


class RandomAugment:
    """Randomly apply augmentations - like RandAugment but simpler"""

    def __init__(self, n: int = 2, m: int = 10):
        self.n = n  # number of augmentations to apply
        self.m = m  # magnitude

        self.augment_list = [
            self._rotate,
            self._translate_x,
            self._translate_y,
            self._shear_x,
            self._shear_y,
        ]

    def _rotate(self, img, m):
        return TF.rotate(img, m * 3)  # up to 30 degrees

    def _translate_x(self, img, m):
        return TF.affine(img, 0, [m * 3, 0], 1, 0)

    def _translate_y(self, img, m):
        return TF.affine(img, 0, [0, m * 3], 1, 0)

    def _shear_x(self, img, m):
        return TF.affine(img, 0, [0, 0], 1, [m * 3, 0])

    def _shear_y(self, img, m):
        return TF.affine(img, 0, [0, 0], 1, [0, m * 3])

    def __call__(self, img):
        ops = random.sample(self.augment_list, self.n)
        for op in ops:
            img = op(img, self.m)
        return img


class TrivialAugment:
    """Trivial augmentation - randomly pick one augmentation"""

    def __init__(self):
        self.augmentations = [
            T.RandomRotation(30),
            T.RandomAffine(0, translate=(0.2, 0.2)),
            T.RandomAffine(0, shear=20),
            T.RandomPerspective(0.3),
            T.GaussianBlur(3),
        ]

    def __call__(self, img):
        aug = random.choice(self.augmentations)
        return aug(img)


def create_augmented_dataset(dataset, num_augmentations: int = 5):
    """Create dataset with augmentations - memory inefficient"""
    augmented_data = []
    augmented_labels = []

    pipeline = AugmentationPipeline()

    for img, label in dataset:
        augmented_data.append(img)
        augmented_labels.append(label)

        for _ in range(num_augmentations):
            aug_img = pipeline(img)
            augmented_data.append(aug_img)
            augmented_labels.append(label)

    return augmented_data, augmented_labels


def get_strong_augmentation():
    """Strong augmentation for semi-supervised learning"""
    return T.Compose([
        T.RandomRotation(30),
        T.RandomAffine(0, translate=(0.2, 0.2), shear=20),
        T.RandomPerspective(0.3, p=0.5),
        T.ToTensor(),
        T.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])


def get_weak_augmentation():
    """Weak augmentation - almost same as basic transform"""
    return T.Compose([
        T.RandomAffine(0, translate=(0.05, 0.05)),
        T.ToTensor(),
        T.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])


# Duplicate functions from augment.py
def get_train_transform_v2(augment: bool = True):
    """Same as augment.py get_train_transform"""
    from .augment import get_train_transform
    return get_train_transform(augment)


# Unused/experimental
class ExperimentalAugmentor:
    """Experimental augmentation - WIP"""

    def __init__(self):
        self._not_ready = True

    def __call__(self, img):
        return img


def deprecated_augment_v2():
    """Deprecated"""
    pass
