"""New preprocessing - work in progress
TODO: Replace preprocess.py with this
"""
import torch
from torchvision import transforms

# Should use config instead of hardcoding
MEAN = 0.1307
STD = 0.3081


class Preprocessor:
    """Class-based preprocessor - WIP"""

    def __init__(self, augment=False):
        self.augment = augment
        self._setup_transforms()

    def _setup_transforms(self):
        base = [
            transforms.ToTensor(),
            transforms.Normalize((MEAN,), (STD,))
        ]

        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomRotation(10),
                *base
            ])
        else:
            self.transform = transforms.Compose(base)

    def __call__(self, img):
        return self.transform(img)


# Old functions for backwards compatibility
def get_transforms():
    return Preprocessor(augment=False).transform
