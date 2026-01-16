"""Setup file - probably outdated"""
from setuptools import setup, find_packages

setup(
    name="mnist-cnn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
    ],
)
