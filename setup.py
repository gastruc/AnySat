#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="anysat",
    version="0.0.1",
    description="AnySat: An Earth Observation Model for Any Resolutions, Scales, and Modalities",
    author="Guillaume Astruc",
    author_email="guillaume.astruc@enpc.fr",
    url="https://github.com/gastruc/anysat",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
