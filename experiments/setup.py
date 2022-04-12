# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "matplotlib >= 2.0.0",
    "hydra_zen >= 0.5.0",
    "pytorch-lightning >= 1.5.0",
]

setup(
    name="rai_experiments",
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.7",
    packages=find_packages(where="src", exclude=["tests", "tests.*"]),
    package_dir={"": "src"},
)
