# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "matplotlib >= 2.0.0",
    "hydra_zen >= 0.7.0",
    "pytorch-lightning >= 1.5.0",
    "protobuf <= 3.20.1",
    "pooch >= 1.6.0",
]


DESCRIPTION = (
    "Convenience functions and utilities for rai-toolbox tutorials and demonstrations"
)

LONG_DESCRIPTION = """
rai-experiments provides convenience functions and utilities for rai-toolbox
tutorials and demonstrations.

Note that the contents of rai-experiments are not tested or maintained to
the same fidelity as rai-toolbox. Thus this package is not meant to be relied
on in production. Use at your own risk!

**Installation**

```console
$ pip install rai-experiments"
```
"""

AUTHOR = "Olivia Brown, Justin Goodwin, Ryan Soklaski, Michael Yee"
AUTHOR_EMAIL = "ryan.soklaski@ll.mit.edu"

setup(
    name="rai_experiments",
    version="0.1.0",  # Make sure this matches `rai_experiments.__version__`!
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.7",
    packages=find_packages(where="src", exclude=["tests", "tests.*"]),
    package_dir={"": "src"},
    license="MIT",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
)
