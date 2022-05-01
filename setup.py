# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from setuptools import find_packages, setup

import versioneer

DISTNAME = "rai_toolbox"
LICENSE = "MIT"
AUTHOR = "Ryan Soklaski, Justin Goodwin, Olivia Brown, Michael Yee"
AUTHOR_EMAIL = "ryan.soklaski@ll.mit.edu"
URL = "https://github.com/mit-ll-responsible-ai/responsible-ai-toolbox"
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering",
]
KEYWORDS = "machine learning robustness pytorch responsible AI"
INSTALL_REQUIRES = [
    "numpy > 1.17.2, != 1.19.3",
    "torchvision >= 0.10.0",
    "torch >= 1.9.0",
    "torchmetrics >= 0.6.0",
    "typing-extensions >= 4.1.1",
]
TESTS_REQUIRE = [
    "pytest >= 3.8",
    "hypothesis >= 6.41.0",
    "mygrad >= 2.0.0",
    "omegaconf >= 2.1.1",
]

DESCRIPTION = "PyTorch-centric library for evaluating and enhancing the robustness of AI technologies"
LONG_DESCRIPTION = """
The rAI-toolbox is designed to enable methods for evaluating and enhancing both the
robustness and the explainability of AI models in a way that is scalable and that
composes naturally with other popular ML frameworks.

A key design principle of the rAI-toolbox is that it adheres strictly to the APIs
specified by the PyTorch machine learning framework. For example, the rAI-toolbox frames
adversarial training workflows solely in terms of the `torch.nn.Optimizer` and
`torch.nn.Module` APIs. This makes it trivial to leverage other libraries and
frameworks from the PyTorch ecosystem to bolster your responsible AI R&D. For
instance, one can naturally leverage the rAI-toolbox together with
PyTorch Lightning to perform distributed adversarial training.
"""


setup(
    name=DISTNAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    url=URL,
    download_url=f"{URL}/tarball/v" + versioneer.get_version(),
    python_requires=">=3.7",
    packages=find_packages(where="src", exclude=["tests", "tests.*"]),
    package_dir={"": "src"},
    package_data={"rai_toolbox": ["py.typed"]},
    extras_require={
        "tests": TESTS_REQUIRE,
        "mushin": [
            "pytorch-lightning >= 1.5.0",
            "hydra-zen >= 0.6.0",
            "xarray >= 0.19.0",
            "matplotlib >= 3.3",
        ],
    },
)
