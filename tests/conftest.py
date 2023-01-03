# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import logging
import os

import pkg_resources
import pytest
from hypothesis import Verbosity, settings

# usage:
#   pytest tests --hypothesis-profile <profile-name>
settings.register_profile("ci", deadline=None)
settings.register_profile("fast", max_examples=10)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)

_installed = {pkg.key for pkg in pkg_resources.working_set}

MUSHIN_EXTRAS = {"pytorch-lightning", "hydra-zen"}

collect_ignore_glob = []

if not MUSHIN_EXTRAS & _installed:
    collect_ignore_glob.append("test_mushin/*.py")

if "torchviaasion" not in _installed:
    collect_ignore_glob.append("*test_augmentations*")
    collect_ignore_glob.append("*test_datasets*")


@pytest.fixture()
def cleandir(tmp_path):
    """Run function in a temporary directory."""
    old_dir = os.getcwd()  # get current working directory (cwd)
    os.chdir(tmp_path)  # change cwd to the temp-directory
    yield tmp_path  # yields control to the test to be run
    os.chdir(old_dir)
    logging.shutdown()
