# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import logging
import os
import tempfile
import pkg_resources
import pytest
from hypothesis import Verbosity, settings

# usage:
#   pytest tests --hypothesis-profile <profile-name>
settings.register_profile("ci", deadline=None)
settings.register_profile("fast", max_examples=10)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)

_installed = {pkg.key for pkg in pkg_resources.working_set}

MUSHIN_EXTRAS = {"pytorch_lightning", "hydra_zen"}

collect_ignore_glob = []

if not MUSHIN_EXTRAS <= _installed:
    collect_ignore_glob.append(f"test_mushin/*.py")


@pytest.fixture()
def cleandir():
    """Run function in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        old_dir = os.getcwd()  # get current working directory (cwd)
        os.chdir(tmpdirname)  # change cwd to the temp-directory
        yield tmpdirname  # yields control to the test to be run
        os.chdir(old_dir)
        logging.shutdown()
