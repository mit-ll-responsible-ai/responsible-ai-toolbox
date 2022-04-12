# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from . import _version
from ._utils.stateful import evaluating, freeze, frozen
from .losses._utils import negate

__all__ = ["evaluating", "frozen", "freeze", "negate"]


__version__ = "0.0.1"
# __version__ = _version.get_versions()["version"]

del _version
