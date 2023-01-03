# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING

from ._utils import to_batch
from ._utils.stateful import evaluating, freeze, frozen
from .losses._utils import negate

__all__ = ["evaluating", "frozen", "freeze", "negate", "to_batch"]

if not TYPE_CHECKING:
    try:
        from ._version import version as __version__
    except ImportError:  # pragma: no cover
        __version__ = "unknown version"
else:  # pragma: no cover
    __version__: str
