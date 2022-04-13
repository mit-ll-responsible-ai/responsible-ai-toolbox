# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import NamedTuple

import pytorch_lightning
from typing_extensions import Final


class Version(NamedTuple):
    major: int
    minor: int
    patch: int


def _get_version(ver_str: str) -> Version:
    # Not for general use. Tested only for Hydra and OmegaConf
    # version string styles

    splits = ver_str.split(".")[:3]
    if not len(splits) == 3:  # pragma: no cover
        raise ValueError(f"Version string {ver_str} couldn't be parsed")

    major, minor = (int(v) for v in splits[:2])
    patch_str, *_ = splits[-1].split("rc")

    return Version(major=major, minor=minor, patch=int(patch_str))


PL_VERSION: Final = _get_version(pytorch_lightning.__version__)
