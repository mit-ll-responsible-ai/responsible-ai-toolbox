# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
import pytest

from rai_toolbox.errors import RAIToolboxDeprecationWarning
from rai_toolbox.mushin.hydra import zen


def test_deprecation():
    with pytest.warns(RAIToolboxDeprecationWarning):
        zen(lambda x: x)
