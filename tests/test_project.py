# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from pytest import Config


def test_version():
    import rai_toolbox

    assert isinstance(rai_toolbox.__version__, str)
    assert rai_toolbox.__version__
    assert "unknown" not in rai_toolbox.__version__


def test_xfail_strict(pytestconfig: Config):
    # Our test suite's xfail must be configured to strict mode
    # in order to ensure that contrapositive tests will actually
    # raise.
    assert pytestconfig.getini("xfail_strict") is True
