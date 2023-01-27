# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT


class RAIToolboxException(Exception):
    """Generic parent class for exceptions thrown by rai-toolbox."""


class RAIToolboxDeprecationWarning(RAIToolboxException, FutureWarning):
    """A deprecation warning issued by rai-toolbox.

    Notes
    -----
    This is a subclass of FutureWarning, rather than DeprecationWarning, so
    that the warnings that it emits are not filtered by default.
    """
