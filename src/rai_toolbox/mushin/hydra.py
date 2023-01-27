# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import warnings

from hydra_zen import zen as _zen

from ..errors import RAIToolboxDeprecationWarning


def zen(*args, **kwargs):
    warnings.warn(
        RAIToolboxDeprecationWarning(
            "`rai_toolbox.hydra.zen` will be deprecated,"
            "please import from `hydra_zen.zen`."
        ),
        stacklevel=2,
    )
    _zen(*args, **kwargs)


zen.__doc__ = _zen.__doc__
