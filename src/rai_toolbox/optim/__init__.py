# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import functools as _fn

from torch.optim import SGD as _SGD

from .frank_wolfe import (
    FrankWolfe,
    L1FrankWolfe,
    L1qFrankWolfe,
    L2FrankWolfe,
    LinfFrankWolfe,
)
from .lp_space import (
    L1NormedGradientOptim,
    L2NormedGradientOptim,
    SignedGradientOptim,
    L2ProjectedOptim,
    LinfProjectedOptim,
    L1qNormedGradientOptim,
    L2NormedGradientOptim,
)
from .optimizer import GradientTransformerOptimizer, ProjectionMixin

# TODO: Get rid of L2PGD and LinfPGD -- their default optimizers are already SGD
L2PGD = _fn.update_wrapper(
    _fn.partial(L2ProjectedOptim, InnerOpt=_SGD), L2ProjectedOptim
)
LinfPGD = _fn.update_wrapper(
    _fn.partial(LinfProjectedOptim, InnerOpt=_SGD), LinfProjectedOptim
)
L1qFW = _fn.update_wrapper(
    _fn.partial(L1qNormedGradientOptim, InnerOpt=FrankWolfe),
    L1qNormedGradientOptim,
)

__all__ = [
    "L1FrankWolfe",
    "L1qFrankWolfe",
    "L2FrankWolfe",
    "LinfFrankWolfe",
    "FrankWolfe",
    "L1NormedGradientOptim",
    "L1qNormedGradientOptim",
    "L2NormedGradientOptim",
    "L2ProjectedOptim",
    "SignedGradientOptim",
    "LinfProjectedOptim",
    "GradientTransformerOptimizer",
    "ProjectionMixin",
    "L2PGD",
    "LinfPGD",
    "L1qFW",
]
