# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# This file is meant to be scanned by the pyright type-checker

from functools import partial
from typing import Union

import torch as tr
from torch.optim import SGD, Adam

from rai_toolbox import negate
from rai_toolbox._typing import Optimizer, OptimizerType, Partial
from rai_toolbox._utils.itertools import flatten_params
from rai_toolbox.optim import (
    FrankWolfe,
    GradientTransformerOptimizer,
    L1FrankWolfe,
    L1NormedGradientOptim,
    L1qFrankWolfe,
    L1qNormedGradientOptim,
    L2FrankWolfe,
    L2NormedGradientOptim,
    L2ProjectedOptim,
    LinfFrankWolfe,
    LinfProjectedOptim,
    SignedGradientOptim,
)


def check_optim_interface():
    x = tr.tensor(1.0)
    SignedGradientOptim([x, x], InnerOpt=SGD)

    # partial'd inner-opt should be OK
    SignedGradientOptim([{"params": x}], InnerOpt=partial(SGD))

    # x needs to be sequence
    SignedGradientOptim(x, InnerOpt=SGD)  # type: ignore


def check_optim_compatibilities():
    def f(Opt: Union[Partial[Optimizer], OptimizerType]):
        opt = Opt([1])
        opt.step()
        flatten_params(opt.param_groups)

    f(SGD)
    f(Adam)
    f(L1FrankWolfe)
    f(L1qFrankWolfe)
    f(L2FrankWolfe)
    f(LinfFrankWolfe)
    f(FrankWolfe)
    f(L1NormedGradientOptim)
    f(L1qNormedGradientOptim)
    f(L2NormedGradientOptim)
    f(L2ProjectedOptim)
    f(SignedGradientOptim)
    f(LinfProjectedOptim)
    f(GradientTransformerOptimizer)


def check_negate():

    # pyright should flag:
    negate(lambda x: None)  # type: ignore
    # because `None` is not negateable. But the following should be OK:
    negate(lambda x: x)
