# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# This file is meant to be scanned by the pyright type-checker

from functools import partial
from typing import Union

import torch as tr
from torch.optim import SGD, Adam

from rai_toolbox import negate
from rai_toolbox._typing import *
from rai_toolbox._utils.itertools import flatten_params
from rai_toolbox.optim import *


def check_optim_interface():
    x = tr.tensor(1.0)
    SignedGradientOptim([x, x], InnerOpt=SGD)

    # partial'd inner-opt should be OK
    SignedGradientOptim([{"params": x}], InnerOpt=partial(SGD))

    # x needs to be sequence
    SignedGradientOptim(x, InnerOpt=SGD)  # type: ignore

    # bad param-group name
    L2NormedGradientOptim([{"hello": x}], InnerOpt=SGD)  # type: ignore

    L2ProjectedOptim([{"hello": x}], InnerOpt=SGD, epsilon=1.0)  # type: ignore


def check_optim_compatibilities():
    def f(Opt: Union[Partial[Optimizer], OptimizerType]):
        opt = Opt([1])
        opt.step()
        flatten_params(opt.param_groups)

    f(SGD)
    f(Adam)
    f(L2PGD)
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
    f(L2PGD)
    f(LinfPGD)
    f(L1qFW)


def check_negate():

    # pyright should flag:
    negate(lambda x: None)  # type: ignore
    # because `None` is not negateable. But the following should be OK:
    negate(lambda x: x)
