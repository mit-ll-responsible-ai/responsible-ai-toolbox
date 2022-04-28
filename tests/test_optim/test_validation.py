# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT_params = [tr.tensor(1.0, requires_grad=True)]
from functools import partial

import pytest
import torch as tr
from torch.optim import SGD

from rai_toolbox.optim import (
    ChainedGradTransformerOptimizer,
    ClampedGradientOptimizer,
    L2NormedGradientOptim,
)

_params = [tr.tensor(1.0, requires_grad=True)]


@pytest.mark.parametrize(
    "bad_optim",
    [
        pytest.param(
            partial(
                L2NormedGradientOptim,
                [{"params": _params, "grad_scale": 1.0}],
            ),
            marks=pytest.mark.xfail(reason="valid input", strict=True),
        ),
        partial(
            L2NormedGradientOptim,
            [{"params": _params, "grad_scale": "apple"}],
        ),
        partial(
            L2NormedGradientOptim,
            [{"params": _params, "grad_bias": "apple"}],
        ),
        partial(
            L2NormedGradientOptim,
            _params,
            grad_scale="apple",
        ),
        partial(L2NormedGradientOptim, _params, grad_bias="apple"),
        pytest.param(
            partial(L2NormedGradientOptim, _params, grad_bias=2.0),
            marks=pytest.mark.xfail(reason="valid input", strict=True),
        ),
    ],
)
def test_bad_grad_scale_bias(bad_optim):
    with pytest.raises(TypeError):
        bad_optim(lr=1.0, param_ndim=None)


def test_bad_inner_opt():
    with pytest.raises(TypeError):
        ClampedGradientOptimizer(params=_params, InnerOpt=1)  # type: ignore


def test_bad_chain_opt():
    with pytest.raises(TypeError):
        ChainedGradTransformerOptimizer(True, 1.0, params=_params, lr=1.0, param_ndim=None)  # type: ignore


def test_bad_params():
    with pytest.raises(TypeError):
        ChainedGradTransformerOptimizer(params=None, InnerOpt=SGD, lr=1.0)


def test_no_clamp_bounds():
    with pytest.raises(ValueError):
        ClampedGradientOptimizer(params=_params, lr=1.0, clamp_min=None, clamp_max=None)


def test_bad_clamp_bounds():
    with pytest.raises(ValueError):
        ClampedGradientOptimizer(params=_params, lr=1.0, clamp_min=1.0, clamp_max=-1.0)
