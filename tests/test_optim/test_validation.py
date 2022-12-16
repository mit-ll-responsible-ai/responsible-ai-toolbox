# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from functools import partial

import hypothesis.strategies as st
import pytest
import torch as tr
from hypothesis import given
from hypothesis.extra import numpy as hnp
from torch import Tensor
from torch.optim import SGD

from rai_toolbox.optim import (
    ChainedParamTransformingOptimizer,
    ClampedGradientOptimizer,
    FrankWolfe,
    L2NormedGradientOptim,
    ParamTransformingOptimizer,
)
from rai_toolbox.optim.lp_space import _LpNormOptimizer

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
        ChainedParamTransformingOptimizer(True, 1.0, params=_params, lr=1.0, param_ndim=None)  # type: ignore


def test_bad_params():
    with pytest.raises(TypeError):
        ChainedParamTransformingOptimizer(params=None, InnerOpt=SGD, lr=1.0)


def test_no_clamp_bounds():
    with pytest.raises(ValueError):
        ClampedGradientOptimizer(params=_params, lr=1.0, clamp_min=None, clamp_max=None)


def test_bad_clamp_bounds():
    with pytest.raises(ValueError):
        ClampedGradientOptimizer(params=_params, lr=1.0, clamp_min=1.0, clamp_max=-1.0)


@given(
    param=hnp.arrays(
        dtype="float64",
        shape=hnp.array_shapes(min_dims=0, max_dims=4),
        elements=st.just(0),
    ).map(lambda x: tr.tensor(x, requires_grad=True)),
    data=st.data(),
)
def test_param_ndim_validation(param: Tensor, data: st.DataObject):
    param_ndim = data.draw(
        st.integers().filter(lambda x: abs(x) > param.ndim) | st.floats()
    )

    with pytest.raises((ValueError, TypeError)):
        L2NormedGradientOptim([param], lr=1.0, param_ndim=param_ndim)  # type: ignore


def test_gradient_transform_that_overwrites_grad_raises():
    class BadOptim(ParamTransformingOptimizer):
        def _pre_step_transform_(self, param: Tensor, optim_group) -> None:
            if param.grad is None:
                return
            param.grad = param.grad + 2  # overwrites gradient

    x = tr.tensor([1.0], requires_grad=True)
    optim = BadOptim([x], lr=1.0)
    (2 * x).backward()

    with pytest.raises(ValueError):
        optim.step()


def test_LpNormProjectedOptimizer_requires_p():
    class MyOptim(_LpNormOptimizer):
        # does not set _p
        pass

    with pytest.raises(TypeError):
        MyOptim([], SGD)


@given(p=st.none() | st.text() | st.lists(st.integers()))
def test_non_numeric_p_raises(p):
    class MyOptim(_LpNormOptimizer):
        _p = p

    with pytest.raises(TypeError):
        MyOptim([], SGD)


@given(lr=st.floats().filter(lambda x: not 0 <= x <= 1))
def test_fw_lr_validation_when_lr_sched_is_disabled(lr):
    x = tr.tensor([1.0], requires_grad=True)

    with pytest.raises(ValueError):
        FrankWolfe([x], lr=lr, use_default_lr_schedule=False)
