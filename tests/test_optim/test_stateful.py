# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import math
from functools import partial
from typing import Optional

import torch as tr
from torch.optim import Adam
from torch.testing import assert_close

from rai_toolbox._typing import Optimizer
from rai_toolbox.optim import (
    ChainedParamTransformingOptimizer,
    ClampedGradientOptimizer,
    L2NormedGradientOptim,
    L2ProjectedOptim,
    ParamTransformingOptimizer,
    SignedGradientOptim,
)


def _check_consistency(
    optim: ParamTransformingOptimizer, other: Optional[Optimizer] = None
) -> None:
    if other is None:
        other = optim.inner_opt
    assert optim.defaults is other.defaults
    assert optim.state is other.state
    assert optim.param_groups is other.param_groups
    assert optim.state_dict() == other.state_dict()
    assert optim.__getstate__() == other.__getstate__()  # type: ignore


def test_ParamTransformingOptimizer_state_mirrors_InnerOpt():
    x1 = tr.tensor(1.0, requires_grad=True)
    x2 = tr.tensor(1.0, requires_grad=True)
    x3 = tr.tensor(1.0, requires_grad=True)

    optim1 = L2ProjectedOptim(
        [x1], InnerOpt=Adam, lr=0.2, epsilon=100.0, amsgrad=True, param_ndim=None
    )
    optim1.add_param_group({"params": [x2], "lr": 0.1})
    optim1.add_param_group({"params": [x3], "epsilon": 0.4})

    assert not optim1.state
    _check_consistency(optim1)

    assert [group["lr"] for group in optim1.param_groups] == [0.2, 0.1, 0.2]  # type: ignore
    assert all(group["param_ndim"] is None for group in optim1.param_groups)

    (x1 + x2 + x3).backward()
    optim1.step()

    assert math.isclose(x1.item(), 0.8, rel_tol=1e-5)  # (default) lr = 0.2
    assert math.isclose(x2.item(), 0.9, rel_tol=1e-5)  # lr = 0.1
    assert math.isclose(x3.item(), 0.4, rel_tol=1e-5)  # projected by epsilon=0.4

    assert optim1.state  # Adam should be stateful
    _check_consistency(optim1)

    optim2 = L2ProjectedOptim(
        [{"params": [x1]}, {"params": [x2]}, {"params": [x3]}],
        InnerOpt=Adam,
        lr=0.1,
        epsilon=1.0,
        param_ndim=0,
    )

    assert optim2.state_dict() != optim1.state_dict()

    _check_consistency(optim2)

    optim2.load_state_dict(optim1.state_dict())

    assert optim2.state_dict() == optim1.state_dict()
    _check_consistency(optim2)
    optim1.zero_grad(set_to_none=True)

    assert all(p.grad is None for p in [x1, x2, x3])


def test_ChainedParamTransformingOptimizer_state_mirrors_InnerOpt():
    x1 = tr.tensor([[1.0, -1.0]], requires_grad=True)
    x2 = tr.tensor([[1.0, -1.0]], requires_grad=True)
    x3 = tr.tensor([[1.0, -1.0]], requires_grad=True)

    optim1 = ChainedParamTransformingOptimizer(
        partial(ClampedGradientOptimizer, clamp_min=-10000),
        L2NormedGradientOptim,
        params=[x1],
        lr=0.2,
        param_ndim=None,
    )
    optim1.add_param_group({"params": [x2], "lr": 0.1, "param_ndim": 0})
    optim1.add_param_group({"params": [x3], "clamp_min": 0.0})

    assert not optim1.state
    _check_consistency(optim1)

    for opt in optim1._chain:
        _check_consistency(optim1, opt)

    (x1**2 + x2**2 + x3**2).sum().backward()
    optim1.step()

    # param_ndim: None
    assert_close(x1.grad, tr.tensor([[0.7071, -0.7071]]), rtol=1e-4, atol=1e-4)

    # param_ndim: 0
    assert_close(x2.grad, tr.tensor([[1.0, -1.0]]), rtol=1e-4, atol=1e-4)

    # clamp_min: 0.0
    assert_close(x3.grad, tr.tensor([[1.0, 0.0]]), rtol=1e-4, atol=1e-4)

    assert_close(x1, tr.tensor([[0.8586, -0.8586]]), rtol=1e-4, atol=1e-4)
    assert_close(x2, tr.tensor([[0.9000, -0.9000]]), rtol=1e-4, atol=1e-4)
    assert_close(x3, tr.tensor([[0.8000, -1.0000]]), rtol=1e-4, atol=1e-4)

    _check_consistency(optim1)

    for opt in optim1._chain:
        _check_consistency(optim1, opt)

    # check load_state_dict:

    optim2 = ChainedParamTransformingOptimizer(
        partial(ClampedGradientOptimizer, clamp_min=-10000),
        L2NormedGradientOptim,
        params=[{"params": [x1]}, {"params": [x2]}, {"params": [x3]}],
        lr=1.0,
    )

    assert optim2.state_dict() != optim1.state_dict()

    _check_consistency(optim2)

    optim2.load_state_dict(optim1.state_dict())

    assert optim2.state_dict() == optim1.state_dict()
    _check_consistency(optim2)

    for opt in optim2._chain:
        _check_consistency(optim2, opt)


def test_custom_repr():
    opt = SignedGradientOptim(
        [tr.tensor(1.0, requires_grad=True)], InnerOpt=Adam, lr=1.0
    )
    assert repr(opt).startswith("SignedGradientOptim [Adam]")
    assert repr(opt).count("[Adam]") == 1  # make sure we didn't replace too many times


def test_custom_repr_for_chained():
    opt = ChainedParamTransformingOptimizer(
        SignedGradientOptim,
        partial(ClampedGradientOptimizer, clamp_min=1.0),
        params=[tr.tensor([1.0], requires_grad=True)],
        lr=1.0,
    )
    assert repr(opt).startswith("ClampedGradientOptimizer ○ SignedGradientOptim [SGD](")
    assert repr(opt).count("[SGD]") == 1  # make sure we didn't replace too many times


class SimpleOptim(ParamTransformingOptimizer):
    def _pre_step_transform_(self, param, optim_group) -> None:
        param *= 2
        assert param.grad is not None
        param.grad *= 2

    _post_step_transform_ = _pre_step_transform_


def test_inplace_transforms_occur_in_no_grad_context():
    x = tr.tensor(1.0, requires_grad=True)
    x.backward(tr.tensor(2.0))
    optim = SimpleOptim([x], lr=1.0, param_ndim=0)
    optim._apply_pre_step_transform_()
    optim._apply_post_step_transform_()

    assert x.item() == 4.0
    assert x.grad is not None
    assert x.grad.item() == 8.0
