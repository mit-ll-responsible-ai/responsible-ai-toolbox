# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import math

import torch as tr
from torch.optim import Adam

from rai_toolbox.optim import (
    GradientTransformerOptimizer,
    L2ProjectedOptim,
    SignedGradientOptim,
)


def _check_consistency(optim: GradientTransformerOptimizer):
    assert optim.defaults is optim.inner_opt.defaults
    assert optim.state is optim.inner_opt.state
    assert optim.param_groups is optim.inner_opt.param_groups
    assert optim.state_dict() == optim.inner_opt.state_dict()
    assert optim.__getstate__() == optim.inner_opt.__getstate__()  # type: ignore


def test_GradientTransformerOptimizer_state_mirrors_InnerOpt():
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

    assert [group["lr"] for group in optim1.param_groups] == [0.2, 0.1, 0.2]
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


def test_custom_repr():
    opt = SignedGradientOptim(
        [tr.tensor(1.0, requires_grad=True)], InnerOpt=Adam, lr=1.0
    )
    assert repr(opt).startswith("SignedGradientOptim [Adam]")
    assert repr(opt).count("[Adam]") == 1  # make sure we didn't replace too many times
