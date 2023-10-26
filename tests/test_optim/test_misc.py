# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from __future__ import annotations

from functools import partial
from typing import Any, Optional

import hypothesis.strategies as st
import pytest
import torch as tr
from hypothesis import assume, given, note
from torch.testing import assert_close

from rai_toolbox.optim import (
    ClampedGradientOptimizer,
    ClampedParameterOptimizer,
    TopQGradientOptimizer,
)

avail_devices = ["cpu"]

if tr.cuda.is_available():
    avail_devices.append("cuda:0")


@given(
    a=st.none() | st.floats(allow_infinity=False, allow_nan=False, width=32),
    b=st.none() | st.floats(allow_infinity=False, allow_nan=False, width=32),
)
def test_clamped_grad_optim(a: Optional[float], b: Optional[float]):
    assume(a is not None or b is not None)

    grad = tr.arange(-1000.0, 1000.0)
    x = tr.ones_like(grad, requires_grad=True)

    kwargs: dict[str, Any] = (
        dict(clamp_min=a, clamp_max=b)
        if b is not None and a is not None and a < b
        else dict(clamp_min=b, clamp_max=a)
    )

    optim = ClampedGradientOptimizer(params=[x], **kwargs, lr=1.0)

    x.backward(gradient=grad)
    optim.step()

    assert_close(x.grad, tr.clamp(grad, kwargs["clamp_min"], kwargs["clamp_max"]))


@given(
    a=st.none() | st.floats(allow_infinity=False, allow_nan=False, width=32),
    b=st.none() | st.floats(allow_infinity=False, allow_nan=False, width=32),
)
def test_clamped_param_optim(a: Optional[float], b: Optional[float]):
    assume(a is not None or b is not None)

    x = tr.arange(-1000.0, 1000.0, requires_grad=True)
    grad = tr.zeros_like(x)

    kwargs: dict[str, Any] = (
        dict(clamp_min=a, clamp_max=b)
        if b is not None and a is not None and a < b
        else dict(clamp_min=b, clamp_max=a)
    )

    optim = ClampedParameterOptimizer(params=[x], **kwargs, lr=1.0)

    x.backward(gradient=grad)
    optim.step()

    assert_close(
        x,
        tr.clamp(tr.arange(-1000.0, 1000.0), kwargs["clamp_min"], kwargs["clamp_max"]),
    )


@given(q=st.sampled_from(tr.linspace(0, 1, 11).tolist()))
def test_top_q_grad(q):
    grad = tr.arange(10.0)
    expected = tr.arange(10.0)
    expected[: round(q * 10)] = 0

    x = tr.ones_like(grad, requires_grad=True)
    optim = TopQGradientOptimizer(params=[x], q=q, lr=1.0, param_ndim=None)

    (x * grad).sum().backward()
    optim.step()

    note(f"x.grad: {x.grad}")
    note(f"expected: {expected}")
    assert_close(x.grad, expected)


@pytest.mark.parametrize("generator_device", avail_devices)
def test_top_q_grad_generator(generator_device):
    seed = 15873642
    Optim = partial(
        TopQGradientOptimizer,
        q=0.5,
        dq=1.0,
        lr=1.0,
        param_ndim=None,
    )

    def get_grad(seed=None) -> tr.Tensor:
        generator = tr.Generator(device=generator_device)
        if seed is not None:
            generator.manual_seed(seed)

        x = tr.ones((2000,), requires_grad=True)
        optim = Optim(params=[x], generator=generator)
        (x * 1).sum().backward()
        optim.step()
        assert isinstance(x.grad, tr.Tensor)
        return x.grad

    seeded_grad0 = get_grad(seed=seed)
    seeded_grad1 = get_grad(seed=seed)
    unseeded_grad = get_grad(seed=0)

    assert_close(seeded_grad0, seeded_grad1)
    assert tr.any(~tr.isclose(seeded_grad0, unseeded_grad))
