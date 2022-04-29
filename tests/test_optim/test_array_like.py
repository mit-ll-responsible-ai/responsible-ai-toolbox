# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Optional

import hypothesis.strategies as st
import torch as tr
from hypothesis import assume, given, note
from torch.testing import assert_allclose

from rai_toolbox.optim import ClampedGradientOptimizer, TopQGradientOptim


@given(
    a=st.none() | st.floats(allow_infinity=False, allow_nan=False, width=32),
    b=st.none() | st.floats(allow_infinity=False, allow_nan=False, width=32),
)
def test_clamped_optim(a: Optional[float], b: Optional[float]):
    assume(a is not None or b is not None)

    grad = tr.arange(-1000.0, 1000.0)
    x = tr.ones_like(grad, requires_grad=True)

    kwargs = (
        dict(clamp_min=a, clamp_max=b)
        if b is not None and a is not None and a < b
        else dict(clamp_min=b, clamp_max=a)
    )

    optim = ClampedGradientOptimizer(params=[x], **kwargs, lr=1.0)

    (x * grad).sum().backward()
    optim.step()

    assert_allclose(x.grad, tr.clamp(grad, kwargs["clamp_min"], kwargs["clamp_max"]))


@given(q=st.sampled_from(tr.linspace(0, 1, 11).tolist()))
def test_top_q_grad(q):
    grad = tr.arange(10.0)
    expected = tr.arange(10.0)
    expected[: round(q * 10)] = 0

    x = tr.ones_like(grad, requires_grad=True)
    optim = TopQGradientOptim(params=[x], q=q, lr=1.0, param_ndim=None)

    (x * grad).sum().backward()
    optim.step()

    note(f"x.grad: {x.grad}")
    note(f"expected: {expected}")
    assert_allclose(x.grad, expected)
