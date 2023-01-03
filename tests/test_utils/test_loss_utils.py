# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import inspect
from functools import wraps
from typing import Optional

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import pytest
import torch as tr
from hypothesis import given

from rai_toolbox import negate


def f1(x: int, scaled: Optional[bool] = None):
    if scaled is True:
        return 2 * x
    return x


def f2(x: int, y: int, z: int):
    return x + y


def f3(x: int, y: int, z: int):
    return x + y - z


@pytest.mark.parametrize("func", [f1, f2, f3])
def test_negate_preserves_signature(func):
    assert inspect.signature(func) == inspect.signature(negate(func))


def call_f_and_negate_f(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs), negate(func)(*args, **kwargs)

    return wrapper


@pytest.mark.parametrize("func", [f1, f2, f3])
@given(data=st.data())
def test_negate(data: st.DataObject, func):
    orig, negated = data.draw(st.builds(call_f_and_negate_f(func)))
    assert negated == -orig


@given(
    hnp.arrays(
        shape=hnp.array_shapes(),
        dtype="float32",
        elements=dict(allow_infinity=False, allow_nan=False),
    ).map(tr.tensor)
)
def test_negate_works_for_tensors(x):
    assert tr.all(-tr.cos(x) == negate(tr.cos)(x))
