# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import hypothesis.strategies as st
import pytest
import torch as tr
from hypothesis import given

from rai_toolbox._utils.itertools import flatten_params

x1 = tr.tensor([])
x2 = tr.tensor([])
x3 = tr.tensor([])
x4 = tr.tensor([])

module = tr.nn.Linear(2, 2)


@st.composite
def empty_tensors(draw):
    return tr.tensor([])


@given(st.lists(empty_tensors()))
def test_flatten_on_sequence_is_identity(sequence):
    flat = flatten_params(sequence)

    assert len(flat) == len(sequence)

    for t1, t2 in zip(flat, sequence):
        assert t1 is t2


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (x1, [x1]),
        (module.parameters(), list(module.parameters())),
        (tr.optim.SGD([x1, x2], lr=0.2).param_groups, [x1, x2]),
        (
            tr.optim.SGD(
                [dict(params=[x1, x2], lr=0.2), dict(params=[x3, x4], lr=0.1)], lr=0.0
            ).param_groups,
            [x1, x2, x3, x4],
        ),
    ],
)
def test_flatten_params(inputs, expected):
    flat = flatten_params(inputs)

    assert len(flat) == len(expected)

    for t1, t2 in zip(flat, expected):
        assert t1 is t2
