# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import hypothesis.strategies as st
import numpy as np
import pytest
import torch as tr
from hypothesis import given, settings
from hypothesis.extra.numpy import array_shapes, arrays
from mygrad import no_autodiff
from mygrad.nnet.activations import softmax
from torch.testing import assert_allclose

from rai_toolbox.losses import jensen_shannon_divergence

softmax = no_autodiff(softmax, to_numpy=True)

common_shape = array_shapes(min_dims=2, max_dims=2)

prob_tensors = arrays(
    shape=st.shared(common_shape, key="common_shape"),
    dtype="float",
    elements=st.floats(-1000, 1000),
).map(lambda arr: tr.tensor(softmax(arr)))


@settings(max_examples=10)
@given(
    st.lists(
        st.sampled_from([tr.tensor(1.0), tr.tensor([1.0]), tr.tensor([[[1.0]]])]),
        min_size=0,
        max_size=4,
    )
)
def test_jsd_validation(bad_input):
    with pytest.raises(ValueError):
        jensen_shannon_divergence(*bad_input)


@given(probs=st.lists(prob_tensors, min_size=2), data=st.data())
def test_jsd_symmetry(probs, data: st.DataObject):
    perm_probs = data.draw(st.permutations(probs))
    jsd1 = jensen_shannon_divergence(*probs)
    jsd2 = jensen_shannon_divergence(*perm_probs)
    assert_allclose(jsd1, jsd2, atol=1e-5, rtol=1e-5)


@given(probs=st.lists(prob_tensors, min_size=2), weight=st.floats(-10, 10))
def test_jsd_scaled_by_weight(probs, weight: float):
    jsd1 = jensen_shannon_divergence(*probs)
    jsd2 = jensen_shannon_divergence(*probs, weight=weight)
    assert_allclose(jsd1 * weight, jsd2, atol=1e-5, rtol=1e-5)


@given(probs=st.lists(prob_tensors, min_size=2))
def test_jsd_bounds(probs):
    jsd = float(jensen_shannon_divergence(*probs).item())
    assert 0 - 1e-6 <= jsd <= np.log(len(probs)) + 1e-6


@given(num_probs=st.integers(2, 100))
def test_jsd_max_bounds(num_probs):
    """JSD(P1, P2, ..., Pn)"""
    probs = (t[None] for t in tr.eye(num_probs))
    jsd = float(jensen_shannon_divergence(*probs).item())
    assert_allclose(jsd, np.log(num_probs))
