# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from typing import Tuple, Union

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_allclose

from rai_toolbox.augmentations.augmix import AugMix, Fork, augment_and_mix


@given(
    num_aug_chains=st.integers(0, 4),
    dirichlet_param=st.floats(0.1, 10.0),
    beta_params=st.floats(0.1, 10.0)
    | st.tuples(st.floats(0.1, 10.0), st.floats(0.1, 10.0)),
    preprocess=st.sampled_from([lambda x: x, lambda x: 2 * x]),
)
def test_zero_chain_depth_returns_datum(
    num_aug_chains: int,
    dirichlet_param: float,
    beta_params: Union[float, Tuple[float, float]],
    preprocess,
):
    x = np.arange(10.0)

    # should be equivalent to preprocess
    augmixed = augment_and_mix(
        x,
        process_fn=preprocess,
        augmentations=[
            lambda x: 2 * x,
            lambda x: 1 + x,
            lambda x: x / 10,
        ],  # augmentations should not have affect
        num_aug_chains=num_aug_chains,
        dirichlet_params=dirichlet_param,
        aug_chain_depth=0,
        beta_params=beta_params,
    )
    assert_allclose(preprocess(x), augmixed)


@given(
    num_aug_chains=st.integers(0, 4),
    dirichlet_param=st.floats(0.1, 10.0),
    aug_chain_depth=st.integers(0, 3),
    preprocess=st.sampled_from([lambda x: x, lambda x: 2 * x]),
)
def test_beta_param_no_augment_returns_datum(
    num_aug_chains: int,
    dirichlet_param: float,
    aug_chain_depth: int,
    preprocess,
):
    # beta params set such that m ~= 0, so that
    # (1 - m) * img_process_fn(image) + m * img_process_fn(augment(image))
    # ~= img_process_fn(image)

    x = np.arange(10.0)

    # should be equivalent to preprocess
    augmixed = augment_and_mix(
        x,
        process_fn=preprocess,
        augmentations=[
            lambda x: 2 * x,
            lambda x: 1 + x,
            lambda x: x / 10,
        ],  # augmentations should not have affect
        num_aug_chains=num_aug_chains,
        dirichlet_params=dirichlet_param,
        aug_chain_depth=aug_chain_depth,
        beta_params=(0.01, 100.0),
    )
    assert_allclose(preprocess(x), augmixed, rtol=1e-5, atol=1e-5)


def identity1(x):
    return x


def identity2(x):
    return x


@settings(max_examples=10)
@given(
    augmentations=st.lists(
        st.sampled_from([identity1, identity2]), min_size=0, max_size=3
    )
)
def test_augmix_reprs(augmentations):
    assert isinstance(repr(AugMix(identity1, augmentations=augmentations)), str)


@settings(max_examples=10)
@given(
    functions=st.lists(st.sampled_from([identity1, identity2]), min_size=1, max_size=7)
)
def test_fork_repr(functions):
    assert isinstance(repr(Fork(*functions)), str)


@settings(max_examples=10)
@given(bad_input=st.sampled_from([[], [1], [identity1, False], [True, identity1]]))
def test_fork_raises_bad_functions(bad_input):
    with pytest.raises((ValueError, TypeError)):
        Fork(*bad_input)


def test_fork():
    two_fork = Fork(lambda x: x, lambda x: 2 * x)
    assert two_fork(2) == (2, 4)

    three_fork = Fork(lambda x: x, lambda x: 2 * x, lambda x: 0 * x)
    assert three_fork(-1.0) == (-1.0, -2.0, -0.0)
