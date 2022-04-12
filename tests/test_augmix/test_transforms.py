# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Tuple, Union

import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.testing import assert_allclose

from rai_toolbox.augmentations.augmix import augment_and_mix


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
