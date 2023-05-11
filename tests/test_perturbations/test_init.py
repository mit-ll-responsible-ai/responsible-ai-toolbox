# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import math

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import pytest
import torch as tr
from hypothesis import given, settings

from rai_toolbox import to_batch
from rai_toolbox.perturbations.init import (
    uniform_like_l1_n_ball_,
    uniform_like_l2_n_ball_,
    uniform_like_linf_n_ball_,
)


@pytest.mark.parametrize(
    "dshape, param_ndim",
    [
        [(500, 1), -1],
        [(500,), 0],
        [(500, 2), -1],
        [(500, 5), -1],
        [(500, 2, 3), -1],
        [(500, 5, 2, 3), -1],
    ],
)
def test_uniform_like_l2_n_ball(dshape, param_ndim):
    # Test adapted from: https://github.com/nengo/nengo/blob/master/nengo/tests/test_dists.py
    samples = tr.zeros(dshape)
    uniform_like_l2_n_ball_(samples, param_ndim=param_ndim)

    samples = to_batch(samples, param_ndim).flatten(1)
    n, d = samples.shape
    assert tr.allclose(samples.mean(dim=0), tr.tensor(0.0), atol=0.1)

    norms = tr.norm(samples, p=2, dim=1)
    assert tr.all(norms >= 0)
    assert tr.all(norms <= 1)

    # probability of not finding a point in [0, r_tol_min], [r_tol_max, 1]
    q = 1e-5
    r_min_d = 0
    r_tol_min = (r_min_d + (1 - r_min_d) * (1 - q ** (1 / n))) ** (1 / d)
    assert norms.min() <= r_tol_min
    r_tol_max = (1 - (1 - r_min_d) * (1 - q ** (1 / n))) ** (1 / d)
    assert norms.max() >= r_tol_max


@pytest.mark.parametrize("dshape", [(1,), (2,), (5,), (2, 3), (5, 2, 3)])
def test_uniform_like_l1_n_ball(dshape):
    samples = tr.zeros(5000, *dshape)
    uniform_like_l1_n_ball_(samples)

    samples = samples.flatten(1)
    assert samples.mean() < 0.03  # the mean should be zero

    # From: https://mathoverflow.net/questions/9185/how-to-generate-random-points-in-ell-p-balls
    # This ensures that all points fall within the simplex
    assert tr.all(samples.abs().sum(1) <= 1)

    norms = tr.norm(samples, p=1, dim=1)
    assert tr.all(norms >= 0)
    assert tr.all(norms <= 1)

    # TODO: This works but should it be the same as L2?
    # probability of not finding a point in [0, r_tol_min], [r_tol_max, 1]
    # q = 1e-5
    # r_min_d = 0
    # r_tol_min = (r_min_d + (1 - r_min_d) * (1 - q ** (1 / n))) ** (1 / d)
    # assert norms.min() <= r_tol_min
    # r_tol_max = (1 - (1 - r_min_d) * (1 - q ** (1 / n))) ** (1 / d)
    # assert norms.max() >= r_tol_max


@given(
    x=hnp.array_shapes(min_dims=0, max_dims=5).map(tr.zeros),
    epsilon=st.floats(1e-6, 1e6),
)
def test_uniform_like_linf_n_ball(x: tr.Tensor, epsilon: float):
    uniform_like_linf_n_ball_(x, epsilon=epsilon)
    assert tr.all(x.abs() < epsilon)


avail_devices = ["cpu"]

if tr.cuda.is_available():
    avail_devices.append("cuda:0")


@pytest.mark.parametrize(
    "init_",
    [uniform_like_l1_n_ball_, uniform_like_l2_n_ball_, uniform_like_linf_n_ball_],
)
@settings(max_examples=20, deadline=None)
@given(
    seed=st.none() | st.integers(10_000, 20_000),
    x_device=st.sampled_from(avail_devices),
    x_dtype=st.sampled_from([tr.float16, tr.float32, tr.float64]),
    generator_device=st.sampled_from(avail_devices),
)
def test_init_draws_from_user_provided_rng(
    init_, seed: int, x_device: str, x_dtype, generator_device: str
):
    # Providing a seeded generator should always produce the same results
    saved = []
    base_gen = tr.Generator(device=generator_device).manual_seed(0)

    for _ in range(10):
        gen = (
            tr.Generator(device=generator_device).manual_seed(seed)
            if seed
            else base_gen
        )
        x = tr.zeros((100,), device=x_device, dtype=x_dtype)
        init_(x, generator=gen)
        saved.append(x)

        assert x.device == tr.device(x_device)
        assert x.dtype == x_dtype

    first = saved[0]

    for other in saved[1:]:
        if seed is None:
            assert not tr.all(first == other)
        else:
            assert tr.all(first == other)


@pytest.mark.parametrize(
    "init_, ord",
    [
        (uniform_like_l1_n_ball_, 1),
        (uniform_like_l2_n_ball_, 2),
        (uniform_like_linf_n_ball_, float("inf")),
    ],
)
@settings(max_examples=20, deadline=None)
@given(epsilon=st.floats(1e-1, 1e6))
def test_epsilon(init_, ord, epsilon: float):
    tr.Generator().manual_seed(0)

    x = tr.empty(1000, 3)
    init_(x, epsilon=epsilon)
    assert math.isclose(
        tr.linalg.norm(x, ord=ord, dim=1).max(),
        epsilon,
        rel_tol=1e-1,
        abs_tol=1e-1,
    )
