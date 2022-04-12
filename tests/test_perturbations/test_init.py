# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import pytest
import torch as tr

from rai_toolbox.perturbations.init import (
    uniform_like_l1_n_ball_,
    uniform_like_l2_n_ball_,
)


@pytest.mark.parametrize("dshape", [(1,), (2,), (5,), (2, 3), (5, 2, 3)])
def test_uniform_like_l2_n_ball(dshape):
    # Test adapted from: https://github.com/nengo/nengo/blob/master/nengo/tests/test_dists.py
    samples = tr.zeros(500, *dshape)
    uniform_like_l2_n_ball_(samples)

    samples = samples.flatten(1)
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
    samples = tr.zeros(500, *dshape)
    uniform_like_l1_n_ball_(samples)

    samples = samples.flatten(1)
    d = samples.shape[1]
    # TODO: The mean appears to decay as 1 / (1 + d)
    assert tr.allclose(samples.mean(dim=0), tr.tensor(1.0 / (1 + d)), atol=0.1)

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
