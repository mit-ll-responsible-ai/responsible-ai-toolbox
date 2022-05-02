# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from functools import partial

import numpy as np
import pytest
import torch as tr
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from omegaconf import ListConfig
from torch.testing import assert_allclose

from rai_toolbox.perturbations.init import (
    uniform_like_l1_n_ball_,
    uniform_like_l2_n_ball_,
    uniform_like_linf_n_ball_,
)
from rai_toolbox.perturbations.models import AdditivePerturbation

simple_arrays = hnp.arrays(
    shape=hnp.array_shapes(min_dims=2, max_dims=4),
    dtype=np.float64,
    elements=st.floats(-1e6, 1e6),
)


@settings(deadline=None)
@pytest.mark.parametrize("perturber", [AdditivePerturbation])
@pytest.mark.parametrize("dtype", [tr.float32, tr.float64, tr.complex64, tr.complex128])
@pytest.mark.parametrize("device", [tr.device("cpu"), tr.device("cuda", index=0)])
@given(x=simple_arrays.map(tr.tensor))
def test_models_dtype(perturber, dtype, device, x):
    if not tr.cuda.is_available() and device.type == "cuda":
        pytest.skip("Skipping cuda device check.")

    model = perturber(x, dtype=dtype, device=device)
    params = list(model.parameters())
    assert params[0].data.dtype == dtype
    assert params[0].data.device == device


@pytest.mark.parametrize(
    "perturber",
    [AdditivePerturbation],
)
@given(x=simple_arrays.map(tr.tensor))
def test_models_additive_pert(perturber, x):
    model = perturber(x)
    params = list(model.parameters())
    assert len(params) == 1
    assert tr.all(params[0].abs() == 0)

    xpert = model(x)
    tr.testing.assert_allclose(xpert, (x + params[0]))


@pytest.mark.parametrize(
    "init_fn",
    [uniform_like_l1_n_ball_, uniform_like_l2_n_ball_, uniform_like_linf_n_ball_],
)
@given(x=simple_arrays.map(tr.tensor), epsilon=st.floats(1e-3, 1e3))
def test_models_init_fn(init_fn, x, epsilon):
    model = AdditivePerturbation(x)
    params = list(model.parameters())
    assert len(params) == 1
    assert tr.all(params[0].abs() == 0)

    fn = partial(init_fn, epsilon=epsilon)
    model = AdditivePerturbation(x, fn)
    params = list(model.parameters())
    assert len(params) == 1
    assert tr.all(params[0].abs() > 0)


def test_init_fn_kwargs():
    gen1 = tr.Generator().manual_seed(1212121212)
    gen2 = tr.Generator().manual_seed(1212121212)
    gen3 = tr.Generator().manual_seed(1111111111)

    pert1 = AdditivePerturbation((3,), uniform_like_l1_n_ball_, generator=gen1)
    pert2 = AdditivePerturbation((3,), uniform_like_l1_n_ball_, generator=gen2)
    pert3 = AdditivePerturbation((3,), uniform_like_l1_n_ball_, generator=gen3)

    assert_allclose(pert1.delta, pert2.delta)
    assert tr.all(pert1.delta != pert3.delta)


def test_init_fn_validation():
    gen1 = tr.Generator().manual_seed(1212121212)
    with pytest.raises(TypeError):
        AdditivePerturbation((3,), generator=gen1)  # no init_fn


@given(
    data_shape=hnp.array_shapes(min_dims=0, min_side=0)
    | hnp.array_shapes(min_dims=0, min_side=0).map(ListConfig),
    data=st.data(),
    via_shape=st.booleans(),
)
def test_additive_perturbation_broadcasting(
    data_shape, data: st.DataObject, via_shape: bool
):
    ndim = len(data_shape)
    batch = tr.zeros(tuple(data_shape))
    pert_ndim = data.draw(st.none() | st.integers(-ndim, ndim))
    pert_model = AdditivePerturbation(
        (data_shape if via_shape else batch), delta_ndim=pert_ndim
    )
    if pert_ndim is not None:
        if pert_ndim < 0:
            pert_ndim += ndim
    else:
        pert_ndim = ndim
    assert pert_model.delta.ndim == pert_ndim
    assert pert_model(batch).shape == batch.shape
