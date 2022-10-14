# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from typing import List, Type

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import torch as tr
from hypothesis import given
from torch.testing import assert_close as assert_allclose  # type: ignore

from rai_toolbox.optim import (
    ChainedParamTransformingOptimizer,
    ParamTransformingOptimizer,
)


class PreAdd1Grad(ParamTransformingOptimizer):
    @tr.no_grad()
    def _pre_step_transform_(self, param, optim_group) -> None:
        assert param.grad is not None
        param.grad += 1


class PreAdd1Param(ParamTransformingOptimizer):
    @tr.no_grad()
    def _pre_step_transform_(self, param, optim_group) -> None:
        param += 1


class PreMul2Grad(ParamTransformingOptimizer):
    @tr.no_grad()
    def _pre_step_transform_(self, param, optim_group) -> None:
        assert param.grad is not None
        param.grad *= 2


class PreMul2Param(ParamTransformingOptimizer):
    @tr.no_grad()
    def _pre_step_transform_(self, param, optim_group) -> None:
        param *= 2


class PostAdd1Grad(ParamTransformingOptimizer):
    @tr.no_grad()
    def _post_step_transform_(self, param, optim_group) -> None:
        assert param.grad is not None
        param.grad += 1


class PostAdd1Param(ParamTransformingOptimizer):
    @tr.no_grad()
    def _post_step_transform_(self, param, optim_group) -> None:
        param += 1


class PostMul2Grad(ParamTransformingOptimizer):
    @tr.no_grad()
    def _post_step_transform_(self, param, optim_group) -> None:
        assert param.grad is not None
        param.grad *= 2


class PostMul2Param(ParamTransformingOptimizer):
    @tr.no_grad()
    def _post_step_transform_(self, param, optim_group) -> None:
        param *= 2


@given(
    x=hnp.arrays(
        elements=st.floats(-100, 100, width=32),
        dtype="float32",
        shape=hnp.array_shapes(min_dims=1, max_dims=4),
    ).map(tr.tensor),
    chain=st.lists(
        st.sampled_from(
            [
                ParamTransformingOptimizer,
                PreAdd1Grad,
                PreAdd1Param,
                PreMul2Grad,
                PreMul2Param,
                PostAdd1Grad,
                PostAdd1Param,
                PostMul2Grad,
                PostMul2Param,
            ]
        )
    ),
)
def test_chained_optim(x: tr.Tensor, chain: List[Type[PreMul2Grad]]):
    expected = x.clone()
    expected.requires_grad_(True)
    expected.backward(gradient=tr.ones_like(x))

    x.requires_grad_(True)
    x.backward(gradient=tr.ones_like(x))

    optim = ChainedParamTransformingOptimizer(*chain, params=[x], lr=0.0)
    optim.step()

    for opt in chain:
        opt._pre_step_transform_(None, expected, {})  # type: ignore

    for opt in chain:
        opt._post_step_transform_(None, expected, {})  # type: ignore

    assert_allclose(actual=x, expected=expected)
    assert_allclose(actual=x.grad, expected=expected.grad)
