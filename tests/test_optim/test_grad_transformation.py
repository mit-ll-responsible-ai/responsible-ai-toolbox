# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import math
from functools import partial
from typing import Callable, Optional, Tuple, Type

import numpy as np
import pytest
import torch as tr
from hypothesis import assume, given, note, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from torch import Tensor
from torch.optim import SGD, Adam
from torch.testing import assert_allclose

from rai_toolbox._typing import Partial
from rai_toolbox.optim import (
    FrankWolfe,
    GradientTransformerOptimizer,
    L1FrankWolfe,
    L1NormedGradientOptim,
    L1qFrankWolfe,
    L1qNormedGradientOptim,
    L2FrankWolfe,
    L2NormedGradientOptim,
    L2ProjectedOptim,
    LinfFrankWolfe,
    LinfProjectedOptim,
    SignedGradientOptim,
)
from rai_toolbox.optim.lp_space import _LpNormOptimizer
from rai_toolbox.optim.optimizer import _to_batch

simple_arrays = hnp.arrays(
    shape=hnp.array_shapes(min_dims=2, max_dims=4),
    dtype=np.float64,
    elements=st.floats(-1e6, 1e6),
)


@st.composite
def tensor_pairs(draw: st.DrawFn) -> Tuple[Tensor, Tensor]:
    """
    Hypothesis strategy that draws two equal-shape, tensors.
    """
    x1: np.ndarray = draw(simple_arrays)
    x2 = draw(hnp.arrays(shape=x1.shape, dtype=x1.dtype, elements=st.floats(-1e6, 1e6)))
    return (tr.tensor(x1), tr.tensor(x2))


simple_tensors = simple_arrays.map(lambda x: tr.tensor(x, requires_grad=True))


@pytest.mark.parametrize(
    "Optimizer",
    [
        partial(L1NormedGradientOptim, lr=0.0),
        partial(L2NormedGradientOptim, lr=0.0),
        partial(SignedGradientOptim, lr=0.0),
        partial(L2ProjectedOptim, lr=0.0, epsilon=1),
        partial(LinfProjectedOptim, lr=0.0, epsilon=1),
        partial(L1qNormedGradientOptim, lr=0.0, q=0.3),
        partial(L1qFrankWolfe, lr=0.0, q=0.3, epsilon=1.0),
        partial(L1FrankWolfe, epsilon=1.0, lr=0.0),
        partial(L2FrankWolfe, epsilon=1.0, lr=0.0),
        partial(LinfFrankWolfe, epsilon=1.0, lr=0.0),
        partial(FrankWolfe),
    ],
)
@given(
    loss=st.integers(),
    include_closure=st.booleans(),
)
def test_closure_is_called_by_step(
    loss: int,
    Optimizer: Partial[tr.optim.Optimizer],
    include_closure: bool,
):
    """Ensures closure is called/passed as expected"""
    optimizer = Optimizer([tr.tensor([0.0])])

    if include_closure:
        out = optimizer.step(lambda: loss)
        assert out == loss
    else:
        out = optimizer.step()
        assert out is None


@pytest.mark.parametrize(
    "Optimizer",
    [
        partial(L1NormedGradientOptim, lr=0.01),
        partial(L2NormedGradientOptim, lr=0.01),
        partial(L1NormedGradientOptim, InnerOpt=Adam, lr=0.01),
        partial(L2NormedGradientOptim, InnerOpt=Adam, lr=0.01),
        partial(SignedGradientOptim, lr=0.01),
        partial(L2ProjectedOptim, lr=0.01, epsilon=1),
        partial(LinfProjectedOptim, lr=0.01, epsilon=1),
        partial(L1qNormedGradientOptim, lr=0.01, q=0.3),
        partial(L1qFrankWolfe, lr=0.5, q=0.3, epsilon=1.0),
        partial(L1FrankWolfe, epsilon=1.0, lr=0.5),
        partial(L2FrankWolfe, epsilon=1.0, lr=0.5),
        partial(LinfFrankWolfe, epsilon=1.0, lr=0.5),
    ],
)
@pytest.mark.parametrize("device", [tr.device("cpu"), tr.device("cuda", index=0)])
@settings(deadline=None)
@given(params_as_dict=st.booleans(), starting_point=st.sampled_from([1.0, -1.0]))
def test_optimizers_descend_quadratic_curve(
    params_as_dict: bool,
    starting_point: float,
    Optimizer: Partial[tr.optim.Optimizer],
    device,
):
    if not tr.cuda.is_available() and device.type == "cuda":
        pytest.skip("Skipping cuda device check.")

    x = tr.tensor([[starting_point]], requires_grad=True, device=device)
    optimizer = Optimizer([x] if not params_as_dict else [{"params": [x]}])

    def closure():
        optimizer.zero_grad()
        (x**2).backward()

    assert abs(x.item()) == 1.0
    assert x.grad is None

    trail = [x.item()]
    for _ in range(100):
        optimizer.step(closure)  # type: ignore
        trail.append(x.item())
    assert abs(x.item()) < 0.01, (optimizer, trail[::10])
    assert x.grad is not None


@given(p=st.floats(1, 2), max_norm=st.floats(0.1, 1.0))
def test_lp_optimizer_uses_correct_norms(p: float, max_norm: float):
    class MyLpOpt(_LpNormOptimizer):
        _p = p

    x = tr.tensor([[1.0, 2.0, 3.0]])

    optim = MyLpOpt([x.clone()], SGD, lr=0.1)

    # ensure per-datum norm is as-expected
    assert tr.allclose(optim.per_datum_norm(x), tr.norm(x, p=p, dim=1))  # type: ignore


x = tr.tensor([[1.0]])


class DummyOpt(tr.optim.Optimizer):
    def __init__(self, params, **kw) -> None:
        super().__init__(params, kw)


@pytest.mark.parametrize(
    "opt, p",
    [
        (L1NormedGradientOptim([x], DummyOpt), 1),
        (L2NormedGradientOptim([x], DummyOpt), 2),
        (L2ProjectedOptim([x], DummyOpt, epsilon=1.0), 2),
        (L2FrankWolfe([x], epsilon=1.0), 2),
    ],
)
def test_lp_optimizer_has_correct_p_value(opt, p):
    assert opt.p == p


@pytest.mark.parametrize("Step", [L2ProjectedOptim, LinfProjectedOptim])
@given(x=simple_arrays.map(tr.tensor), eps=st.floats(1e-3, 1e3))
def test_project_returns_fixed_point(
    Step: Type[L2ProjectedOptim], x: Tensor, eps: float
):
    """Ensures that an unclamped projection returns a fixed point of itself.
    x1 <- projection(x0)
    x1 <- projection(x1)
    """
    x_orig = x.clone()

    s = Step([x], SGD, epsilon=eps, lr=0)

    s.step()  # test project via lr=0
    x1 = x.detach().clone()

    # don't permit no-op
    assume(not tr.all(x1 == x_orig))

    s.project()
    x2 = x.detach().clone()

    note(f"x1: {x1}")
    note(f"x2: {x2}")
    assert tr.allclose(x1, x2, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "Step, norm",
    [
        (L2ProjectedOptim, lambda x: np.linalg.norm(x.reshape(len(x), -1), axis=1)),
        (
            LinfProjectedOptim,
            lambda x: np.linalg.norm(x.reshape(len(x), -1), ord=np.inf, axis=1),
        ),
    ],
)
@given(x=simple_arrays.map(tr.tensor), eps=st.floats(1e-3, 1e3))
def test_projected_value_is_in_constraint_set(
    Step: Type[L2ProjectedOptim],
    norm: Callable[[np.ndarray], np.ndarray],
    x: Tensor,
    eps: float,
):
    """
    Ensures that projected value x satisfies:
         Dist[x - x_0] <= eps
    """
    x_orig = x.clone()
    step = Step([x], SGD, lr=1, epsilon=eps)
    step.step()

    # don't permit no-op
    assume(not tr.all(x == x_orig))

    x1_np = x.numpy()
    dists = norm(x1_np)
    note(f"x1: {x1_np.shape}")
    note(f"dists: {dists.shape}")
    # need to add slight buffer to eps to avoid numerical precision
    # issues that don't actually affect performance
    assert np.all(dists <= (eps + 1e-6))


@pytest.mark.parametrize(
    "Step",
    [
        L1NormedGradientOptim,
        L2NormedGradientOptim,
        SignedGradientOptim,
        # projection with large epsilon should not affect step-size scaling
        partial(L2ProjectedOptim, epsilon=1e20),
        partial(LinfProjectedOptim, epsilon=1e20),
    ],
)
@given(
    tensors=tensor_pairs(),
    step_size=st.floats(0, 1e3),
    scaling_factor=st.floats(0, 1e3),
)
def test_step_scales_linearly_with_stepsize(
    Step: Type[GradientTransformerOptimizer],
    tensors: Tuple[Tensor, Tensor],
    step_size: float,
    scaling_factor: float,
):
    """
    Ensures step vector, dx, scales linearly with step-size along each
    of its dimensions.
    """
    x, grad = tensors
    tr.zeros_like(x)
    x1 = x.clone()
    x2 = x.clone()

    step_fn = Step([x1], SGD, lr=step_size)
    x1.grad = grad.clone()
    step_fn.step()

    transform_gradient_step_fn = Step([x2], SGD, lr=scaling_factor * step_size)
    x2.grad = grad.clone()
    transform_gradient_step_fn.step()

    dx = x1 - x
    scaled_dx = x2 - x
    assert tr.allclose(scaling_factor * dx, scaled_dx, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "Step",
    [
        L1NormedGradientOptim,
        L2NormedGradientOptim,
        partial(L2ProjectedOptim, epsilon=1),
        SignedGradientOptim,
        partial(LinfProjectedOptim, epsilon=1),
    ],
)
@given(
    tensors=tensor_pairs(),
    step_size=st.floats(1e-6, 1e3),
    scaling_factor=st.floats(1e-3, 1e3),
)
def test_transform_gradient_step_is_invariant_to_grad_scale(
    Step: Type[GradientTransformerOptimizer],
    tensors: Tuple[Tensor, Tensor],
    step_size: float,
    scaling_factor: float,
):
    """
    Ensures step vector, dx, is independent of ||grad||
    """
    x, grad = tensors
    x1 = x.clone()
    x2 = x.clone()

    step_fn = Step([x1], SGD, lr=step_size)
    x1.grad = grad.clone()
    step_fn.step()

    # don't permit no-op
    assume(not tr.all(x1 == x2))

    step_fn = Step([x2], SGD, lr=step_size)
    x2.grad = grad.clone() * scaling_factor
    step_fn.step()

    assert tr.allclose(x1, x2, atol=1e-3, rtol=1e-3)


def normed(x: Tensor) -> Tensor:
    """Return x such that each entry along axis-0 is L2-normalized"""
    flat_x = x.view(len(x), -1)
    return (flat_x / (1e-20 + tr.norm(flat_x, dim=-1, keepdim=True))).view(x.shape)


@pytest.mark.parametrize(
    "Step, sign_only",
    [
        (L1NormedGradientOptim, False),
        (L2NormedGradientOptim, False),
        (SignedGradientOptim, True),
        # projection with large epsilon should not affect grad-direction
        (partial(L2ProjectedOptim, epsilon=1e20), False),
        (partial(LinfProjectedOptim, epsilon=1e20), True),
    ],
)
@given(
    tensors=tensor_pairs(),
    step_size=st.floats(1e-3, 1e3),
)
def test_step_is_parallel_to_grad(
    Step: Type[GradientTransformerOptimizer],
    sign_only: bool,
    tensors: Tuple[Tensor, Tensor],
    step_size: float,
):
    """
    Ensures that step vector, dx, is aligned with grad. If `sign_only`
    is True, only check that their dimensions agree in sign.
    """
    x, grad = tensors
    x_orig = x.clone()

    step_fn = Step([x], SGD, lr=step_size)
    x.grad = grad.clone()
    step_fn.step()
    dx = x - x_orig

    if sign_only:
        # only check that each dimension agrees in sign
        assert tr.all((dx * -grad) >= 0)
    else:
        assert tr.allclose(normed(dx), -normed(grad), atol=1e-3, rtol=1e-3)


@given(
    start=st.floats(-20, 20).filter(lambda x: 0.05 < abs(x)),
    n=st.integers(1, 10),
    epsilon=st.floats(0.1, 1.0),
)
def test_fw_lr_disabled_lr_sched(start: float, n: int, epsilon: float):
    # Without lr scaling, linf-FW w/ lr=1 should just bounce back and forth
    # between sides of the eps-1 ball.
    x = tr.tensor([start], requires_grad=True)
    optim = LinfFrankWolfe([x], epsilon=epsilon, lr=1.0, use_default_lr_schedule=False)

    collect = []
    for _ in range(n):
        optim.zero_grad()
        (x**2).sum().backward()
        optim.step()
        collect.append(x.item())

    val = abs(start) / start * epsilon

    expected = [val if n % 2 else -val for n in range(n)]
    assert all(
        [math.isclose(out, y, rel_tol=1e-5) for out, y in zip(collect, expected)]
    ), (collect, expected)


@pytest.mark.parametrize(
    "Optim, p",
    [
        (L1FrankWolfe, 1),
        (partial(L1qFrankWolfe, q=1.0), 1),
        (L2FrankWolfe, 2),
        (LinfFrankWolfe, float("inf")),
    ],
)
@given(
    start=hnp.arrays(
        dtype="float64",
        shape=hnp.array_shapes(min_dims=2, max_dims=2, max_side=10),
        elements=st.floats(-100, 100).filter(lambda x: 0.1 < abs(x)),
    ).map(lambda x: x.tolist()),
    epsilon=st.floats(1.0, 10.0),
)
def test_lp_fw_constraint_sets(
    Optim, p: float, start: Tuple[float, float], epsilon: float
):
    # Ensures Lp-FW optimizer's step up x**2 pushes point onto Lp-ball
    # Exercises dimensionalities 1 -> 10
    x = tr.tensor(start, requires_grad=True)
    optim = Optim([x], epsilon=epsilon)

    (-(x**2)).sum().backward()
    optim.step()

    orig = tr.tensor(start)
    # Ensure step is performed in same direction as initial displacement
    assert tr.all(tr.einsum("nd,nd", x, orig) > 0.0)
    # Ensure result of step resides on Lp-ball
    assert tr.allclose(tr.norm(x, p=p, dim=1), tr.tensor(epsilon))  # type: ignore


@pytest.mark.parametrize(
    "Optimizer",
    [
        partial(L1NormedGradientOptim, lr=0.5),
        partial(L1qNormedGradientOptim, lr=0.5, q=0.1),
        partial(L2NormedGradientOptim, lr=0.5),
        partial(SignedGradientOptim, lr=0.5),
        partial(L2ProjectedOptim, lr=0.5, epsilon=1.5),
        partial(LinfProjectedOptim, lr=0.5, epsilon=1.5),
        partial(L1FrankWolfe, lr=0.5, epsilon=1.5),
        partial(L1qFrankWolfe, lr=0.5, q=0.1, epsilon=1.0),
        partial(L2FrankWolfe, lr=0.5, epsilon=1.5),
        partial(LinfFrankWolfe, lr=0.5, epsilon=1.5),
    ],
)
@given(
    param=hnp.arrays(
        dtype="float64",
        shape=hnp.array_shapes(min_dims=0, max_dims=3),
        elements=st.floats(-100, 100),
    ).map(lambda x: x.tolist())
)
def test_grad_transform_optim_param_ndim_equivalence(Optimizer, param):

    # Default: use param of shape (N=1, D0, ...)
    t1 = tr.tensor([param], requires_grad=True)
    # Test against: param of shape (D0, ...)
    t2 = tr.tensor(param, requires_grad=True)

    # Exercises param_ndim via optim group
    optim = Optimizer(
        [{"params": t1, "param_ndim": -1}, {"params": t2, "param_ndim": None}]
    )

    (t1**2).sum().backward()
    (t2**2).sum().backward()
    optim.step()

    assert t2.ndim == tr.tensor(param).ndim
    assert t1.ndim == t2.ndim + 1

    # Ensure optimizer produces identical results
    assert_allclose(t1, t2[None])


@given(
    param=hnp.arrays(
        dtype="float64",
        shape=hnp.array_shapes(min_dims=0, max_dims=4),
        elements=st.floats(-100, 100).filter(lambda x: 1e-5 < abs(x)),
    ),
    data=st.data(),
)
def test_l2_normed_grad_for_arbitrary_param_ndim(param, data: st.DataObject):
    # Tests all combinations of `p.ndim` vs `param_ndim` and ensures that the
    # expected normalization occurs.

    x = tr.tensor(param, requires_grad=True)
    x_orig = x.clone()
    param_ndim = data.draw(st.none() | st.integers(-x.ndim, x.ndim), label="param_ndim")
    optimizer = L2NormedGradientOptim([x], lr=1.0, param_ndim=param_ndim)

    (x**2).sum().backward()
    optimizer.step()

    if param_ndim is None or param_ndim == x.ndim:
        assert_allclose(tr.norm(x.grad, p=2).item(), 1.0)  # type: ignore
        return

    if param_ndim < 0:
        param_ndim += x.ndim

    if param_ndim == 0:
        assert_allclose(tr.sign(x_orig), x.grad)
        return

    assert x.grad is not None
    assert x.shape == x_orig.shape
    assert x.grad.shape == x_orig.shape

    # Reshapes to (N, D) where D is the size of the trailing `param_ndim`
    # dimensions. Each D-dim vector should have been normalized by the optimizer
    x_grad = x.grad.view(-1, *x.grad.shape[-param_ndim:]).flatten(1)
    norms = tr.norm(x_grad, p=2, dim=1)  # type: ignore
    assert_allclose(norms, tr.ones_like(norms))


@pytest.mark.parametrize(
    "param_ndim, expected_shape",
    [
        (0, (30, 1)),
        (1, (15, 2)),
        (2, (3, 5, 2)),
        (None, (1, 3, 5, 2)),
        (-1, (3, 5, 2)),
        (-2, (15, 2)),
        (-3, (30, 1)),
    ],
)
def test_to_batch_examples(param_ndim, expected_shape):
    p = tr.zeros((3, 5, 2))
    out = _to_batch(p, param_ndim)
    assert out.shape == expected_shape


@given(shape=hnp.array_shapes(min_dims=0, min_side=0, max_dims=10, max_side=5))
def test_param_ndim_0(shape):
    p = tr.zeros(shape)
    out = _to_batch(p, param_ndim=0)
    assert out.shape == (tr.numel(p), 1)


@pytest.mark.parametrize(
    "Optimizer",
    [
        partial(L1NormedGradientOptim, lr=0.5),
        partial(L1qNormedGradientOptim, lr=0.5, q=0.1),
        partial(L2NormedGradientOptim, lr=0.5),
        partial(SignedGradientOptim, lr=0.5),
        partial(L2ProjectedOptim, lr=0.5, epsilon=1.5),
        partial(LinfProjectedOptim, lr=0.5, epsilon=1.5),
    ],
)
@pytest.mark.parametrize(
    "InnerOpt",
    [SGD, Adam],
)
def test_inner_opt_is_set_as_expected(Optimizer, InnerOpt):
    t1 = tr.tensor([1.0], requires_grad=True)
    opt = Optimizer([t1], InnerOpt=InnerOpt)
    assert isinstance(opt.inner_opt, InnerOpt)
    assert opt.inner_opt.param_groups[0]["params"][0] is t1


@settings(max_examples=10)
@given(seed=st.none() | st.integers(10_000, 20_000))
def test_l1q_with_dq_draws_from_user_provided_rng(seed: Optional[int]):
    # Providing a seeded generator should always produce the same results
    saved_grads = set()
    base_gen = tr.Generator().manual_seed(0)

    for _ in range(100):
        # if seed is not None, then the rng in the step should
        #
        gen = tr.Generator().manual_seed(seed) if seed else base_gen

        x = tr.tensor([1.0, 1.0, 1.0, 1.0], requires_grad=True)
        optim = L1qNormedGradientOptim(
            [x],
            InnerOpt=tr.optim.SGD,
            q=0.50,
            dq=1.0,
            lr=1.0,
            param_ndim=None,
            generator=gen,
        )

        (tr.tensor([0.0, 1.0, 2.0, 3.0]) * x).sum().backward()
        optim.step()
        assert x.grad is not None
        saved_grads.add(tuple(x.grad.tolist()))
    if seed:
        assert len(saved_grads) == 1
    else:
        assert len(saved_grads) == 3


@pytest.mark.parametrize(
    "Optim",
    [
        L2NormedGradientOptim,
        L1NormedGradientOptim,
        SignedGradientOptim,
        partial(L1qNormedGradientOptim, q=1.0),
        partial(L2ProjectedOptim, epsilon=1e6),
        partial(LinfProjectedOptim, epsilon=1e6),
    ],
)
@given(
    scales=st.tuples(*[st.floats(0.1, 3.0)] * 3),
    biases=st.tuples(*[st.floats(-10, 10.0)] * 3),
    via_defaults=st.booleans(),
    x=hnp.arrays(
        dtype=float,
        shape=hnp.array_shapes(min_dims=0, max_dims=2, min_side=1),
        elements=st.floats(0.1, 10),
    ),
)
def test_grad_scale_and_bias(
    scales: Tuple[float, float, float],
    biases: Tuple[float, float, float],
    via_defaults: bool,
    x: np.ndarray,
    Optim: Type[GradientTransformerOptimizer],
):
    x1 = tr.tensor(x.copy(), requires_grad=True)
    x2 = tr.tensor(x.copy(), requires_grad=True)
    x3 = tr.tensor(x.copy(), requires_grad=True)

    s1, s2, s3 = scales
    b1, b2, b3 = biases
    if via_defaults:
        defaults = {"grad_scale": s3, "grad_bias": b3}
        kw = {}
    else:
        defaults = {}
        kw = {"grad_scale": s3, "grad_bias": b3}

    optim = Optim(
        [
            {"params": [x1], "grad_scale": s1, "grad_bias": b1},
            {"params": [x2], "grad_scale": s2, "grad_bias": b2},
            {"params": [x3]},  # scale/bias set via defaults
        ],
        lr=1.0,
        param_ndim=None,
        defaults=defaults,
        **kw,
    )

    (x1**2 + x2**2 + x3**2).sum().backward()
    optim.step()

    assert x1.grad is not None
    assert x2.grad is not None
    assert x3.grad is not None

    g1_unnormed = (x1.grad - b1) / s1
    g2_unnormed = (x2.grad - b2) / s2
    g3_unnormed = (x3.grad - b3) / s3

    assert_allclose(g1_unnormed, g2_unnormed)
    assert_allclose(g1_unnormed, g3_unnormed)

    if b1 != b2 or s1 != s2:
        assert tr.any(x1 != x2), (x1, x2)

    if b1 != b3 or s1 != s3:
        assert tr.any(x1 != x3), (x1, x3)
