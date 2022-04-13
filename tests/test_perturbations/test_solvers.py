# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from functools import partial
from typing import Any

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import SGD

from rai_toolbox.optim import L2ProjectedOptim, LinfProjectedOptim
from rai_toolbox.perturbations import AdditivePerturbation
from rai_toolbox.perturbations.solvers import (
    _replace_best,
    gradient_descent,
    random_restart,
)


simple_arrays = hnp.arrays(
    shape=hnp.array_shapes(min_dims=2, max_dims=2),
    dtype=np.float64,
    elements=st.floats(-1e6, 1e6),
)


@st.composite
def tensors(draw: st.DrawFn) -> Tensor:
    """
    Hypothesis strategy that draws two equal-shape, tensors.
    """
    x1: np.ndarray = draw(simple_arrays)
    return torch.tensor(x1)


@settings(deadline=None)
@pytest.mark.parametrize(
    "Step, norm",
    [
        (L2ProjectedOptim, lambda x, y: np.linalg.norm(x - y, axis=1)),
        (LinfProjectedOptim, lambda x, y: np.linalg.norm(x - y, ord=np.inf, axis=1)),
    ],
)
@given(tensors=tensors(), eps=st.floats(1e-3, 1))
def test_pgp(Step, norm, tensors, eps):
    orig = tensors
    target = torch.randint(0, 2, size=(orig.shape[0],))
    model = torch.nn.Sequential(
        torch.nn.Linear(orig.shape[1], 2, bias=False), torch.nn.Tanh()
    ).double()

    optimizer = partial(Step, epsilon=eps)

    xadv, _ = gradient_descent(
        model=model,
        data=orig,
        steps=10,
        target=target,
        optimizer=optimizer,
        targeted=False,
        use_best=True,
        criterion=None,
        lr=2.5 * eps / 7,
    )

    dist = norm(xadv.numpy(), orig.numpy())[0]
    assert dist <= eps + 1e-3


def pytorch_simple_model(x: Tensor, device: Any = None):
    import torch

    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            return x

    model = Model().eval().to(device)
    x = torch.zeros(2, 3, 34, 34).to(device)
    y = model(x).argmax(axis=-1)
    return model, x, y


PERTURBATIONS = [
    L2ProjectedOptim,
    LinfProjectedOptim,
]


@pytest.mark.parametrize("Optim", PERTURBATIONS)
@given(x=tensors())
def test_targeted_attacks(Optim, x) -> None:
    with torch.no_grad():
        model, x, y = pytorch_simple_model(x)
        loss = F.cross_entropy(model(x), y, reduction="none")

        num_classes = model(x).shape[-1]
        target_classes = (y + 1) % num_classes

        _, loss_adv = gradient_descent(
            model=model,
            data=x,
            steps=10,
            target=target_classes,
            optimizer=Optim,
            targeted=True,
            lr=1.0,
            epsilon=1.0,
        )
        assert torch.all(loss_adv < loss)


@pytest.mark.parametrize("Optim", PERTURBATIONS)
@given(x=tensors(), use_best=st.booleans())
@pytest.mark.parametrize("repeats", [-1, 0, 1])
def test_random_restart_repeats(Optim, x, use_best, repeats):
    if repeats < 1:
        with pytest.raises(ValueError):
            random_restart(gradient_descent, repeats=repeats)
    else:
        perturber = random_restart(gradient_descent, repeats=repeats)
        with torch.no_grad():
            model, x, y = pytorch_simple_model(x)
            loss = -F.cross_entropy(model(x), y, reduction="none")
            optimizer = partial(Optim, lr=1, epsilon=1)

            _, loss_adv = perturber(
                model=model,
                steps=10,
                data=x,
                target=y,
                optimizer=optimizer,
                use_best=use_best,
            )
            assert torch.all(loss_adv <= loss)


@given(x=tensors(), eps=st.floats(-1, 1))
def test_epsilon_check(x, eps):
    model, x, y = pytorch_simple_model(x)
    optimizer = partial(L2ProjectedOptim, lr=1, epsilon=eps)

    if eps < 0:
        with pytest.raises(AssertionError):
            gradient_descent(
                model=model,
                steps=10,
                data=x,
                target=y,
                optimizer=optimizer,
            )
    else:
        gradient_descent(
            model=model,
            steps=10,
            data=x,
            target=y,
            optimizer=optimizer,
        )


@given(x=tensors(), eps=st.floats(0, 1))
def test_use_best(x, eps):
    model, x, y = pytorch_simple_model(x)
    optimizer = partial(L2ProjectedOptim, lr=1, epsilon=eps)

    criterion_reduce = torch.nn.CrossEntropyLoss(reduction="mean")
    criterion_batch = torch.nn.CrossEntropyLoss(reduction="none")

    # test runtime warning
    with pytest.raises(ValueError):
        gradient_descent(
            model=model,
            steps=10,
            data=x,
            target=y,
            optimizer=optimizer,
            criterion=criterion_reduce,
            use_best=True,
        )

    # get losses with use best = False
    _, losses = gradient_descent(
        model=model,
        steps=10,
        data=x,
        target=y,
        optimizer=optimizer,
        use_best=False,
        criterion=criterion_batch,
    )

    # get best loss
    _, losses_best = gradient_descent(
        model=model,
        steps=10,
        data=x,
        target=y,
        optimizer=optimizer,
        use_best=True,
        criterion=criterion_batch,
    )

    assert torch.all(losses_best <= losses)


@given(min=st.booleans())
def test_replace_best(min):
    loss = torch.zeros(10)
    data = torch.zeros(10, 3)

    best_loss, best_data = _replace_best(loss, None, data, None)
    assert torch.all(best_loss == loss)
    assert torch.all(best_data == data)

    loss = torch.cat((torch.ones(5), -torch.ones(5)))
    data = torch.rand(10, 3)

    if min:
        expected_data = torch.cat((best_data[:5], data[5:]), 0)
    else:
        expected_data = torch.cat((data[:5], best_data[5:]), 0)

    best_loss, _ = _replace_best(loss, best_loss, data, best_data, min=min)

    if min:
        assert torch.all(best_loss <= 0.0)
    else:
        assert torch.all(best_loss >= 0.0)
    assert torch.all(best_data == expected_data)


class IdentityModel(Module):
    def forward(self, data):
        return 1 * data


@given(batch_size=st.integers(1, 10), x=st.floats(-10.0, 10.0))
def test_solutions_invariant_to_batch_size(batch_size, x):
    def loss(pred, target):
        return (pred - target) ** 2

    data = torch.tensor([x] * batch_size)
    target = torch.tensor([0.0] * batch_size)

    # should apply update:: x += 2 * x
    adv, _ = gradient_descent(
        model=IdentityModel(),
        data=data,
        target=target,
        optimizer=partial(SGD, lr=1),
        steps=1,
        criterion=loss,
    )

    assert adv.shape == (batch_size,)
    assert torch.allclose(adv, 3 * data)


@given(x=st.floats(-10.0, 10.0), effective_step_size=st.integers(1, 10))
def test_steps_lr_equivalence_under_special_circumstances(
    x: float, effective_step_size: int
):
    """For L1 loss, identity model, and SGD optimizer: `steps` and `lr`
    should be interchangeable"""

    def loss(pred, target):
        return (pred - target).abs()

    data = torch.tensor([x])
    target = torch.tensor([0.0])

    # should apply update:: x += 1
    adv_via_step, _ = gradient_descent(
        model=IdentityModel(),
        data=data,
        target=target,
        optimizer=SGD,
        lr=1,
        steps=effective_step_size,
        criterion=loss,
    )

    adv_via_lr, _ = gradient_descent(
        model=IdentityModel(),
        data=data,
        target=target,
        optimizer=SGD,
        lr=float(effective_step_size),
        steps=1,
        criterion=loss,
    )
    assert adv_via_step.item() == adv_via_lr.item()


class SomeClass:
    pass


@pytest.mark.parametrize(
    "pert_model",
    [None, lambda x: torch.tensor(x), SomeClass],
)
def test_pert_model_validation(pert_model):
    with pytest.raises(TypeError):
        _ = gradient_descent(
            model=IdentityModel(),
            data=torch.tensor([1.0]),
            target=torch.tensor([0.0]),
            optimizer=SGD,
            steps=1,
            lr=1,
            criterion=lambda pred, target: (pred - target) ** 2,
            perturbation_model=pert_model,
        )


@pytest.mark.parametrize(
    "pert_model",
    [AdditivePerturbation, partial(AdditivePerturbation), "instance"],
)
@given(x=st.floats(-10.0, 10.0))
def test_various_forms_of_pert_model(pert_model, x: float):
    # pert_model can be Type[PertModel], Partial[PertModel], or PertModel
    if pert_model == "instance":
        pert_model = AdditivePerturbation((1,))

    data = torch.tensor([x])

    adv, _ = gradient_descent(
        model=IdentityModel(),
        data=data,
        target=torch.tensor([0.0]),
        optimizer=SGD,
        steps=1,
        lr=1,
        criterion=lambda pred, target: (pred - target) ** 2,
        perturbation_model=pert_model,
    )

    assert torch.allclose(adv, 3 * data)

def test_solve_with_fn_as_model():
    adv, _ = gradient_descent(
        model=lambda x: x ** 2,
        data=torch.tensor([2.0]),
        target=torch.tensor([0.0]),
        optimizer=SGD,
        steps=1,
        lr=1,
        criterion=lambda pred, _: pred,
        targeted=True
    )
    assert adv.item() == -2.0


def test_solve_works_within_no_grad():
    with torch.no_grad():
        adv, _ = gradient_descent(
            model=lambda x: x ** 2,
            data=torch.tensor([2.0]),
            target=torch.tensor([0.0]),
            optimizer=SGD,
            steps=1,
            lr=1,
            criterion=lambda pred, _: pred,
            targeted=True
        )
    assert adv.item() == -2.0