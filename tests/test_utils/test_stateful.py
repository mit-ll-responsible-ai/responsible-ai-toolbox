# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
import weakref
import hypothesis.strategies as st
import pytest
import torch as tr
from hypothesis import given

from rai_toolbox import evaluating, freeze, frozen
from rai_toolbox._utils.itertools import flatten_params


def tensors():
    return st.booleans().map(lambda x: tr.tensor(1.0, requires_grad=x))


@st.composite
def models(draw):
    module = tr.nn.Linear(1, 1)
    if draw(st.booleans()):
        module.eval()
    return module


class DummyModule(tr.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = list(params)

    def parameters(self, recurse: bool = True):
        yield from self.params


def as_module(params):
    return DummyModule(params)


def via_freeze(targets, target_tensors, *_):
    unfreeze = freeze(*targets)
    assert all([not p.requires_grad for p in target_tensors])
    unfreeze()


def via_frozen_context(targets, target_tensors, requires_grad_states):
    context = frozen(*targets)

    # check creating context manager doesn't change tensor state
    assert all(
        [
            p.requires_grad is state
            for p, state in zip(target_tensors, requires_grad_states)
        ]
    )

    with context:
        # check frozen within context
        assert all([not p.requires_grad for p in target_tensors])


def via_frozen_decorator(targets, target_tensors, requires_grad_states):
    def f():
        # check frozen within decorated function
        assert all(not p.requires_grad for p in target_tensors)

    decorated = frozen(*targets)(f)

    # check creating decorator doesn't change tensor state
    assert all(
        [
            p.requires_grad is state
            for p, state in zip(target_tensors, requires_grad_states)
        ]
    )
    decorated()


freezables = (
    tensors()
    | st.lists(tensors())
    | st.lists(tensors()).map(DummyModule)
    | st.lists(tensors(), min_size=1).map(lambda x: tr.optim.SGD(x, lr=0.1))
)


@given(targets=st.lists(freezables))
@pytest.mark.parametrize(
    "freeze_methodology", [via_freeze, via_frozen_context, via_frozen_decorator]
)
def test_freeze(targets, freeze_methodology):

    target_tensors = []
    for target in targets:
        if isinstance(target, tr.nn.Module):
            target = target.parameters()
        elif isinstance(target, tr.optim.Optimizer):
            target = target.param_groups
        target_tensors.extend(flatten_params(target))

    requires_grad_states = [p.requires_grad for p in target_tensors]

    freeze_methodology(targets, target_tensors, requires_grad_states)

    # check all states restored
    assert all(
        [
            p.requires_grad is state
            for p, state in zip(target_tensors, requires_grad_states)
        ]
    )


@given(x=tensors(), num_repeats=st.integers(0, 5))
def test_freeze_handles_multiple_references_to_same_tensor(
    x: tr.Tensor, num_repeats: int
):
    original_requires_grad = x.requires_grad
    unfreeze = freeze([x] * (num_repeats + 1))
    assert x.requires_grad is False
    unfreeze()
    assert x.requires_grad is original_requires_grad


def via_eval_context(targets, states):
    context = evaluating(*targets)

    # check creating context manager doesn't change module state
    assert all([p.training is state for p, state in zip(targets, states)])

    with context:
        # check eval within context
        assert all([not p.training for p in targets])


def via_eval_decorator(targets, states):
    def f():
        # check eval within decorated function
        assert all(not p.training for p in targets)

    decorated = evaluating(*targets)(f)

    # check creating decorator doesn't change module state
    assert all([p.training is state for p, state in zip(targets, states)])
    decorated()


@given(models=st.lists(models()))
@pytest.mark.parametrize("eval_methodology", [via_eval_decorator, via_eval_context])
def test_evaluating(models, eval_methodology):
    states = [m.training for m in models]
    eval_methodology(models, states)

    # check states restored
    assert all([p.training is state for p, state in zip(models, states)])


@given(x=models(), num_repeats=st.integers(0, 5))
def test_evaluate_handles_multiple_references_to_same_module(
    x: tr.nn.Module, num_repeats: int
):
    original_status = x.training
    with evaluating(*[x] * (num_repeats + 1)):
        assert x.training is False

    assert x.training is original_status


def test_requires_grad_True_within_freeze_is_restored_to_False():
    x = tr.tensor(1., requires_grad=False)
    assert not x.requires_grad
    with frozen(x):
        x.requires_grad_(True)
        assert x.requires_grad
    assert not x.requires_grad

def test_train_True_within_eval_is_restored_to_False():
    model = tr.nn.Linear(1, 1)
    model.eval()
    assert not model.training

    with evaluating(model):
        assert not model.training
        model.train(True)
        assert model.training
    assert not model.training

@given(...)
def test_freeze_uses_weakrefs(requires_grad: bool):
    x = tr.tensor(1.0, requires_grad=requires_grad)
    xref = weakref.ref(x)
    assert xref() is x
    context = freeze(x)
    del x
    assert xref() is None

@given(...)
def test_evaluating_uses_weakrefs(eval_: bool):
    model = tr.nn.Linear(1, 1)
    if eval_:
        model.eval()

    xref = weakref.ref(model)
    assert xref() is model
    context = evaluating(model)
    del model
    assert xref() is None
