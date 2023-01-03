# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from functools import partial
from typing import Generic, TypeVar

import pytest
from typing_extensions import Protocol, runtime_checkable

from rai_toolbox._typing import instantiates_to
from rai_toolbox.perturbations import AdditivePerturbation, PerturbationModel

T = TypeVar("T")


class A:
    pass


class SubClassA(A):
    pass


@runtime_checkable
class Prot(Protocol):
    def f(self):
        pass


class B:
    def f(self):
        pass


class SubClassProt(Prot):
    x: int = 1


class NonProtocolGeneric(Generic[T]):
    pass


@pytest.mark.parametrize(
    "obj",
    [
        1,
        "a",
        None,
        (1, 2),
        lambda x: x,
        partial(lambda x: x),
        A(),
        B,
        partial(B),
        B(),
        NonProtocolGeneric,
        NonProtocolGeneric(),
    ],
)
def test_not_instantiable_to_A(obj):
    assert not instantiates_to(obj, A)


@pytest.mark.parametrize("obj", [A, partial(A), SubClassA, partial(SubClassA)])
def test_yes_instantiable_class(obj):
    assert instantiates_to(obj, A)


@pytest.mark.parametrize("obj", [B, partial(B), SubClassProt, partial(SubClassProt)])
def test_yes_instantiable_protocol(obj):
    assert instantiates_to(obj, Prot)


@pytest.mark.parametrize(
    "obj",
    [
        1,
        "a",
        None,
        (1, 2),
        lambda x: x,
        partial(lambda x: x),
        A(),
        A,
        partial(A),
        B(),
        NonProtocolGeneric,
        NonProtocolGeneric(),
    ],
)
def test_not_protocol(obj):
    assert not instantiates_to(obj, Prot)


@pytest.mark.parametrize("obj, protocol", [(AdditivePerturbation, PerturbationModel)])
def test_protocols(obj, protocol):
    assert instantiates_to(obj, protocol)
