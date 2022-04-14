# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import functools
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
)

import numpy as np
from torch import Tensor
from typing_extensions import Protocol, TypeAlias, TypedDict, TypeGuard

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

__all__ = [
    "InstantiatesTo",
    "instantiates_to",
    "Optimizer",
    "OptimParams",
    "OptimizerType",
    "ParamGroup",
    "Partial",
]


class Partial(Protocol[T_co]):
    def __call__(self, *args: Any, **kwargs: Any) -> T_co:  # pragma: no cover
        ...


class Optimizer(Protocol):  # pragma: no cover
    defaults: Dict[str, Any]
    state: dict
    param_groups: Iterable[Mapping[Any, Any]]

    def __setstate__(self, state: dict) -> None:
        ...

    def state_dict(self) -> dict:
        ...

    def load_state_dict(self, state_dict: dict) -> None:
        ...

    def zero_grad(self, set_to_none: Optional[bool] = ...) -> None:
        ...

    def step(self, closure: Optional[Callable[[], Any]] = ...) -> Optional[Any]:
        ...

    def add_param_group(self, param_group: dict) -> None:
        ...


class ParamGroup(TypedDict):
    params: Iterable[Tensor]


# Needed instead of Type[Optimizer] to deal with __init__ sig mismatch
OptimizerType: TypeAlias = Callable[..., Optimizer]


# type for Optim(params: <>)
OptimParams = Union[Sequence[Tensor], Iterable[ParamGroup]]


InstantiatesTo = Union[Type[T_co], Partial[T_co]]


def _is_protocol(cls: Any) -> bool:
    return issubclass(cls, Generic) and getattr(cls, "_is_protocol", False)


# This should *only* be used internally and with care.
# There are complications for using this with protocols.
def instantiates_to(x: Any, co_var_type: Type[T]) -> TypeGuard[InstantiatesTo[T]]:
    """Checks if `x(...)` will return type `co_var_type`.

    Accomodates structural subtyping via protocols.

    Parameters
    ----------
    x : Any

    co_var_type : Type[T]
        A type or a protocol. Note that protocols should not
        include instance-level attributes.

    Returns
    -------
    TypeGuard[InstantiatesTo[T]]
        True if `isinstance(x(...), co_var_type)` would return True."""
    assert isinstance(co_var_type, type), f"`type_` must be a type, got {co_var_type}"

    if inspect.isclass(x):
        obj = x
    elif isinstance(x, functools.partial):
        obj = x.func
        if not inspect.isclass(obj):
            return False
    else:
        return False

    return (
        isinstance(obj, co_var_type)
        if _is_protocol(co_var_type)
        else issubclass(obj, co_var_type)
    )


Scalar = Union[
    int,
    float,
    complex,
    str,
    bytes,
    np.generic,
]


class _SupportsArray(Protocol):
    @overload
    def __array__(self, __dtype: Any = ...) -> np.ndarray:
        ...

    @overload
    def __array__(self, dtype: Any = ...) -> np.ndarray:
        ...


# mypy doesn't support recursive types
ArrayLike = Union[
    Scalar,
    Sequence[Scalar],
    Sequence[Sequence[Any]],
    _SupportsArray,
]
