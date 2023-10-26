# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# pyright: strict
import functools
import inspect
from typing import Any, Callable, Dict, Generic, Iterable, Mapping, Type, TypeVar, Union

import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter
from typing_extensions import Protocol, TypeAlias, TypeGuard

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
    "OptimParams",
]


class Partial(Protocol[T_co]):
    def __call__(self, *args: Any, **kwargs: Any) -> T_co:  # pragma: no cover
        ...


class Optimizer(Protocol):  # pragma: no cover
    defaults: Dict[str, Any]
    state: Any
    param_groups: Any

    def __setstate__(self, state: Any) -> None:
        ...

    def state_dict(self) -> Any:
        ...

    def load_state_dict(self, state_dict: Any) -> None:
        ...

    def zero_grad(self, set_to_none: Any = ...) -> None:
        # In order to maintain compatibility across torch versions we
        # annotate `set_to_none: Any`
        # - torch 1.12 annotates `set_to_none: bool`
        # - torch <1.12 annotates `set_to_none: Optional[bool]`
        ...

    # def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
    def step(self, closure: Any = None) -> Any:
        ...

    def add_param_group(self, param_group: Any) -> None:
        ...


ParamGroup = Mapping[str, Any]

# Needed instead of Type[Optimizer] to deal with __init__ sig mismatch
OptimizerType: TypeAlias = Callable[..., Optimizer]


# type for Optim(params: <>)
OptimParams = Union[Iterable[Tensor], Iterable[Parameter], Iterable[ParamGroup]]


InstantiatesTo = Callable[..., T]


def _is_protocol(cls: Any) -> bool:
    return issubclass(cls, Generic) and getattr(cls, "_is_protocol", False)


# This should *only* be used internally and with care.
# There are complications for using this with protocols.
def instantiates_to(x: Any, co_var_type: Type[T]) -> TypeGuard[InstantiatesTo[T]]:
    """Checks if `x(...)` will return type `co_var_type`.

    Accommodates structural subtyping via protocols.

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


class ArrayLike(Protocol):
    def __array__(self, dtype: Any = ...) -> Any:
        ...
