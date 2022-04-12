# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from functools import wraps
from typing import Any, Callable, TypeVar, cast

from typing_extensions import Protocol


class Negateable(Protocol):
    def __neg__(self) -> Any:  # pragma: no cover
        ...


T = TypeVar("T", bound=Callable[..., Negateable])


def negate(func: T) -> T:
    """A wrapper that negates (applies the `-` operator) to the function's output.

    Parameters
    ----------
    func : Callable[..., Negateable]

    Examples
    --------
    >>> from rai_toolbox import negate
    >>> f = negate(lambda x: 2 * x)
    >>> f(1)
    -2
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return -func(*args, **kwargs)

    return cast(T, wrapper)
