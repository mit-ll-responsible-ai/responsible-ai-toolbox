# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from numbers import Real
from typing import Any, Iterable, Mapping, Optional, Tuple, TypeVar, Union

import torch as tr

T = TypeVar("T", bound=Any)


class Unsatisfiable(AssertionError):  # pragma: no cover
    pass


def get_device(obj: Union[tr.nn.Module, tr.Tensor]) -> tr.device:
    if isinstance(obj, tr.nn.Module):
        for p in obj.parameters():
            return p.device
        return tr.device("cpu")

    elif isinstance(obj, tr.Tensor):
        return obj.device

    else:  # pragma: no cover
        raise TypeError(f"Expected torch.nn.Module or torch.Tensor, got {obj}")


def _safe_name(x: Any) -> str:
    return getattr(x, "__name__", str(x))


def value_check(
    name: str,
    value: T,
    *,
    type_: Union[type, Tuple[type, ...]] = Real,
    min_: Optional[Union[int, float]] = None,
    max_: Optional[Union[int, float]] = None,
    incl_min: bool = True,
    incl_max: bool = True,
) -> T:
    """
    For internal use only.

    Used to check the type of `value`. Numerical types can also be bound-checked.

    Examples
    --------
    >>> value_check("x", 1, type_=str)
    TypeError: `x` must be of type(s) `str`, got 1 (type: int)

    >>> value_check("x", 1, min_=20)
    ValueError: `x` must satisfy 20 <= x  Got: 1

    >>> value_check("x", 1, min_=1, incl_min=False)
    ValueError: `x` must satisfy 1 < x  Got: 1

    >>> value_check("x", 1, min_=1, incl_min=True) # ok
    1
    >>> value_check("x", 0.0, min_=-10, max_=10)  # ok
    0.0

    Raises
    ------
    TypeError, ValueError"""
    # check internal params
    assert isinstance(name, str), name
    assert min_ is None or isinstance(min_, (int, float)), min_
    assert max_ is None or isinstance(max_, (int, float)), max_
    assert isinstance(incl_min, bool), incl_min
    assert isinstance(incl_max, bool), incl_max

    if not isinstance(value, type_):
        raise TypeError(
            f"`{name}` must be of type(s) `{_safe_name(type_)}`, got {value} (type: {_safe_name(type(value))})"
        )

    if min_ is not None and max_ is not None:
        if incl_max and incl_min:
            if not (min_ <= max_):
                raise Unsatisfiable(f"{min_} <= {max_}")
        elif not min_ < max_:
            raise Unsatisfiable(f"{min_} < {max_}")

    min_satisfied = (
        (min_ <= value if incl_min else min_ < value) if min_ is not None else True
    )
    max_satisfied = (
        (value <= max_ if incl_min else value < max_) if max_ is not None else True
    )

    if not min_satisfied or not max_satisfied:
        lsymb = "<=" if incl_min else "<"
        rsymb = "<=" if incl_max else "<"

        err_msg = f"`{name}` must satisfy"

        if min_ is not None:
            err_msg += f" {min_} {lsymb} "

        err_msg += f"{name}"

        if max_ is not None:
            err_msg += f" {rsymb} {max_}"

        err_msg += f"  Got: {value}"

        raise ValueError(err_msg)
    return value


def check_param_group_value(
    name: str,
    param_groups: Iterable[Mapping[str, Any]],
    *,
    type_: Union[type, Tuple[type, ...]] = Real,
    min_: Optional[Union[int, float]] = None,
    max_: Optional[Union[int, float]] = None,
    incl_min: bool = True,
    incl_max: bool = True,
) -> None:
    for group in param_groups:
        value_check(
            name,
            group[name],
            type_=type_,
            max_=max_,
            min_=min_,
            incl_min=incl_min,
            incl_max=incl_max,
        )
