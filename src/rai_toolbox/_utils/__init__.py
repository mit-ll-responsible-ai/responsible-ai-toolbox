# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from numbers import Real
from typing import Any, Optional, Tuple, Union

import torch as tr


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


def _safe_name(x):
    return getattr(x, "__name__", str(x))


def value_check(
    name: str,
    value: Any,
    *,
    type_: Union[type, Tuple[type, ...]] = Real,
    min_: Optional[Union[int, float]] = None,
    max_: Optional[Union[int, float]] = None,
    incl_min: bool = True,
    incl_max: bool = True,
):
    """
    For internal use only.

    Used to check the type of `value`. Numerical types can also be bounds-checked.

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
