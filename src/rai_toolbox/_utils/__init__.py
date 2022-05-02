# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from numbers import Real
from typing import Any, Iterable, Mapping, Optional, Tuple, TypeVar, Union, cast

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
    optional: bool = False,
    lower_name: str = "",
    upper_name: str = "",
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

    if optional and value is None:
        return value

    if not isinstance(value, type_):
        raise TypeError(
            f"`{name}` must be {'None or' if optional else ''}of type(s) `{_safe_name(type_)}`, got {value} (type: {_safe_name(type(value))})"
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
            if lower_name:  # pragma: no cover
                min_ = f"{lower_name}(= {min_})"  # type: ignore
            err_msg += f" {min_} {lsymb}"

        err_msg += f" {name}"

        if max_ is not None:
            if upper_name:
                max_ = f"{upper_name}(= {max_})"  # type: ignore
            err_msg += f" {rsymb} {max_}"

        err_msg += f"  Got: {value}"

        raise ValueError(err_msg)
    return cast(T, value)


def check_param_group_value(
    name: str,
    param_groups: Iterable[Mapping[str, Any]],
    *,
    type_: Union[type, Tuple[type, ...]] = Real,
    min_: Optional[Union[int, float]] = None,
    max_: Optional[Union[int, float]] = None,
    incl_min: bool = True,
    incl_max: bool = True,
    optional: bool = False,
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
            optional=optional,
        )


def _reshape_to_batch(x: tr.Tensor, param_ndim: Optional[int]) -> tr.Tensor:
    """Reshapes to a shape-`(N, d1, ..., dm)`, where `(d1, ..., dm)` has `param_ndim`
    dimensions. Dimensions will be added or consolidated to achieve this."""
    if param_ndim is None:
        param_ndim = x.ndim

    if param_ndim < 0:
        param_ndim += x.ndim

    # `1 + param_ndim` is the required dimensionality
    # for a shape-(N, d0, d1, ...) tensor, where (d0, d1, ...)
    # is the shape of the param_ndim-dimension tensor.
    #
    # We compute `ndim_delta` to determine if we need to add
    # or consolidate dimensions to create the shape-(N, d0, d1, ...)
    # tensor.
    ndim_delta = (1 + param_ndim) - x.ndim

    if ndim_delta > 0:
        # E.g.:
        #   p.shape: (d0, )
        #   desired shape: (N=1, d0)
        x = x[ndim_delta * (None,)]
    elif ndim_delta < 0:
        # E.g.:
        #   p.shape: (d0, d1, d2, d3)
        #   desired shape: (N=d0*d1, d2, d3)
        x = x.view(-1, *x.shape[x.ndim - param_ndim :])
    if x.ndim < 2:  # make at least 2D
        # (N,) -> (N, 1)
        x = x.view(*x.shape, *(1,) * (2 - x.ndim))
    return x


def to_batch(p: tr.Tensor, param_ndim: Optional[int]) -> tr.Tensor:
    """
    Returns a view of `p`, reshaped as shape-(N, d0, ...) where (d0, ...)
    has `param_ndim` entries.

    See Parameters for further description

    Parameters
    ----------
    p : Tensor

    param_ndim: Optional[int]
        Determines the shape of the resulting parameter

        - A positive number determines the dimensionality of the tensor that the transformation will act on.
        - A negative number indicates the 'offset' from the dimensionality of the tensor.
        - `None` means that the transformation will be applied to the tensor without any broadcasting.

    Returns
    -------
    reshaped_p: Tensor, shape-(N, d0, ...)
        Where
        - (d0, ...) is of length `param_ndim` for `param_ndim > 0`
        - (d0, ...) is (1,) for `param_ndim == 0`
        - (d0, ...) is of length `p.ndim - |param_ndim|` for `param_ndim < 0`

    Examples
    --------
    >>> import torch as tr
    >>> x = tr.rand((3, 5, 2))

    >>> to_batch(x, param_ndim=0).shape
    torch.Size([30, 1])

    >>> to_batch(x, param_ndim=1).shape
    torch.Size([15, 2])

    >>> to_batch(x, param_ndim=2).shape
    torch.Size([3, 5, 2])

    >>> to_batch(x, param_ndim=3).shape
    torch.Size([1, 3, 5, 2])

    >>> to_batch(x, param_ndim=None).shape  # same as `param_ndim=x.ndim`
    torch.Size([1, 3, 5, 2])

    >>> to_batch(x, param_ndim=-1).shape
    torch.Size([3, 5, 2])

    >>> to_batch(x, param_ndim=-2).shape
    torch.Size([15, 2])

    >>> to_batch(x, param_ndim=-3).shape
    torch.Size([30, 1])
    """

    # atleast_2d needed for case where p was scalar
    vp = _reshape_to_batch(p, param_ndim=param_ndim)

    if p.grad is not None:
        vp.grad = _reshape_to_batch(p.grad, param_ndim=param_ndim)

    # vp (vp.grad) must be a view of p (p.grad). There is
    # not a simple way to assert this.

    # our views must be size-preserving
    assert tr.numel(vp) == tr.numel(p)
    return vp


def validate_param_ndim(param_ndim: Optional[int], p: tr.Tensor) -> None:
    if param_ndim is not None and p.ndim < abs(param_ndim):
        raise ValueError(
            f"`param_ndim={param_ndim}` specified for parameter "
            f"with ndim={p.ndim} is not valid. `abs(param_ndim) <= "
            f"ndim` must hold."
        )
