# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)

import torch
from torch import Tensor
from torch.optim import SGD, Optimizer

from rai_toolbox._typing import Optimizer as Opt
from rai_toolbox._typing import OptimizerType, OptimParams, ParamGroup, Partial

_T = TypeVar("_T", bound=Optional[Union[Tensor, float]])

__all__ = ["GradientTransformerOptimizer", "ProjectionMixin"]


class DatumParamGroup(ParamGroup):
    param_ndim: Optional[int]


def _identity(x):
    return x


def _atleast_2d(x: Tensor):
    if x.ndim < 2:
        # (N,) -> (N, 1)
        return x.view(*x.shape, *(1,) * (2 - x.ndim))
    return x


def _shares_memory(x: Tensor, y: Tensor):
    return x.storage().data_ptr() == y.storage().data_ptr()  # type: ignore


def _to_batch(p: Tensor, param_ndim: Optional[int]) -> Tensor:
    """
    Returns a view of `p`, reshaped as shape-(N, d0, ...) where (d0, ...)
    has `param_ndim` entries.

    See Parameters for further description

    Parameters
    ----------
    p: Tensor

    param_ndim: Optional[int]
        Determines the shape of the resulting parameter

        - A positive number determines the dimensionality of the tensor that the transformation will act on.
        - A negative number indicates the 'offset' from the dimensionality of the tensor.
        - `None` means that the transformation will be applied to the tensor without any broadcasting.

    Returns
    -------
    reshaped_p: Tensor

    Examples
    --------
    >>> import torch as tr
    >>> x = tr.rand((3, 5, 2))

    >>> _to_batch(x, 0).shape
    torch.Size([30, 1])

    >>> _to_batch(x, 1).shape
    torch.Size([15, 2])

    >>> _to_batch(x, 2).shape
    torch.Size([3, 5, 2])

    >>> _to_batch(x, None).shape
    torch.Size([1, 3, 5, 2])

    >>> _to_batch(x, -1).shape
    torch.Size([3, 5, 2])

    >>> _to_batch(x, -2).shape
    torch.Size([15, 2])

    >>> _to_batch(x, -3).shape
    torch.Size([30, 1])
    """
    if param_ndim is None:
        param_ndim = p.ndim

    assert p.ndim >= abs(param_ndim)

    if param_ndim < 0:
        param_ndim += p.ndim

    # `1 + param_ndim` is the required dimensionality
    # for a shape-(N, d0, d1, ...) tensor, where (d0, d1, ...)
    # is the shape of the param_ndim-dimension tensor.
    #
    # We compute `ndim_delta` to determine if we need to add
    # or consolidate dimensions to create the shape-(N, d0, d1, ...)
    # tensor.
    ndim_delta = (1 + param_ndim) - p.ndim

    reshape = _identity  # type: ignore

    if ndim_delta > 0:
        # E.g.:
        #   p.shape: (d0, )
        #   desired shape: (N=1, d0)
        def reshape(x):
            return x[ndim_delta * (None,)]

    elif ndim_delta < 0:
        # E.g.:
        #   p.shape: (d0, d1, d2, d3)
        #   desired shape: (N=d0*d1, d2, d3)
        param_shape = p.shape[p.ndim - param_ndim :]

        def reshape(x):
            return x.view(-1, *param_shape)

    # atleast_2d needed for case where p was scalar
    vp = _atleast_2d(reshape(p))

    if p.grad is not None:
        vp.grad = _atleast_2d(reshape(p.grad))

    # vp (vp.grad) must be a view of p (p.grad). There is
    # not a simple way to assert this.

    # our views must be size-preserving
    assert torch.numel(vp) == torch.numel(p)
    return vp


class GradientTransformerOptimizer(Optimizer, metaclass=ABCMeta):
    """An optimizer that performs an in-place transformation to the
    gradient of each parameter, before performing the gradient-based
    update on each parameter::

       param = step(param, transformation(param.grad), ...)

    Notes
    -----
    `GradientTransformerOptimizer` is designed to be combined with other,
    standard gradient-based optimizers (e.g. Adam) via encapsulation, rather
    then through inheritance. I.e., `GradientTransformerOptimizer(InnerOpt=<...>)`
    permits a standard optimizer to include an additional gradient-transformation

    Methods
    -------
    _inplace_grad_transform_
    """

    param_groups: List[DatumParamGroup]

    def __init__(
        self,
        params: OptimParams,
        InnerOpt: Union[Partial[Opt], OptimizerType] = SGD,
        *,
        param_ndim: Optional[int] = -1,
        defaults: Optional[dict] = None,
        **inner_opt_kwargs,
    ) -> None:
        """
        Parameters
        ----------
        params: Iterable
            iterable of parameters to optimize or dicts defining parameter groups

        InnerOpt: Type[Optimizer] | Partial[Optimizer], optional (default=SGD)
            The optimizer to update parameters

        param_ndim : Optional[int] = -1
            Controls how `_inplace_grad_transform_` is broadcast onto the gradient
            of a given parameter. This can be specified per param-group. By default,
            the gradient transformation broadcasts over the first dimension in a
            batch-like style.

               - A positive number determines the dimensionality of the gradient that the transformation will act on.
               - A negative number indicates the 'offset' from the dimensionality of the gradient.
               - `None` means that the transformation will be applied to the gradient without any broadcasting.

        defaults: Optional[dict]
            Specifies default parameters for all parameter groups.

        Notes
        -----
        Additional Explanation of `param_ndim`:

        If the gradient has a shape (d0, d1, d2) and `param_ndim=1` then the
        transformation will be broadcast over each shape-(d2,) sub-tensor in the
        gradient (of which there are d0 * d1).

        E.g. if a gradient has a shape (d0, d1, d2, d3), and if `delta_ndim=-1`,
        then the transformation will broadcast over each shape-(d1, d2, d3) sub-tensor
        in the gradient (of which there are d0).
        """
        if defaults is None:
            defaults = {}
        defaults["param_ndim"] = param_ndim

        super().__init__(params, defaults)  # type: ignore
        self.inner_opt = InnerOpt(self.param_groups, **inner_opt_kwargs)

        for group in self.param_groups:
            param_ndim = group["param_ndim"]
            if param_ndim is not None and not isinstance(param_ndim, int):
                raise TypeError(
                    f"`param_ndim` must be an int or None, got: {param_ndim}"
                )

            for p in group["params"]:
                p: Tensor
                if param_ndim is None:
                    continue
                if p.ndim < abs(param_ndim):
                    raise ValueError(
                        f"`param_ndim={param_ndim}` specified for parameter with "
                        f"ndim={p.ndim} is not valid. `abs(param_ndim) <= ndim` must "
                        f"hold."
                    )

    @abstractmethod
    def _inplace_grad_transform_(
        self, param: Tensor, optim_group: DatumParamGroup
    ) -> None:  # pragma: no cover
        """Applies a transformation, in place, on the gradient of the each parameter
        in the provided param group.

        This transform should *always* be designed to broadcast over the leading
        dimension of the parameters's gradient. That is, each gradient  should be
        assumed to have the shape-(N, d0, ...) and the transformation should be
        applied - in-place - to each shape-(d0, ...) sub-tensor.

        Prior to calling `_in_place_grad_transform`, `GradientTransformerOptimizer`
        will reshape each parameter and its gradient to have the appropriate shape,
        in accordance with the value specified for `param_ndim` such that the shape
        (d0, ...) contains `param_ndim` entries.

        In the case where `param_ndim=0`, the tensor shapes will be (N, 1)."""
        raise NotImplementedError()

    def _apply_gradient_transforms_(self):
        for group in self.param_groups:
            for p in group["params"]:
                p: Tensor
                orig_p = p
                if p.grad is None:
                    continue
                assert orig_p.grad is not None

                p = _to_batch(p, group["param_ndim"])

                self._inplace_grad_transform_(p, optim_group=group)

                if p.grad is None or not _shares_memory(orig_p.grad, p.grad):
                    raise ValueError(
                        f"`{type(self).__name__}._inplace_grad_transform_` did "
                        " not modify the gradient of the parameter in-place."
                        " \nNote that setting `p.grad` directly replaces the"
                        " tensor, rather than writing to the tensor."
                    )

    @torch.no_grad()
    def _create_gradient_transforming_closure(
        self, closure: Callable[[], _T]
    ) -> Callable[[], Optional[_T]]:
        def grad_transforming_closure():
            with torch.enable_grad():
                loss = closure()

            self._apply_gradient_transforms_()
            return loss

        return grad_transforming_closure

    @overload
    def step(self, closure: Callable[[], _T]) -> _T:  # pragma: no cover
        ...

    @overload
    def step(self) -> None:  # pragma: no cover
        ...

    @overload
    def step(
        self, closure: Optional[Callable[[], _T]] = None
    ) -> Optional[_T]:  # pragma: no cover
        ...

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            closure = self._create_gradient_transforming_closure(closure)
            loss = self.inner_opt.step(closure)
        else:
            self._apply_gradient_transforms_()
            self.inner_opt.step()
            loss = None
        loss = cast(Optional[Union[float, Tensor]], loss)
        return loss


class ProjectionMixin(metaclass=ABCMeta):
    """A mixin that adds a parameter-projection method to an optimizer.

    Calling `.project()` will apply the in-place projection on each parameter
    stored by the optimizer.

    Notes
    -----
    'param_ndim' must be included in the optimizer's param groups, which describes
    how the projection should be applied to each parameter (i.e., whether or not
    it is broadcasted).

        - A positive number determines the dimensionality of the tensor that the projection will act on.
        - A negative number indicates the 'offset' from the dimensionality of the tensor.
        - `None` means that the projection will be applied to the tensor without any broadcasting.

    Methods
    -------
    _project_parameter_
    """

    param_groups: Iterable[DatumParamGroup]

    @abstractmethod
    def _project_parameter_(
        self, param: Tensor, optim_group: Mapping[str, Any]
    ) -> None:  # pragma: no cover
        """Applies an in-place projection on each parameter in the given param group.

        This operation should *always* be designed to broadcast over the leading
        dimension of the tensor. That is, each parameter gradient should be assumed
        to have the shape-(N, d0, ...) and the transformation should be applied -
        in-place - to each shape-(d0, ...) sub-tensor."""
        raise NotImplementedError()

    @torch.no_grad()
    def project(self) -> None:
        """Project each parameter in-place."""
        for group in self.param_groups:
            param_ndim = group["param_ndim"]

            for p in group["params"]:
                p = _to_batch(p, param_ndim)
                self._project_parameter_(param=p, optim_group=group)
