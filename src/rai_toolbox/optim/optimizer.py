# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import inspect
from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
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
from typing_extensions import TypedDict

from rai_toolbox._typing import InstantiatesTo
from rai_toolbox._typing import Optimizer as Opt
from rai_toolbox._typing import OptimizerType, OptimParams, Partial, instantiates_to

_T = TypeVar("_T", bound=Optional[Union[Tensor, float]])

__all__ = ["GradientTransformerOptimizer", "ProjectionMixin"]


REQUIRED: Any = inspect.signature(SGD).parameters["lr"].default


class DatumParamGroup(TypedDict):
    params: List[Tensor]
    param_ndim: Optional[int]
    grad_scale: float
    grad_bias: float


def _shares_memory(x: Tensor, y: Tensor) -> bool:
    return x.storage().data_ptr() == y.storage().data_ptr()


def _reshape_to_batch(x: Tensor, param_ndim: Optional[int]) -> Tensor:
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


def _to_batch(p: Tensor, param_ndim: Optional[int]) -> Tensor:
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

    # atleast_2d needed for case where p was scalar
    vp = _reshape_to_batch(p, param_ndim=param_ndim)

    if p.grad is not None:
        vp.grad = _reshape_to_batch(p.grad, param_ndim=param_ndim)

    # vp (vp.grad) must be a view of p (p.grad). There is
    # not a simple way to assert this.

    # our views must be size-preserving
    assert torch.numel(vp) == torch.numel(p)
    return vp


class GradientTransformerOptimizer(Optimizer, metaclass=ABCMeta):
    r"""An optimizer that performs an in-place transformation to the
    gradient of each parameter, before performing the gradient-based
    update on each parameter::

       param = step(param, transformation_(param.grad), ...)

    Notes
    -----
    `GradientTransformerOptimizer` mirrors state with `InnerOpt` so that their
    `param_groups`, `defaults`, and `state` are always in sync.

    `GradientTransformerOptimizer` is designed to be combined with other,
    standard gradient-based optimizers (e.g. Adam) via encapsulation, rather
    then through inheritance. I.e., `GradientTransformerOptimizer(InnerOpt=<...>)`
    will apply a in-place gradient transform on a parameter, before using `InnerOpt.step(...)` to update said parameter.

    If a closure is supplied to the `.step(...)` method, then the in-place
    gradient transformation is applied after the closure call and prior to
    the parameter steps.

    Methods
    -------
    _inplace_grad_transform_
    """

    param_groups: List[DatumParamGroup]

    def __init__(
        self,
        params: Optional[OptimParams] = None,
        InnerOpt: Union[Opt, Partial[Opt], OptimizerType] = SGD,
        *,
        param_ndim: Union[int, None] = -1,
        grad_scale: float = 1.0,
        grad_bias: float = 0.0,
        defaults: Optional[Dict[str, Any]] = None,
        **inner_opt_kwargs,
    ) -> None:
        r"""
        Parameters
        ----------
        params : Sequence[Tensor] | Iterable[ParamGroup]
            iterable of parameters to optimize or dicts defining parameter groups

        InnerOpt : Type[Optimizer] | Partial[Optimizer], optional (default=`torch.nn.optim.SGD`)
            The optimizer that updates the parameters after their gradients have
            been transformed.

        param_ndim : int | None, optional (default=-1)
            Controls how `_inplace_grad_transform_` is broadcast onto the gradient
            of a given parameter. This can be specified per param-group. By default,
            the gradient transformation broadcasts over the first dimension in a
            batch-like style.

            - A positive number determines the dimensionality of the gradient that the transformation will act on.
            - A negative number indicates the 'offset' from the dimensionality of the gradient (see "Notes" for examples).
            - `None` means that the transformation will be applied directly to the gradient without any broadcasting.

        grad_scale : float, optional (default=1.0)
            Multiplies each gradient in-place after the in-place transformation is
            performed. This can be specified per param-group.

        grad_bias : float, optional (default=0.0)
            Added to each gradient in-place after the in-place transformation is
            performed. This can be specified per param-group.

        defaults : Optional[Dict[str, Any]]
            Specifies default parameters for all parameter groups.

        **inner_opt_kwargs : Any
            Named arguments used to initialize `InnerOpt`.

        Notes
        -----
        Additional Explanation of `param_ndim`:

        If the gradient has a shape `(d0, d1, d2)` and `param_ndim=1` then the
        transformation will be broadcast over each shape-(d2,) sub-tensor in the
        gradient (of which there are `d0 * d1`).

        If a gradient has a shape `(d0, d1, d2, d3)`, and if `param_ndim=-1`,
        then the transformation will broadcast over each shape-`(d1, d2, d3)`
        sub-tensor in the gradient (of which there are d0). This is equivalent
        to `param_ndim=3`.

        If `param_ndim=0` then the transformation is applied elementwise to the
        gradient by temporarily reshaping the gradient to a shape-(T, 1) tensor.

        Examples
        --------
        Let's create a gradient-transforming optimizer that replaces the gradient
        of each parameter with the elementwise sign of the gradient (:math:`\pm 1`) prior to
        performing the step of the inner optimizer:

        >>> import torch as tr
        >>> from rai_toolbox.optim import GradientTransformerOptimizer
        >>> class SignedGradientOptim(GradientTransformerOptimizer):
        ...
        ...     def _inplace_grad_transform_(self, param: tr.Tensor, **_kwds) -> None:
        ...         if param.grad is None:
        ...             return
        ...         tr.sign(param.grad, out=param.grad)  # operates in-place

        Now we'll use this optimizer – with `torch.nn.optim.AdamW` providing the actual
        parameter-update functionality – to update the parameter.

        >>> x = tr.tensor([-10.0, 10.0], requires_grad=True)
        >>> optim = SignedGradientOptim([x], InnerOpt=tr.optim.AdamW, lr=0.1)

        Using `x` in a calculation and compute an associated gradient for it:

        >>> (10_000 * x).sum().backward()

        Updating `x` using our grad-sign + AdamW optimizer:

        >>> optim.step()
        >>> x
        tensor([-10.9000,   8.9000], requires_grad=True)

        This was a simple optimizer which did not involve any broadcasting in the gradient
        transformation; the next example will involve broadcasting.

        **Controlling the gradient transformation with param_ndim**

        To understand the role of `param_ndim` let's design an optimizer that normalizes a
        gradient by its max value – along some user-specified dimension – prior to
        performing the gradient-based update to its parameter.

        >>> class MaxNormedGradientOptim(GradientTransformerOptimizer):
        ...
        ...     def _inplace_grad_transform_(self, param: tr.Tensor, **_kwds) -> None:
        ...         if param.grad is None:
        ...             return
        ...
        ...         g = param.grad.flatten(1) # (N, d1, ..., dm) -> (N, d1 * ... * dm)
        ...         max_norms = tr.max(g, dim=1).values
        ...         max_norms = max_norms.view(-1, *([1] * (param.ndim - 1)))  # reshape to have dimenionality-m
        ...         param.grad /= tr.clamp(max_norms, 1e-20, None)  # clamp to prevent div by 0

        Note that we design `_inplace_grad_transform_` to operate in-place on the gradient
        and that treat the gradient as if it has a shape `(N, d1, ..., dm)`, where we
        want to compute the max over each of the `N` sub-tensors of shape-`(d1, ..., dm)`.

        Critically, we did not use `param_ndim` at all in this method;
        `GradientTransformerOptimizer` assumes that we designed this method to broadcast in
        a batch-style, as we did, and automatically leverages `param_ndim` accordingly.

        Now we will create a shape-(2, 2) parameter to see how `MaxNormedGradientOptim`
        can compute the max-norm over various dimensions of the parameter. Let's print out
        the transformed gradient when we use each of `param_ndim`: `0`, `1`, or `2`.

        >>> x = tr.tensor([[1.0, 2.0],
        ...                [20.0, 10.0]], requires_grad=True)
        >>> for param_ndim in [0, 1, 2]:
        ...     optim = MaxNormedGradientOptim([x], param_ndim=param_ndim, InnerOpt=tr.optim.SGD, lr=0.0)
        ...
        ...     loss = (x * x).sum()
        ...     loss.backward()
        ...     optim.step()
        ...     print("param_ndim: {param_ndim}, normed grad:\n{x.grad}\n..")
        ...     optim.zero_grad()
        param_ndim: 0, normed grad:
        tensor([[1., 1.],
                [1., 1.]])
        ..
        param_ndim: 1, normed grad:
        tensor([[0.5000, 1.0000],
                [1.0000, 0.5000]])
        ..
        param_ndim: 2, normed grad:
        tensor([[0.0500, 0.1000],
                [1.0000, 0.5000]])

        See that `param_ndim=0` applies the max-norm elementwise, whereas `param_ndim=1`
        applied the max-norm to each 1D row of the gradient, and `param_ndim=2` applies
        the max-norm over the entire 2D gradient.
        """
        if defaults is None:
            defaults = {}
        defaults.setdefault("param_ndim", param_ndim)
        defaults.setdefault("grad_scale", grad_scale)
        defaults.setdefault("grad_bias", grad_bias)

        if instantiates_to(InnerOpt, Optimizer):
            if params is None:
                raise TypeError(
                    "`params` cannot be `None` when `InnerOpt` is an un-instantiated "
                    "optimizer type."
                )
            super().__init__(params, defaults)  # type: ignore
            self.inner_opt = InnerOpt(self.param_groups, **inner_opt_kwargs)  # type: ignore
        elif isinstance(InnerOpt, Optimizer):
            self.inner_opt = InnerOpt
            super().__init__(self.inner_opt.param_groups, defaults)
        else:
            raise TypeError(
                f"`InnerOpt` must be an Optimizer type or instance, got: {InnerOpt}"
            )

        # ensure inner-opt's defaults include those of `self`
        self.inner_opt.defaults.update(
            **{
                k: v
                for k, v in self.inner_opt.defaults.items()
                if k not in self.defaults
            },
            **self.defaults,
        )

        # state of `self` must mirror that of inner-opt
        self.__setstate__(self.inner_opt.__getstate__())  # type: ignore

        for group in self.param_groups:
            param_ndim = group["param_ndim"]
            if param_ndim is not None and not isinstance(param_ndim, int):
                raise TypeError(
                    f"`param_ndim` must be an int or None, got: {param_ndim}"
                )

            if not isinstance(group["grad_scale"], (float, int)):
                raise TypeError(
                    f"grad_scale must be a float, got {group['grad_scale']}"
                )

            if not isinstance(group["grad_bias"], (float, int)):
                raise TypeError(f"grad_bias must be a float, got {group['grad_bias']}")

            for p in group["params"]:
                p: Tensor
                if param_ndim is not None and p.ndim < abs(param_ndim):
                    raise ValueError(
                        f"`param_ndim={param_ndim}` specified for parameter "
                        f"with ndim={p.ndim} is not valid. `abs(param_ndim) <= "
                        f"ndim` must hold."
                    )

    def state_dict(self) -> dict:
        return self.inner_opt.state_dict()

    def __setstate__(self, state: dict):
        self.inner_opt.__setstate__(state)
        super().__setstate__(self.inner_opt.__getstate__())  # type: ignore

    def __getstate__(self) -> dict:
        return self.inner_opt.__getstate__()  # type: ignore

    def __repr__(self) -> str:
        return super().__repr__().replace("(", f"[{type(self.inner_opt).__name__}](", 1)

    @abstractmethod
    def _inplace_grad_transform_(
        self, param: Tensor, optim_group: DatumParamGroup
    ) -> None:  # pragma: no cover
        """Applies a transformation, in place, on the gradient of the each parameter
        in the provided param group.

        Parameters
        ----------
        param : torch.Tensor
            The parameter whose gradient (stored in `.grad`) will be modified in-place.

        optim_group : Dict[str, Any]
            The parameter group associated with `param`; contains per-parameter
            configured values that can affect the gradient transformation.

        Notes
        -----
        This transform should *always* be designed to broadcast over the leading
        dimension of the parameter's gradient. That is, each gradient  should be
        assumed to have the shape-(N, d0, ...) and the transformation should be
        applied - in-place - to each shape-(d0, ...) sub-tensor.

        Prior to calling `_in_place_grad_transform_`, `GradientTransformerOptimizer`
        will temporarily reshape each parameter and its gradient to have the appropriate
        shape – in accordance with the value specified for `param_ndim` – such that
        the shape (d0, ...) contains `|param_ndim|` entries.

        In the case where `param_ndim=0`, the transformation will be applied to a
        shape-(T, 1) tensor, where `T` corresponds to the total number of elements
        in the tensor."""
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
                assert p.grad is not None

                self._inplace_grad_transform_(p, optim_group=group)

                if group["grad_scale"] != 1.0:
                    p.grad *= group["grad_scale"]

                if group["grad_bias"] != 0.0:
                    p.grad += group["grad_bias"]

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
            loss = self.inner_opt.step(closure)  # type: ignore
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
    'param_ndim' can be included in the optimizer's param groups in order to describe
    how the projection should be applied to each parameter (i.e., whether or not
    it is broadcasted).

    - A positive number determines the dimensionality of the tensor that the projection will act on.
    - A negative number indicates the 'offset' from the dimensionality of the tensor.
    - `None` means that the projection will be applied to the tensor without any broadcasting.

    Methods
    -------
    _project_parameter_

    Examples
    --------
    Let's create a SGD-based optimizer that clamps each parameter's values so that
    they all fall within `[-1, 1]` after performing it's gradient-based step on the parameter.

    >>> import torch as tr
    >>> from rai_toolbox.optim import ProjectionMixin
    >>> class ClampedSGD(tr.optim.SGD, ProjectionMixin):
    ...     def _project_parameter_(self, param: tr.Tensor, optim_group: dict) -> None:
    ...         param.clamp_(min=-1.0, max=1.0)  # note: projection operates in-place
    ...
    ...     @tr.no_grad()
    ...     def step(self, closure=None):
    ...         loss = super().step(closure)
    ...         self.project()
    ...         return loss

    >>> x = tr.tensor([-0.1, 0.1], requires_grad=True)
    >>> optim = ClampedSGD([x], lr=1.0)
    >>> loss = (10_000 * x).sum()
    >>> loss.backward()
    >>> optim.step()  # parameters updated via SGD.step() and then clamped
    >>> x
    tensor([-1., -1.], requires_grad=True)

    Note that this is a particularly simple projection function, which acts
    elementwise on each parameter, and thus does not require us to include
    `param_ndim` in the optimizer's param-groups.
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
            param_ndim = group.get("param_ndim")

            for p in group["params"]:
                p = _to_batch(p, param_ndim)
                self._project_parameter_(param=p, optim_group=group)


class ChainedGradTransformerOptimizer(GradientTransformerOptimizer):
    """Chains together an arbitrary number of gradient-transforming optimizers,
    composing their transformation functions to modify the gradients of parameters
    in-place. `InnerOpt.step()` then updates each parameter using the transformed
    gradients.

    I.e. Passing `Opt1, Opt2, ..., OptN` to `ChainedGradTransformerOptimizer` will
    update a parameter's gradient using: `OptN.fn_(...(Opt2.fn_(Opt1.fn_(param.grad)))`,
    where `fn_` is a shorthand for `_inplace_grad_transform_`.

    Notes
    -----
    `ChainedGradTransformerOptimizer` mirrors state with `InnerOpt`, and with all of
    the user-specified chained gradient-trasnformers, so that their `param_groups`,
    `defaults`, and `state` are always in sync.
    """

    def __init__(
        self,
        *transforming_optimizers: InstantiatesTo[GradientTransformerOptimizer],
        params: Optional[OptimParams] = None,
        InnerOpt: Union[Opt, Partial[Opt], OptimizerType] = SGD,
        param_ndim: Union[int, None] = -1,
        grad_scale: float = 1,
        grad_bias: float = 0,
        defaults: Optional[Dict[str, Any]] = None,
        **inner_opt_kwargs,
    ) -> None:
        r"""
        Parameters
        ----------
        *transforming_optimizers: InstantiatesTo[GradientTransformerOptimizer],
            An arbitrary number of gradient-transforming optimizers, whose
            `_inplace_grad_transform_` methods will be composed from left to right –
            `Opt1, Opt2, ..., OptN -> fN_(...f2_(f1_(grad)))` – to update a parameter's
            gradient prior to updating the parameter.

        params : Optional[Sequence[Tensor] | Iterable[ParamGroup]]
            iterable of parameters to optimize or dicts defining parameter groups

        InnerOpt : Type[Optimizer] | Partial[Optimizer], optional (default=`torch.nn.optim.SGD`)
            The optimizer that updates the parameters after their gradients have
            been transformed.

        param_ndim : int | None, optional (default=-1)
            Controls how `_inplace_grad_transform_` is broadcast onto the gradient
            of a given parameter. This can be specified per param-group. By default,
            the gradient transformation broadcasts over the first dimension in a
            batch-like style.

            - A positive number determines the dimensionality of the gradient that the transformation will act on.
            - A negative number indicates the 'offset' from the dimensionality of the gradient (see "Notes" for examples).
            - `None` means that the transformation will be applied directly to the gradient without any broadcasting.

        grad_scale : float, optional (default=1.0)
            Multiplies each gradient in-place after the in-place transformation is
            performed. This can be specified per param-group.

        grad_bias : float, optional (default=0.0)
            Added to each gradient in-place after the in-place transformation is
            performed. This can be specified per param-group.

        defaults : Optional[Dict[str, Any]]
            Specifies default parameters for all parameter groups.

        **inner_opt_kwargs : Any
            Named arguments used to initialize `InnerOpt`.

        Notes
        -----
        Additional Explanation of `param_ndim`:

        If the gradient has a shape `(d0, d1, d2)` and `param_ndim=1` then the
        transformation will be broadcast over each shape-(d2,) sub-tensor in the
        gradient (of which there are `d0 * d1`).

        If a gradient has a shape `(d0, d1, d2, d3)`, and if `param_ndim=-1`,
        then the transformation will broadcast over each shape-`(d1, d2, d3)`
        sub-tensor in the gradient (of which there are d0). This is equivalent
        to `param_ndim=3`.

        If `param_ndim=0` then the transformation is applied elementwise to the
        gradient by temporarily reshaping the gradient to a shape-(T, 1) tensor.

        Examples
        --------
        **Basic Example**

        Let's chain together two gradient-transforming optimizers supplied by rAI-toolbox:
        `TopQGradientOptim` and `ClampedGradientOptimizer`

        >>> from rai_toolbox.optim import (
        ... ChainedGradTransformerOptimizer,
        ... ClampedGradientOptimizer,
        ... TopQGradientOptim,
        ... )
        >>> import torch as tr
        >>> from functools import partial

        >>> x1 = tr.ones(3, requires_grad=True)  # shape-(3,)

        Our optimizer will retain only the top-33rd percentile elements in the gradient:
        the smallest elements will be zero'd. Then the resulting gradient will be
        clamped so that its largest possible entry is `2.8`. Finally, the standard `SGD`
        optimizer will be used, with `lr=1.0`, to update the parameter(s) using the
        transformed gradients.

        We specify `TopQGradientOptim` and then `ClampedGradientOptimizer`; the
        transformations are applied in order from left to right. Providing per-optimizer
        defaults is achieved most naturally using :py:func:`functools.partial`.

        >>> optim = ChainedGradTransformerOptimizer(
        ...     partial(TopQGradientOptim, q=0.33),
        ...     partial(ClampedGradientOptimizer, clamp_max=2.8),
        ...     params=[x1],
        ...     lr=1.0,
        ...     param_ndim=None, # we don't want any broadcasting to occur
        ... )
        ClampedGradientOptimizer ○ TopQGradientOptim [SGD](
        Parameter Group 0
            clamp_max: 2.8
            clamp_min: None
            dampening: 0
            dq: 0.0
            grad_bias: 0
            grad_scale: 1
            lr: 1.0
            maximize: False
            momentum: 0
            nesterov: False
            param_ndim: None
            q: 0.33
            weight_decay: 0
        )

        Let's verify that `optim` transforms our gradients as-expected.

        >>> (tr.tensor([1.0, 2.0, 3.0]) * x1).sum().backward()
        >>> optim.step()
        >>> x1.grad  # element-0 should be zero'd by top-q; element-2 should be clamped to 2.8
        tensor([0.0000, 2.0000, 2.8000])

        See that `SGD([x1], lr=1.0).step()` is used to update our parameters; this can be
        controlled via the `InnerOpt` argument.

        >>> x1
        tensor([ 1.0000, -1.0000, -1.8000], requires_grad=True)

        **Adding Parameter Groups**

        Our chained gradient-transforming optimizers mirror their states with `optim` and
        `SGD`, thus we can add parameter groups and the group's settings will be applied to
        our chain as-expected.

        Let's add a 2D parameter, where we want to apply the top-q sparsification row-wise
        (via `param_ndim=1`), and retain only 64th-percentile gradient elements.

        >>> x2 = tr.ones(2, 3, requires_grad=True)  # shape-(2, 3)
        >>> optim.add_param_group(dict(params=x2, param_ndim=1, q=0.64))
        >>> optim
        ClampedGradientOptimizer ○ TopQGradientOptim [SGD](
        Parameter Group 0
            clamp_max: 2.8
            clamp_min: None
            dampening: 0
            dq: 0.0
            grad_bias: 0
            grad_scale: 1
            lr: 1.0
            maximize: False
            momentum: 0
            nesterov: False
            param_ndim: None
            q: 0.33
            weight_decay: 0
        Parameter Group 1
            clamp_max: 2.8
            clamp_min: None
            dampening: 0
            dq: 0.0
            grad_bias: 0
            grad_scale: 1
            lr: 1.0
            maximize: False
            momentum: 0
            nesterov: False
            param_ndim: 1
            q: 0.64

        >>> optim.zero_grad()
        >>> (tr.tensor([1.0, 2.0, 3.0]) * (x1 + x2)).sum().backward()
        >>> optim.step()
        >>> x1.grad
        tensor([0.0000, 2.8000, 2.8000])
        >>> x2.grad
        tensor([[0.0000, 0.0000, 2.8000],
            [0.0000, 0.0000, 2.8000]])
        """
        self._chain = ()

        super().__init__(
            params,
            InnerOpt,
            param_ndim=param_ndim,
            grad_scale=grad_scale,
            grad_bias=grad_bias,
            defaults=defaults,
            **inner_opt_kwargs,
        )
        for _opt in transforming_optimizers:
            if not instantiates_to(_opt, GradientTransformerOptimizer):
                raise TypeError(
                    f"*transforming_optimizers must contain `Type[GradientTransformerOptimizer]`, got: {transforming_optimizers}"
                )
        self._chain = tuple(
            opt(params, InnerOpt=self.inner_opt, defaults=self.defaults)
            for opt in transforming_optimizers
        )

    def _inplace_grad_transform_(
        self, param: Tensor, optim_group: DatumParamGroup
    ) -> None:
        # [f1, f2, f3] -> f3(f2(f1(grad)))
        for opt in self._chain:
            opt._inplace_grad_transform_(param=param, optim_group=optim_group)

    def __setstate__(self, state: dict):
        self.inner_opt.__setstate__(state)
        state = self.inner_opt.__getstate__()  # type: ignore
        for c in self._chain:
            c.__setstate__(state)
        super().__setstate__(state)

    def __repr__(self) -> str:
        return (
            super()
            .__repr__()
            .replace(
                type(self).__name__,
                " ○ ".join(type(c).__name__ for c in self._chain[::-1]),
            )
        )
