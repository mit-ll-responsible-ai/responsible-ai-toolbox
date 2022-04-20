# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from random import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.optim import SGD
from typing_extensions import Final

from rai_toolbox._typing import Optimizer as Opt
from rai_toolbox._typing import OptimizerType, OptimParams, Partial

from .optimizer import DatumParamGroup, GradientTransformerOptimizer, ProjectionMixin

__all__ = [
    "L1NormedGradientOptim",
    "L2NormedGradientOptim",
    "SignedGradientOptim",
    "L2ProjectedOptim",
    "LinfProjectedOptim",
    "L1NormedGradientOptim",
    "L1qNormedGradientOptim",
]
_TINY = torch.finfo(torch.float32).tiny


class _HasEpsilon(DatumParamGroup):
    epsilon: float


class _LpNormOptimizer(GradientTransformerOptimizer):
    r"""A base optimizer whose step normalizes based on the p-norm:

    .. math::

       ||x||_p = (|x|_1^{p}, ...,  |x|_n^{p})^{1/p}

    When performing gradient-based updates to the optimizer's parameters,
    the gradients are normalized by the p-norm.
    """

    _p: Union[int, float]

    def __init__(
        self,
        params: OptimParams,
        InnerOpt: Union[Partial[Opt], OptimizerType] = SGD,
        *,
        param_ndim: Optional[int] = -1,
        div_by_zero_eps: float = _TINY,
        **kwargs,
    ):
        """
        Parameters
        ----------
        params : Sequence[Tensor] | Iterable[ParamGroup]
            iterable of parameters to optimize or dicts defining parameter groups

        InnerOpt : Type[Optimizer] | Partial[Optimizer], optional (default=`torch.nn.optim.SGD`)
            The optimizer that updates the parameters after their gradients have
            been transformed.

        param_ndim : Optional[int]
            Controls how `_inplace_grad_transform_` is broadcast onto the gradient
            of a given parameter. This can be specified per param-group. By default,
            the gradient transformation broadcasts over the first dimension in a
            batch-like style.

            - A positive number determines the dimensionality of the gradient that the transformation will act on.
            - A negative number indicates the 'offset' from the dimensionality of the gradient (see "Notes" for examples).
            - `None` means that the transformation will be applied directly to the gradient without any broadcasting.

        defaults : Optional[Dict[str, Any]]
            Specifies default parameters for all parameter groups.

        div_by_zero_eps : float

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
        """
        if not hasattr(self, "_p"):
            raise TypeError(f"{type(self).__name__} must have the attribute `_p` set.")
        else:
            if not isinstance(self.p, (int, float)):
                raise TypeError(
                    f"{type(self).__name__}.p must be an int or float, got {self.p}"
                )

        super().__init__(params, InnerOpt=InnerOpt, param_ndim=param_ndim, **kwargs)
        self.div_by_zero_eps = div_by_zero_eps

    @property
    def p(self) -> Union[float, int]:
        return self._p

    def per_datum_norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.norm(x, p=self.p, dim=1)  # type: ignore

    def _inplace_grad_transform_(self, param: Tensor, **_unused_kwargs) -> None:
        if param.grad is None:  # pragma: no cover
            return

        g = param.grad.flatten(1)
        g_norm = self.per_datum_norm(g).view(-1, *([1] * (param.ndim - 1)))
        param.grad /= torch.clamp(g_norm, self.div_by_zero_eps, None)


class SignedGradientOptim(GradientTransformerOptimizer):
    r"""A gradient-tranforming optimizer that takes the elementwise sign
    of a parameter's gradient prior to using `InnerOp.step` to update the
    corresponding parameter.

    See Also
    --------
    L1NormedGradientOptim
    L2NormedGradientOptim
    ProjectionMixin
    GradientTransformerOptimizer

    Examples
    --------
    Let's create use `SignedGradientOptim` along with a SGD-step with a
    learning rate of `1.0`.

    >>> import torch as tr
    >>> from rai_toolbox.optim import SignedGradientOptim

    Creating a parameter for our optimizer to update, and our optimizer. We
    want the norm to be computed over the entire gradient tensor – without
    broadcasting – so we specify `param_ndim=None`.

    >>> x = tr.tensor([-1.5, 1.5], requires_grad=True)
    >>> optim = SignedGradientOptim([x], InnerOpt=tr.optim.SGD, lr=1.0)

    Performing a simple calculation with `x` and performing backprop to create
    a gradient.

    >>> (tr.tensor([-2.0, 20.0]) * x).sum().backward()
    >>> x.grad # the original gradient
    tensor([-2., 20.])

    Performing a step with our optimizer transforms the gradient in-place, and then updates the parameter using `SGD([x], lr=1.0).step()`.

    >>> optim.step()
    >>> x.grad # the normalized gradient
    tensor([-1.,  1.])
    >>> x  # the updated parameter
    tensor([-0.5000,  0.5000], requires_grad=True)
    """

    def __init__(
        self,
        params: OptimParams,
        InnerOpt: Union[Partial[Opt], OptimizerType] = SGD,
        *,
        param_ndim: Optional[int] = None,
        defaults: Optional[Dict[str, Any]] = None,
        **inner_opt_kwargs,
    ) -> None:
        """
        Parameters
        ----------
        params : Sequence[Tensor] | Iterable[ParamGroup]
            iterable of parameters to optimize or dicts defining parameter groups

        InnerOpt : Type[Optimizer] | Partial[Optimizer], optional (default=`torch.nn.optim.SGD`)
            The optimizer that updates the parameters after their gradients have
            been transformed.

        param_ndim : Optional[int]
            Controls how `_inplace_grad_transform_` is broadcast onto the gradient
            of a given parameter. This can be specified per param-group. By default,
            the gradient transformation broadcasts over the first dimension in a
            batch-like style.

            - A positive number determines the dimensionality of the gradient that the transformation will act on.
            - A negative number indicates the 'offset' from the dimensionality of the gradient (see "Notes" for examples).
            - `None` means that the transformation will be applied directly to the gradient without any broadcasting.

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
        """
        super().__init__(
            params,
            InnerOpt,
            param_ndim=param_ndim,
            defaults=defaults,
            **inner_opt_kwargs,
        )

    def _inplace_grad_transform_(self, param: Tensor, **_unused_kwargs) -> None:
        if param.grad is None:  # pragma: no cover
            return

        torch.sign(param.grad, out=param.grad)


class L1NormedGradientOptim(_LpNormOptimizer):
    r"""A gradient-tranforming  optimizer that normalizes the each gradient by
    its :math:`L^1`-norm prior to using `InnerOp.step` to update the
    corresponding parameter.

    See Also
    --------
    L2NormedGradientOptim
    SignedGradientOptim
    ProjectionMixin
    GradientTransformerOptimizer

    Examples
    --------
    Let's create an optimizer that normalizes all parameter gradients using
    their :math:`L^1`-norm, and then updates the parameters with a standard
    SGD-step with a learning rate of `1.0`.

    >>> import torch as tr
    >>> from rai_toolbox.optim import L1NormedGradientOptim

    Creating a parameter for our optimizer to update, and our optimizer. We
    want the norm to be computed over the entire gradient tensor – without
    broadcasting – so we specify `param_ndim=None`.

    >>> x = tr.tensor([-1.0, 1.0], requires_grad=True)
    >>> optim = L1NormedGradientOptim([x], param_ndim=None, InnerOpt=tr.optim.SGD, lr=1.0)

    Performing a simple calculation with `x` and performing backprop to create
    a gradient.

    >>> (tr.tensor([2.0, 2.0]) * x).sum().backward()
    >>> x.grad # the un-normed gradient
    tensor([2., 2.])

    Performing a step with our optimizer transforms the gradient in-place, and then updates the parameter using `SGD([x], lr=1.0).step()`.

    >>> optim.step()
    >>> x.grad # the normalized gradient
    tensor([0.5000, 0.5000])
    >>> x  # the updated parameter
    tensor([-1.5000,  0.5000], requires_grad=True)
    """
    _p: Final = 1


class L2NormedGradientOptim(_LpNormOptimizer):
    r"""A gradient-tranforming  optimizer that normalizes the each gradient by
    its :math:`L^2`-norm prior to using `InnerOp.step` to update the
    corresponding parameter.

    See Also
    --------
    L1NormedGradientOptim
    SignedGradientOptim
    ProjectionMixin
    GradientTransformerOptimizer

    Examples
    --------
    Let's create an optimizer that normalizes all parameter gradients using
    their :math:`L^2`-norm, and then updates the parameters with a standard
    SGD-step with a learning rate of `1.0`.

    >>> import torch as tr
    >>> from rai_toolbox.optim import L2NormedGradientOptim

    Creating a parameter for our optimizer to update, and our optimizer. We
    want the norm to be computed over the entire gradient tensor – without
    broadcasting – so we specify `param_ndim=None`.

    >>> x = tr.tensor([-1.0, 1.0], requires_grad=True)
    >>> optim = L2NormedGradientOptim([x], param_ndim=None, InnerOpt=tr.optim.SGD, lr=1.0)

    Performing a simple calculation with `x` and performing backprop to create
    a gradient.

    >>> (tr.tensor([2.0, 2.0]) * x).sum().backward()
    >>> x.grad # the un-normed gradient
    tensor([2., 2.])

    Performing a step with our optimizer transforms the gradient in-place, and then updates the parameter using `SGD([x], lr=1.0).step()`.

    >>> optim.step()
    >>> x.grad # the normalized gradient
    tensor([0.7071, 0.7071])
    >>> x  # the updated parameter
    tensor([-1.7071,  0.2929], requires_grad=True)
    """
    _p: Final = 2


class L2ProjectedOptim(L2NormedGradientOptim, ProjectionMixin):
    r"""A gradient-tranforming optimizer that constrains the updated parameters
    to lie within an epsilon-sized ball in :math:`L^2` space centered on the origin.

    A step with this optimizer normalizes the each gradient by
    its :math:`L^2`-norm prior to using `InnerOp.step` to update the
    corresponding parameter. Each parameter is then projected into an epsilon-sized
    ball in :math:`L^2` space centered on the origin.

    See Also
    --------
    L2NormedGradientOptim
    LinfProjectedOptim
    ProjectionMixin
    GradientTransformerOptimizer

    Examples
    --------
    Let's create an optimizer that normalizes all parameter gradients using
    their :math:`L^2`-norm, and then updates the parameters with a standard
    SGD-step with a learning rate of `1.0`. After the step, each parameter
    will be projected into a :math:`L^2`-ball of radius `0.8`.

    >>> import torch as tr
    >>> from rai_toolbox.optim import L2ProjectedOptim

    Creating a parameter for our optimizer to update, and our optimizer. We
    want the norm to be computed over the entire gradient tensor – without
    broadcasting – so we specify `param_ndim=None`. This also controls the
    projection behavior.

    >>> x = tr.tensor([-1.0, 1.0], requires_grad=True)
    >>> optim = L2ProjectedOptim([x], param_ndim=None, InnerOpt=tr.optim.SGD, lr=1.0, epsilon=0.8)

    Performing a simple calculation with `x` and performing backprop to create
    a gradient.

    >>> (tr.tensor([2.0, 2.0]) * x).sum().backward()
    >>> x.grad # the un-normed gradient
    tensor([2., 2.])

    Performing a step with our optimizer transforms the gradient in-place, updates the
    parameter using `SGD([x], lr=1.0).step()`, and then projects the parameter into
    the constraint set.

    >>> optim.step()
    >>> x.grad # the normalized gradient
    tensor([0.7071, 0.7071])
    >>> x  # the updated parameter
    tensor([-0.7885,  0.1353], requires_grad=True)
    >>> x.norm(p=2).item() # `x` lies on the L2-ball of radius 0.8
    0.800000011920929
    """

    def __init__(
        self,
        params: OptimParams,
        InnerOpt: Union[Partial[Opt], OptimizerType] = SGD,
        *,
        epsilon: float,
        param_ndim: Union[int, None] = -1,
        defaults: Optional[Dict[str, Any]] = None,
        **inner_opt_kwargs,
    ):
        """
        Parameters
        ----------
        params : Sequence[Tensor] | Iterable[ParamGroup]
            iterable of parameters to optimize or dicts defining parameter groups

        InnerOpt : Type[Optimizer] | Partial[Optimizer], optional (default=`torch.nn.optim.SGD`)
            The optimizer that updates the parameters after their gradients have
            been transformed.

        epsilon : float
            Specifies the size of the L2-space ball that all parameters will be
            projected into, post optimization step.

        param_ndim : Union[int, None], optional (default=-1)
            Controls how `_inplace_grad_transform_` is broadcast onto the gradient
            of a given parameter. This can be specified per param-group. By default,
            the gradient transformation broadcasts over the first dimension in a
            batch-like style.

            - A positive number determines the dimensionality of the gradient that the transformation will act on.
            - A negative number indicates the 'offset' from the dimensionality of the gradient (see "Notes" for examples).
            - `None` means that the transformation will be applied directly to the gradient without any broadcasting.

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
        """
        assert epsilon >= 0
        self.div_by_zero_epsilon = epsilon

        if defaults is None:
            defaults = {}
        defaults["epsilon"] = epsilon

        super().__init__(
            params,
            InnerOpt=InnerOpt,
            defaults=defaults,
            param_ndim=param_ndim,
            **inner_opt_kwargs,
        )

    def _project_parameter_(self, param: Tensor, optim_group: _HasEpsilon) -> None:
        """Applies an in-place projection on the given parameter"""
        param.renorm_(p=self.p, dim=0, maxnorm=optim_group["epsilon"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        self.project()
        return loss


class LinfProjectedOptim(SignedGradientOptim, ProjectionMixin):
    r"""A gradient-tranforming optimizer that constrains the updated parameter values to fall within :math:`[-\epsilon, \epsilon]`.

    A step with this optimizer takes the elementwise sign of a parameter's gradient
    prior to using `InnerOp.step` to update the corresponding parameter. The updated
    parameter is then clamped elementwise to :math:`[-\epsilon, \epsilon]`.

    See Also
    --------
    L2NormedGradientOptim
    LinfProjectedOptim
    ProjectionMixin
    GradientTransformerOptimizer

    Examples
    --------
    Let's use `LinfProjectedOptim` along with a standard SGD-step with a learning rate
    of `1.0`. After the step, each parameter will have its values clamped to :math:`[-1.8, 1.8]`.

    >>> import torch as tr
    >>> from rai_toolbox.optim import L2ProjectedOptim

    Creating a parameter for our optimizer to update, and our optimizer. We
    specify `epsilon=1.8` so that the parameters are projected to the desired domain.

    >>> x = tr.tensor([-1.0, 0.5], requires_grad=True)
    >>> optim = LinfProjectedOptim([x], epsilon=1.8, InnerOpt=tr.optim.SGD, lr=1.0)

    Performing a simple calculation with `x` and performing backprop to create
    a gradient.

    >>> (tr.tensor([2.0, -2.0]) * x).sum().backward()
    >>> x.grad # the un-normed gradient
    tensor([2., -2.])

    Performing a step with our optimizer transforms the gradient in-place, updates the
    parameter using `SGD([x], lr=1.0).step()`, and then projects the parameter into
    the constraint set.

    >>> optim.step()
    >>> x.grad # the normalized gradient
    tensor([1.0, -1.0])
    >>> x  # the updated parameter
    tensor([-1.8000,  1.5000], requires_grad=True)
    """

    def __init__(
        self,
        params: OptimParams,
        InnerOpt: Union[Partial[Opt], OptimizerType] = SGD,
        *,
        param_ndim=None,
        epsilon: float,
        **inner_opt_kwargs,
    ):
        """
        Parameters
        ----------
        params : Sequence[Tensor] | Iterable[ParamGroup]
            iterable of parameters to optimize or dicts defining parameter groups

        InnerOpt : Type[Optimizer] | Partial[Optimizer], optional (default=`torch.nn.optim.SGD`)
            The optimizer that updates the parameters after their gradients have
            been transformed.

        epsilon : float
            Specifies the size of the L2-space ball that all parameters will be
            projected into, post optimization step.

        defaults : Optional[Dict[str, Any]]
            Specifies default parameters for all parameter groups.

        param_ndim : Optional[int]
            Clamp is performed elementwise, and thus `param_ndim` need not be adjusted.

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
        """
        assert epsilon >= 0
        defaults = dict(epsilon=epsilon)

        super().__init__(
            params,
            InnerOpt=InnerOpt,
            defaults=defaults,
            param_ndim=param_ndim,
            **inner_opt_kwargs,
        )

    def _project_parameter_(self, param: Tensor, optim_group: _HasEpsilon) -> None:
        epsilon = optim_group["epsilon"]
        param.clamp_(min=-epsilon, max=epsilon)

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        self.project()
        return loss


class L1qNormedGradientOptim(GradientTransformerOptimizer):
    r"""Sparse gradient step normalized by the :math:`L^1`-norm and with updated parameters constrained within an epsilon-sized :math:`L^1` ball about their
    original values.

    Given :math:`x` and :math:`\epsilon`, the constraint set is given by:

    .. math::

       S = \{x | \|x\|_1 \leq \epsilon\}
    """

    def __init__(
        self,
        params: OptimParams,
        InnerOpt: Union[Partial[Opt], OptimizerType] = SGD,
        *,
        param_ndim: Optional[int] = -1,
        epsilon: float,
        q: float,
        pert_q: float,
        div_by_zero_eps: float = _TINY,
        **inner_opt_kwargs,
    ):
        """
        Parameters
        ----------
        params: Sequence[Tensor] | Iterable[ParamGroup]
            iterable of parameters to optimize or dicts defining parameter groups

        InnerOpt: Type[Optimizer]
            The optimizer to update parameters

        epsilon:  float
            The Linf constraint
        """
        assert epsilon >= 0
        self.div_by_zero_epsilon = epsilon
        defaults = dict(epsilon=epsilon)
        super().__init__(
            params,
            InnerOpt=InnerOpt,
            defaults=defaults,
            param_ndim=param_ndim,
            **inner_opt_kwargs,
        )

        self.div_by_zero_eps = div_by_zero_eps
        self.q = q
        self.q0, self.q1 = q, q
        if pert_q:
            self.q0 = max(0.0, q - pert_q)
            self.q1 = min(1.0, q + pert_q)

    def _inplace_grad_transform_(
        self, param: Tensor, optim_group: Dict[str, Any]
    ) -> None:
        if param.grad is None:  # pragma: no cover
            return

        epsilon = optim_group["epsilon"]
        q = self.q
        if self.q0 < q or self.q1 > q:
            # TODO: refactor RNG
            q = (self.q1 - self.q0) * random() + self.q0

        # Convert percent to number of pixels
        shp = param.grad.shape
        nb = shp[0]

        num_pix = np.prod(shp[1:])
        num_q = 1.0 - q
        num_q = max(1, int(num_q * num_pix))

        g = param.grad.flatten(1)

        batch_idx = torch.tensor([[i] * num_q for i in range(nb)])

        _, corners_q = torch.topk(g.abs(), num_q, dim=1)
        s = torch.zeros_like(g)
        s[batch_idx, corners_q] = g.sign()[batch_idx, corners_q]

        s_norm = torch.norm(s, dim=1, p=1, keepdim=True)  # type: ignore
        s /= torch.clamp(s_norm, self.div_by_zero_eps, None)
        s *= epsilon

        param.grad[...] = s.view(shp)  # type: ignore
