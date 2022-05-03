# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Generator, Tensor, default_generator
from torch.optim import SGD

from rai_toolbox._typing import Optimizer as Opt
from rai_toolbox._typing import OptimizerType, OptimParams, Partial
from rai_toolbox._utils import check_param_group_value, value_check

from .optimizer import REQUIRED, DatumParamGroup, ParamTransformingOptimizer

__all__ = ["ClampedGradientOptimizer", "TopQGradientOptimizer"]


class ClampedParamGroup(DatumParamGroup):
    clamp_min: Optional[float]
    clamp_max: Optional[float]


class _ClampedOptim(ParamTransformingOptimizer):
    param_groups: List[ClampedParamGroup]

    def __init__(
        self,
        params: Optional[OptimParams] = None,
        InnerOpt: Union[Opt, Partial[Opt], OptimizerType] = SGD,
        *,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
        defaults: Optional[Dict[str, Any]] = None,
        param_ndim: Optional[int] = None,
        **inner_opt_kwargs,
    ) -> None:
        """
        Parameters
        ----------
        params : Sequence[Tensor] | Iterable[Mapping[str, Any]]
            Iterable of parameters or dicts defining parameter groups.

        InnerOpt : Type[Optimizer] | Partial[Optimizer], optional (default=`torch.nn.optim.SGD`)
            The optimizer that updates the parameters after their gradients have
            been transformed.

        epsilon : float
            Specifies the size of the L2-space ball that all parameters will be
            projected into, post optimization step.

        clamp_min: Optional[float]
            Lower-bound of the range to be clamped to.  Must be specified if `clamp_max` is `None`.

        clamp_max: Optional[float]
            Upper-bound of the range to be clamped to. Must be specified if `clamp_min`
            is `None`.

        grad_scale : float, optional (default=1.0)
            Multiplies each gradient in-place after the in-place transformation is
            performed. This can be specified per param-group.

        grad_bias : float, optional (default=0.0)
            Added to each gradient in-place after the in-place transformation is
            performed. This can be specified per param-group.

        defaults : Optional[Dict[str, Any]]
            Specifies default parameters for all parameter groups.

        param_ndim : Optional[int]
            Controles how `_pre_step_transform_` and `_post_step_transform_`  are
            broadcast onto a given parameter. This has no effect for
            `ClampedGradientOptimizer` and `ClampedParameterOptimizer`.

        **inner_opt_kwargs : Any
            Named arguments used to initialize `InnerOpt`.
        """
        if defaults is None:
            defaults = {}
        defaults.setdefault("clamp_min", clamp_min)
        defaults.setdefault("clamp_max", clamp_max)
        super().__init__(
            params,
            InnerOpt,
            defaults=defaults,
            param_ndim=param_ndim,
            **inner_opt_kwargs,
        )

        for group in self.param_groups:
            if group["clamp_min"] is None and group["clamp_max"] is None:
                raise ValueError("Either `clamp_min` or `clamp_max` must be specified")

            if group["clamp_min"] is not None and group["clamp_max"] is not None:
                value_check(
                    "clamp_min",
                    group["clamp_min"],
                    max_=group["clamp_max"],
                    upper_name="max_clamp",
                )


class ClampedGradientOptimizer(_ClampedOptim):
    """A gradient-tranforming  optimizer that clamps the elements of a gradient to
    fall within user-specified bounds **prior** to using `InnerOp.step` to update the
    corresponding parameter.

    See Also
    --------
    ClampedParameterOptimizer

    Examples
    --------
    Let's clamp each element of the parameter's gradient to `[-1, 3]` prior to
    performing a step with `SGD` using a learning rate of `1.0`.

    >>> import torch as tr
    >>> from rai_toolbox.optim import ClampedGradientOptimizer
    >>> x = tr.ones(2, requires_grad=True)
    >>> optim = ClampedGradientOptimizer(params=[x], lr=1.0, clamp_min=-1.0, clamp_max=3.0)

    >>> x.backward(gradient=tr.tensor([-0.5, 10]))
    >>> optim.step()
    >>> x.grad
    tensor([-0.5000,  3.0000])
    >>> x
    tensor([ 1.5000, -2.0000], requires_grad=True)"""

    def _pre_step_transform_(
        self, param: Tensor, optim_group: ClampedParamGroup
    ) -> None:
        if param.grad is None:  # pragma: no cover
            return
        param.grad.clamp_(min=optim_group["clamp_min"], max=optim_group["clamp_max"])


class ClampedParameterOptimizer(_ClampedOptim):
    """A parameter optimizer that clamps the elements of a parameter to fall within
    user-specified bounds **after** `InnerOp.step` has updated the parameter

    See Also
    --------
    ClampedGradientOptimizer

    Examples
    --------
    Let's perform a step with `SGD` using a learning rate of `1.0` to each of our parameters and then clamp their parameters to `[-1.0, 3.0]`.

    >>> import torch as tr
    >>> from rai_toolbox.optim import ClampedParameterOptimizer
    >>> x = tr.ones(2, requires_grad=True)
    >>> optim = ClampedParameterOptimizer(params=[x], lr=1.0, clamp_min=-1.0, clamp_max=3.0)

    >>> x.backward(gradient=tr.tensor([0.5, -10.0]))
    >>> optim.step()
    >>> x
    tensor([0.5000, 3.0000], requires_grad=True)"""

    def _post_step_transform_(
        self, param: Tensor, optim_group: ClampedParamGroup
    ) -> None:
        param.clamp_(min=optim_group["clamp_min"], max=optim_group["clamp_max"])


class TopQGradientOptimizer(ParamTransformingOptimizer):
    """A gradient-tranforming  optimizer that zeros the elements of a gradient whose
    absolue magnitudes fall below the Qth percentile. `InnerOp.step` is then to update
    the corresponding parameter.

    See Also
    --------
    L1qNormedGradientOptim
    ParamTransformingOptimizer"""

    def __init__(
        self,
        params: OptimParams,
        InnerOpt: Union[Partial[Opt], OptimizerType] = SGD,
        *,
        q: float = REQUIRED,
        dq: float = 0.0,
        param_ndim: Union[int, None] = -1,
        defaults: Optional[Dict[str, Any]] = None,
        generator: Generator = default_generator,
        **inner_opt_kwargs,
    ):

        r"""
        Parameters
        ----------
        params : Sequence[Tensor] | Iterable[Mapping[str, Any]]
            Iterable of parameters or dicts defining parameter groups.

        InnerOpt : Type[Optimizer] | Partial[Optimizer], optional (default=`torch.nn.optim.SGD`)
            The optimizer that updates the parameters after their gradients have
            been transformed.

        q : float
            Specifies the (fractional) percentile of absolute-largest gradient elements
            to retain when sparsifying the gradient. E.g., `q=0.9` means that only the
            gradient elements within the 90th-percentile will be retained.

            Must be within `[0.0, 1.0]`. The sparsification is applied to the gradient
            in accordance to `param_ndim`.

        dq : float, optional (default=0.0)
            If specified, the sparsity factor for each gradient transformation will
            be drawn from a uniform distribution over :math:`[q - dq, q + dq] \in [0.0, 1.0]`.

        param_ndim : Union[int, None], optional (default=-1)
            Determines how a parameter and its gradient is temporarily reshaped prior
            to being passed to both `_pre_step_transform_` and `_post_step_transform_`.
            By default,the transformation broadcasts over the tensor's first dimension
            in a batch-like style. This can be specified per param-group

            - A positive number determines the dimensionality of the tensor that the transformation will act on.
            - A negative number indicates the 'offset' from the dimensionality of the tensor (see "Notes" for examples).
            - `None` means that the transformation will be applied directly to the tensor without any broadcasting.

            See `ParamTransformingOptimizer` for more details and examples.

        grad_scale : float, optional (default=1.0)
            Multiplies each gradient in-place after the in-place transformation is
            performed. This can be specified per param-group.

        grad_bias : float, optional (default=0.0)
            Added to each gradient in-place after the in-place transformation is
            performed. This can be specified per param-group.

        defaults : Optional[Dict[str, Any]]
            Specifies default parameters for all parameter groups.

        generator : torch.Generator, optional (default=`torch.default_generator`)
            Controls the RNG source.

        **inner_opt_kwargs : Any
            Named arguments used to initialize `InnerOpt`.

        Examples
        --------
        Let's use `TopQGradientOptimizer` along with a standard SGD-step with a learning
        rate of `1.0`. We'll sparsify the gradient of a 2D parameter using varying
        percentile values. We set `param_ndim=None` so that no broadcasting occurs.

        >>> import torch as tr
        >>> from rai_toolbox.optim import TopQGradientOptimizer

        >>> gradient = tr.tensor([[0.5,   1.0],
        ...                       [-2.5, 0.30]])
        >>> for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ...     x = tr.ones((2, 2), requires_grad=True)
        ...     optim = TopQGradientOptimizer(params=[x], lr=1.0, q=q, param_ndim=None)
        ...     x.backward(gradient=gradient)
        ...     optim.step()
        ...     print(f"grad (q={q})\n{x.grad}\nx:\n{x}\n---")
        grad (q=0.0)
        tensor([[ 0.5000,  1.0000],
                [-2.5000,  0.3000]])
        x:
        tensor([[0.5000, 0.0000],
                [3.5000, 0.7000]], requires_grad=True)
        ---
        grad (q=0.25)
        tensor([[ 0.5000,  1.0000],
                [-2.5000,  0.0000]])
        x:
        tensor([[0.5000, 0.0000],
                [3.5000, 1.0000]], requires_grad=True)
        ---
        grad (q=0.5)
        tensor([[ 0.0000,  1.0000],
                [-2.5000,  0.0000]])
        x:
        tensor([[1.0000, 0.0000],
                [3.5000, 1.0000]], requires_grad=True)
        ---
        grad (q=0.75)
        tensor([[ 0.0000,  0.0000],
                [-2.5000,  0.0000]])
        x:
        tensor([[1.0000, 1.0000],
                [3.5000, 1.0000]], requires_grad=True)
        ---
        grad (q=1.0)
        tensor([[0., 0.],
                [0., 0.]])
        x:
        tensor([[1., 1.],
                [1., 1.]], requires_grad=True)
        ---

        We'll repeat this exercise using `param_ndim=1` so that the top-Q
        sparsification is applied to each row independently (i.e. it is "broadcast"
        over each 1D sub-tensor in our gradient).

        >>> gradient = tr.tensor([[0.5,   1.0],
        ...                       [-2.5, 0.30]])
        >>> for q in [0.0, 0.5, 1.0]:
        ...     x = tr.ones((2, 2), requires_grad=True)
        ...     optim = TopQGradientOptimizer(params=[x], lr=1.0, q=q, param_ndim=1)
        ...     x.backward(gradient=gradient)
        ...     optim.step()
        ...     print(f"grad (q={q})\n{x.grad}\nx:\n{x}\n---")
        grad (q=0.0)
        tensor([[ 0.5000,  1.0000],
                [-2.5000,  0.3000]])
        x:
        tensor([[0.5000, 0.0000],
                [3.5000, 0.7000]], requires_grad=True)
        ---
        grad (q=0.5)
        tensor([[ 0.0000,  1.0000],
                [-2.5000,  0.0000]])
        x:
        tensor([[1.0000, 0.0000],
                [3.5000, 1.0000]], requires_grad=True)
        ---
        grad (q=1.0)
        tensor([[0., 0.],
                [0., 0.]])
        x:
        tensor([[1., 1.],
                [1., 1.]], requires_grad=True)
        ---
        """
        if defaults is None:
            defaults = {}
        defaults.setdefault("q", q)
        defaults.setdefault("dq", dq)

        super().__init__(
            params,
            InnerOpt=InnerOpt,
            defaults=defaults,
            param_ndim=param_ndim,
            **inner_opt_kwargs,
        )
        check_param_group_value(
            "q", self.param_groups, optional=True, min_=0.0, max_=1.0
        )
        check_param_group_value(
            "dq", self.param_groups, optional=True, min_=0.0, max_=1.0
        )
        self._generator = value_check("generator", generator, type_=torch.Generator)

    def _pre_step_transform_(self, param: Tensor, optim_group: Dict[str, Any]) -> None:

        if param.grad is None:  # pragma: no cover
            return

        q = optim_group["q"]
        dq = optim_group["dq"]

        if dq > 0.0:
            _qlow = max(0.0, q - dq)
            _qhigh = min(1.0, q + dq)

            q = float(
                (_qhigh - _qlow) * torch.rand(1, generator=self._generator) + _qlow
                if dq and (_qlow < q or _qhigh > q)
                else q
            )

        # Convert percent to number of entries
        shp = param.grad.shape

        num_q = 1.0 - q
        num_q = round(num_q * np.prod(shp[1:]))

        g = param.grad.flatten(1)
        s = torch.zeros_like(g)

        if num_q:
            _, corners_q = torch.topk(g.abs(), num_q, dim=1)
            batch_idx = torch.tensor([[i] * num_q for i in range(shp[0])])
            s[batch_idx, corners_q] = g[batch_idx, corners_q]

        param.grad[...] = s.view(shp)  # type: ignore
