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

from .optimizer import REQUIRED, DatumParamGroup, GradientTransformerOptimizer


class ClampedParamGroup(DatumParamGroup):
    clamp_min: Optional[float]
    clamp_max: Optional[float]


class ClampedGradientOptimizer(GradientTransformerOptimizer):
    """A gradient-tranforming  optimizer that clamps the elements of a gradient to
    fall within user-specified bounds, prior to using `InnerOp.step` to update the
    corresponding parameter."""

    param_groups: List[ClampedParamGroup]

    def __init__(
        self,
        params: Optional[OptimParams] = None,
        InnerOpt: Union[Opt, Partial[Opt], OptimizerType] = SGD,
        *,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
        defaults: Optional[Dict[str, Any]] = None,
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

        div_by_zero_eps : float, optional (default=`torch.finfo(torch.float32).tiny`)
            A lower bound used to clamp the normalization factor to prevent div-by-zero.

        **inner_opt_kwargs : Any
            Named arguments used to initialize `InnerOpt`.

        Examples"""
        if defaults is None:
            defaults = {}
        defaults.setdefault("clamp_min", clamp_min)
        defaults.setdefault("clamp_max", clamp_max)
        super().__init__(params, InnerOpt, defaults=defaults, **inner_opt_kwargs)

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

    def _inplace_grad_transform_(
        self, param: Tensor, optim_group: ClampedParamGroup
    ) -> None:
        if param.grad is None:
            return
        param.grad.clamp_(min=optim_group["clamp_min"], max=optim_group["clamp_max"])


class TopQGradientOptim(GradientTransformerOptimizer):
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

    def _inplace_grad_transform_(
        self, param: Tensor, optim_group: Dict[str, Any]
    ) -> None:

        if param.grad is None:  # pragma: no cover
            return
        q = optim_group["q"]
        dq = optim_group["dq"]

        _qlow = max(0.0, q - dq)
        _qhigh = min(1.0, q + dq)

        q = (
            (_qhigh - _qlow) * torch.rand(1, generator=self._generator) + _qlow
            if dq and (_qlow < q or _qhigh > q)
            else q
        )

        # Convert percent to number of entries
        shp = param.grad.shape
        nb = shp[0]

        num_pix = np.prod(shp[1:])
        num_q = 1.0 - q
        num_q = max(1, int(num_q * num_pix))

        g = param.grad.flatten(1)

        batch_idx = torch.tensor([[i] * num_q for i in range(nb)])

        _, corners_q = torch.topk(g.abs(), num_q, dim=1)
        s = torch.zeros_like(g)
        s[batch_idx, corners_q] = g[batch_idx, corners_q]

        param.grad[...] = s.view(shp)  # type: ignore
