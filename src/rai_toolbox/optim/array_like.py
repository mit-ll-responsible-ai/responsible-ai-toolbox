# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Any, Dict, Optional, Union

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
        if defaults is None:
            defaults = {}
        defaults.setdefault("clamp_min", clamp_min)
        defaults.setdefault("clamp_max", clamp_max)

        super().__init__(params, InnerOpt, defaults=defaults, **inner_opt_kwargs)

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
