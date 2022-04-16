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
    """param = step(param, sign(param.grad))"""

    def _inplace_grad_transform_(self, param: Tensor, **_unused_kwargs) -> None:
        if param.grad is None:  # pragma: no cover
            return

        torch.sign(param.grad, out=param.grad)


class L1NormedGradientOptim(_LpNormOptimizer):
    _p: Final = 1


class L2NormedGradientOptim(_LpNormOptimizer):
    _p: Final = 2


class L2ProjectedOptim(L2NormedGradientOptim, ProjectionMixin):
    def __init__(
        self,
        params: OptimParams,
        InnerOpt: Union[Partial[Opt], OptimizerType] = SGD,
        *,
        param_ndim: Optional[int] = -1,
        epsilon: float,
        **inner_opt_kwargs,
    ):

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

    def _project_parameter_(self, param: Tensor, optim_group: _HasEpsilon) -> None:
        """Applies an in-place projection on the given parameter"""
        param.renorm_(p=self.p, dim=0, maxnorm=optim_group["epsilon"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        self.project()
        return loss


class LinfProjectedOptim(SignedGradientOptim, ProjectionMixin):
    def __init__(
        self,
        params: OptimParams,
        InnerOpt: Union[Partial[Opt], OptimizerType] = SGD,
        *,
        epsilon: float,
        **inner_opt_kwargs,
    ):

        assert epsilon >= 0
        defaults = dict(epsilon=epsilon)

        super().__init__(
            params,
            InnerOpt=InnerOpt,
            defaults=defaults,
            param_ndim=None,
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
    r"""Sparse gradient step normalized by the :math:`\ell_1`-norm and with updated parameters constrained within an epsilon-sized :math:`\ell_1` ball about their
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
        params: Iterable
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
