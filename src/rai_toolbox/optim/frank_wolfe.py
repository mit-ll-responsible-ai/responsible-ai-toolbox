# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Optional

import torch
from torch.optim import Optimizer

from rai_toolbox._typing import OptimParams

from .lp_space import L1qNormedGradientOptim, L2NormedGradientOptim, SignedGradientOptim
from .optimizer import GradientTransformerOptimizer

__all__ = [
    "FrankWolfe",
    "L1FrankWolfe",
    "L2FrankWolfe",
    "L1qFrankWolfe",
    "L1qNormedGradientOptim",
]


class FrankWolfe(Optimizer):
    r"""Implements the Frank-Wolfe minimization algorithm [1]_.

    .. math::
        w_{k+1} = (1 - l_r) w_k + l_r * s_k

    where :math:`s_k` is the linear minimization oracle (LMO).

    It is critical to note that this optimizer assumes that the `.grad` attribute
    of each parameter has been modified so as to store the *negative* of the LMO
    for that parameter, and not the gradient itself.


    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm#Algorithm"""

    def __init__(
        self,
        params: OptimParams,
        *,
        lr: float = 2.0,
        lmo_scaling_factor: float = 1.0,
        use_default_lr_schedule: bool = True,
        div_by_zero_eps: float = torch.finfo(torch.float32).tiny,
    ):
        r"""
        Parameters
        ----------
        params : Iterable
            Iterable of tensor parameters to optimize or dicts defining parameter groups.

        lr :  float
            Indicates the weight with which the LMO contributes to the parameter
            update. See ``use_default_lr_schedule`` for additional details. If
            ``use_default_lr_schedule=False`` then ``lr`` must be be in the
            domain [0, 1].

        lmo_scaling_factor : float, optional (default=1.0)
            A scaling factor applied to :math:`s_k` prior to each step.

        use_default_lr_schedule : bool, optional (default=True)
            If ``True``, then the per-parameter "learning rate" is scaled
            by :math:`\hat{l_r} = l_r / (l_r + k)` where k is the update index
            for that parameter.

        div_by_zero_eps : float


        """
        self._eps = div_by_zero_eps
        self._use_default_lr_schedule = use_default_lr_schedule

        if not self._use_default_lr_schedule and not (0 <= lr <= 1):
            raise ValueError("`lr` must reside in the domain [0, 1]")

        defaults = dict(lr=lr, lmo_scaling_factor=lmo_scaling_factor)

        super().__init__(params, defaults)  # type: ignore

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lmo_scale = group["lmo_scaling_factor"]
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    if not state:
                        state["step"] = 0

                    alpha = (
                        lr / max(self._eps, lr + state["step"])
                        if self._use_default_lr_schedule
                        else lr
                    )

                    # alpha weighting included in LMO (more efficient)
                    weighted_lmo = -(alpha * lmo_scale) * p.grad

                    p *= 1 - alpha
                    p += weighted_lmo

                    state["step"] += 1

        return loss


class L1qFrankWolfe(L1qNormedGradientOptim):
    def __init__(
        self,
        params: OptimParams,
        *,
        epsilon: float,
        q: float,
        pert_q: float,
        param_ndim: Optional[int] = -1,
        **inner_opt_kwargs,
    ):
        super().__init__(
            params,
            InnerOpt=FrankWolfe,
            epsilon=epsilon,
            q=q,
            pert_q=pert_q,
            param_ndim=param_ndim,
            **inner_opt_kwargs,
        )


class L1FrankWolfe(GradientTransformerOptimizer):
    """Performs Franke Wolfe optimization [1] using an epsilon-sized L1 ball as
    the constraint set.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm#Algorithm"""

    def __init__(
        self,
        params: OptimParams,
        *,
        epsilon: float,
        param_ndim: Optional[int] = -1,
        **inner_opt_kwargs,
    ):
        """
        Parameters
        ----------
        params : Iterable
            Iterable of tensor parameters to optimize or dicts defining parameter
            groups.

        epsilon : float
            The radius of the of the L1 ball. Can be specified per parameter-group.

        Notes
        -----
        The method ``L1FrankWolfe._inplace_grad_transform_`` is responsible for
        computing the *negative* LMO for a parameter and setting to ``param.grad``.
        """
        super().__init__(
            params,
            InnerOpt=FrankWolfe,
            lmo_scaling_factor=epsilon,
            param_ndim=param_ndim,
            **inner_opt_kwargs,
        )

    def _inplace_grad_transform_(self, param: torch.Tensor, **_unused_kwargs) -> None:
        if param.grad is None:  # pragma: no cover
            return
        # Computes the negative linear minimization oracle and sets it to
        # `param.grad`
        argmax = torch.argmax(torch.abs(param.grad), dim=1)
        signs = torch.sign(param.grad[torch.arange(len(param)), argmax])
        param.grad.mul_(0.0)
        param.grad[torch.arange(len(param)), argmax] = signs


class L2FrankWolfe(L2NormedGradientOptim):
    """Performs Franke Wolfe optimization [1] using an epsilon-sized L2 ball as
    the constraint set.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm#Algorithm"""

    def __init__(
        self,
        params: OptimParams,
        *,
        epsilon: float,
        param_ndim: Optional[int] = -1,
        **inner_opt_kwargs,
    ):
        """
        Parameters
        ----------
        params : Iterable
            Iterable of tensor parameters to optimize or dicts defining parameter
            groups.

        epsilon : float
            The radius of the of the L2 ball. Can be specified per parameter-group.

        Notes
        -----
        The method ``L2FrankWolfe._inplace_grad_transform_`` is responsible for
        computing the *negative* LMO for a parameter and setting to ``param.grad``.
        """
        super().__init__(
            params,
            InnerOpt=FrankWolfe,
            lmo_scaling_factor=epsilon,
            param_ndim=param_ndim,
            **inner_opt_kwargs,
        )


class LinfFrankWolfe(SignedGradientOptim):
    """Performs Franke Wolfe optimization [1] using an epsilon-sized L-inf ball as
    the constraint set.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm#Algorithm"""

    def __init__(
        self,
        params: OptimParams,
        *,
        epsilon: float,
        param_ndim: Optional[int] = -1,
        **inner_opt_kwargs,
    ):
        """
        Parameters
        ----------
        params : Iterable
            Iterable of tensor parameters to optimize or dicts defining parameter
            groups.

        epsilon : float
            The radius of the of the L-inf ball. Can be specified per parameter-group.

        Notes
        -----
        The method ``LinfFrankWolfe._inplace_grad_transform_`` is responsible for
        computing the *negative* LMO for a parameter and setting to ``param.grad``.
        """
        super().__init__(
            params,
            InnerOpt=FrankWolfe,
            lmo_scaling_factor=epsilon,
            param_ndim=param_ndim,
            **inner_opt_kwargs,
        )
