# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Union

import torch
from torch.optim import Optimizer

from rai_toolbox._typing import OptimParams
from rai_toolbox._utils import check_param_group_value, value_check

from .lp_space import L1qNormedGradientOptim, L2NormedGradientOptim, SignedGradientOptim
from .optimizer import ParamTransformingOptimizer

_TINY = torch.finfo(torch.float32).tiny

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

        lr :  float, optional (default=2.0)
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

        div_by_zero_eps : float, optional (default=`torch.finfo(torch.float32).tiny`)
            Prevents div-by-zero error in learning rate schedule.
        """
        self._eps = value_check("div_by_zero_eps", div_by_zero_eps, min_=0.0)
        self._use_default_lr_schedule = value_check(
            "use_default_lr_schedule", use_default_lr_schedule, type_=bool
        )

        defaults = dict(lr=lr, lmo_scaling_factor=lmo_scaling_factor)

        super().__init__(params, defaults)  # type: ignore

        check_param_group_value("lmo_scaling_factor", self.param_groups)

        check_param_group_value(
            "lr",
            self.param_groups,
            min_=0.0,
            max_=1.0 if not self._use_default_lr_schedule else None,
        )

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
    r"""A Frank-Wolfe [1]_ optimizer that, when computing the LMO, sparsifies a
    parameter's gradient. Each updated parameter is constrained to fall within an
    :math:`\epsilon`-sized ball in :math:`L^1` space, centered on the origin.

    The sparsification process retains only the signs (i.e., :math:`\pm 1`) of the
    gradient's elements. The transformation is applied to the gradient in accordance
    with `param_ndim`.

    See Also
    --------
    FrankWolfe
    L1FrankWolfe
    L2FrankWolfe
    LinfFrankWolfe

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm#Algorithm
    """

    def __init__(
        self,
        params: OptimParams,
        *,
        q: float,
        epsilon: float,
        dq: float = 0.0,
        lr: float = 2.0,
        use_default_lr_schedule: bool = True,
        param_ndim: Union[int, None] = -1,
        div_by_zero_eps: float = _TINY,
        generator: torch.Generator = torch.default_generator,
    ):
        r"""
        Parameters
        ----------
        params : Sequence[Tensor] | Iterable[Mapping[str, Any]]
            Iterable of parameters or dicts defining parameter groups.

        q : float
            Specifies the (fractional) percentile of absolute-largest gradient elements
            to retain when sparsifying the gradient. E.g., `q=0.9`means that only the
            gradient elements within the 90th-percentile will be retained.

            Must be within `[0.0, 1.0]`. The sparsification is applied to the gradient
            in accordance to `param_ndim`.

        epsilon : float
            Specifies the size of the L1-space ball that all parameters will be
            projected into, post optimization step.

        dq : float, optional (default=0.0)
            If specified, the sparsity factor for each gradient transformation will
            be drawn from a uniform distribution over :math:`[q - dq, q + dq] \in [0.0, 1.0]`.

        lr :  float, optional (default=2.0)
            Indicates the weight with which the LMO contributes to the parameter
            update. See `use_default_lr_schedule` for additional details. If
            `use_default_lr_schedule=False` then `lr` must be be in the
            domain `[0, 1]`.

        use_default_lr_schedule : bool, optional (default=True)
            If `True`, then the per-parameter "learning rate" is scaled
            by :math:`\hat{l_r} = l_r / (l_r + k)` where k is the update index
            for that parameter, which starts at 0.

        param_ndim : Union[int, None], optional (default=-1)
            Determines how a parameter and its gradient is temporarily reshaped prior
            to being passed to both `_pre_step_transform_` and `_post_step_transform_`.
            By default,the transformation broadcasts over the tensor's first dimension
            in a batch-like style. This can be specified per param-group

            - A positive number determines the dimensionality of the tensor that the transformation will act on.
            - A negative number indicates the 'offset' from the dimensionality of the tensor (see "Notes" for examples).
            - `None` means that the transformation will be applied directly to the tensor without any broadcasting.

            See `ParamTransformingOptimizer` for more details and examples.

        defaults : Optional[Dict[str, Any]]
            Specifies default parameters for all parameter groups.

        div_by_zero_eps : float, optional (default=`torch.finfo(torch.float32).tiny`)
            Prevents div-by-zero error in learning rate schedule.

        generator : torch.Generator, optional (default=`torch.default_generator`)
            Controls the RNG source.

        Examples
        --------
        Using `L1qFrankWolfe`, we'll sparsify the parameter's gradient to retain signs of
        the top 70% elements, and we'll constrain the updated parameter to fall within a
        :math:`L^1`-ball of radius `1.8`.

        >>> import torch as tr
        >>> from rai_toolbox.optim import L1qFrankWolfe

        Creating a parameter for our optimizer to update, and our optimizer. We
        specify `param_ndim=None` so that the sparsification/normalization occurs on the
        gradient without any broadcasting.

        >>> x = tr.tensor([1.0, 1.0, 1.0], requires_grad=True)
        >>> optim = L1qFrankWolfe(
        ...     [x],
        ...     q=0.30,
        ...     epsilon=1.8,
        ...     param_ndim=None,
        ... )

        Performing a simple calculation with `x` and performing backprop to create
        a gradient.

        >>> (tr.tensor([0.0, 1.0, 2.0]) * x).sum().backward()

        Performing a step with our optimizer uses the Frank-Wolfe algorithm to update
        its parameters; the resulting parameter was updated with a LMO based on a
        sparsified, sign-only gradient. Note that the parameter falls within/on the
        :math:`L^1`-ball of radius `1.8`.

        >>> optim.step()
        >>> x  # the updated parameter; has a L1-norm of 1.8
        tensor([ 0.0000, -0.9000, -0.9000], requires_grad=True)
        """
        super().__init__(
            params,
            InnerOpt=FrankWolfe,
            lmo_scaling_factor=epsilon,
            q=q,
            dq=dq,
            param_ndim=param_ndim,
            div_by_zero_eps=div_by_zero_eps,
            generator=generator,
            lr=lr,
            use_default_lr_schedule=use_default_lr_schedule,
        )


class L1FrankWolfe(ParamTransformingOptimizer):
    r"""A Frank-Wolfe [1]_ optimizer that constrains each updated parameter to fall
    within an :math:`\epsilon`-sized ball in :math:`L^1` space, centered on the origin.

    Notes
    -----
    The method `L1NormedGradientOptim._pre_step_transform_` is responsible for
    computing the *negative* linear minimization oracle for a parameter and storing it
    on `param.grad`.

    See Also
    --------
    FrankWolfe
    L2FrankWolfe
    LinfFrankWolfe
    L1qFrankWolfe

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm#Algorithm
    """

    def __init__(
        self,
        params: OptimParams,
        *,
        epsilon: float,
        lr: float = 2.0,
        use_default_lr_schedule: bool = True,
        param_ndim: Union[int, None] = -1,
        div_by_zero_eps: float = _TINY,
    ):
        r"""
        Parameters
        ----------
        params : Sequence[Tensor] | Iterable[Mapping[str, Any]]
            Iterable of parameters or dicts defining parameter groups.

        epsilon : float
            The radius of the of the L1 ball to which each updated parameter will be
            constrained. Can be specified per parameter-group.

        lr :  float, optional (default=2.0)
            Indicates the weight with which the LMO contributes to the parameter
            update. See `use_default_lr_schedule` for additional details. If
            `use_default_lr_schedule=False` then `lr` must be be in the
            domain `[0, 1]`.

        use_default_lr_schedule : bool, optional (default=True)
            If ``True``, then the per-parameter "learning rate" is scaled
            by :math:`\hat{l_r} = l_r / (l_r + k)` where k is the update index
            for that parameter, which starts at 0.

        param_ndim : Union[int, None], optional (default=-1)
            Determines how a parameter and its gradient is temporarily reshaped prior
            to being passed to both `_pre_step_transform_` and `_post_step_transform_`.
            By default,the transformation broadcasts over the tensor's first dimension
            in a batch-like style. This can be specified per param-group

            - A positive number determines the dimensionality of the tensor that the transformation will act on.
            - A negative number indicates the 'offset' from the dimensionality of the tensor (see "Notes" for examples).
            - `None` means that the transformation will be applied directly to the tensor without any broadcasting.

            See `ParamTransformingOptimizer` for more details and examples.

        div_by_zero_eps : float, optional (default=`torch.finfo(torch.float32).tiny`)
            Prevents div-by-zero error in learning rate schedule.

        Examples
        --------
        Using `L1FrankWolfe`, we'll constrain the updated parameter to fall within a
        :math:`L^1`-ball of radius `1.8`.

        >>> import torch as tr
        >>> from rai_toolbox.optim import L1FrankWolfe

        Creating a parameter for our optimizer to update, and our optimizer. We
        specify `param_ndim=None` so that the constrain occurs on the parameter without
        any broadcasting.

        >>> x = tr.tensor([1.0, 1.0], requires_grad=True)
        >>> optim = L1FrankWolfe([x], epsilon=1.8, param_ndim=None)

        Performing a simple calculation with `x` and performing backprop to create
        a gradient.

        >>> (tr.tensor([1.0, 2.0]) * x).sum().backward()

        Performing a step with our optimizer uses the Frank-Wolfe algorithm to update
        its parameters. Note that the updated parameter falls within/on the
        :math:`L^1`-ball of radius `1.8`.

        >>> optim.step()
        >>> x
        tensor([ 0.0000, -1.8000], requires_grad=True)
        """
        super().__init__(
            params,
            InnerOpt=FrankWolfe,
            lmo_scaling_factor=epsilon,
            lr=lr,
            use_default_lr_schedule=use_default_lr_schedule,
            param_ndim=param_ndim,
            div_by_zero_eps=div_by_zero_eps,
        )

    def _pre_step_transform_(self, param: torch.Tensor, **_unused_kwargs) -> None:
        if param.grad is None:  # pragma: no cover
            return
        # Computes the negative linear minimization oracle and sets it to
        # `param.grad`
        argmax = torch.argmax(torch.abs(param.grad), dim=1)
        signs = torch.sign(param.grad[torch.arange(len(param)), argmax])
        param.grad.mul_(0.0)
        param.grad[torch.arange(len(param)), argmax] = signs


class L2FrankWolfe(L2NormedGradientOptim):
    r"""A Frank-Wolfe [1]_ optimizer that constrains each updated parameter to fall
    within an :math:`\epsilon`-sized ball in :math:`L^2` space, centered on the origin.

    Notes
    -----
    The method `L2NormedGradientOptim._pre_step_transform_` is responsible for
    computing the *negative* linear minimization oracle for a parameter and storing it
    on `param.grad`.

    See Also
    --------
    FrankWolfe
    L1FrankWolfe
    LinfFrankWolfe
    L1qFrankWolfe

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm#Algorithm
    """

    def __init__(
        self,
        params: OptimParams,
        *,
        epsilon: float,
        lr: float = 2.0,
        use_default_lr_schedule: bool = True,
        param_ndim: Union[int, None] = -1,
        div_by_zero_eps: float = _TINY,
    ):
        r"""
        Parameters
        ----------
        params : Sequence[Tensor] | Iterable[Mapping[str, Any]]
            Iterable of parameters or dicts defining parameter groups.

        epsilon : float
            The radius of the of the L2 ball to which each updated parameter will be
            constrained. Can be specified per parameter-group.

        lr :  float, optional (default=2.0)
            Indicates the weight with which the LMO contributes to the parameter
            update. See `use_default_lr_schedule` for additional details. If
            `use_default_lr_schedule=False` then `lr` must be be in the
            domain `[0, 1]`.

        use_default_lr_schedule : bool, optional (default=True)
            If ``True``, then the per-parameter "learning rate" is scaled
            by :math:`\hat{l_r} = l_r / (l_r + k)` where k is the update index
            for that parameter, which starts at 0.

        param_ndim : Union[int, None], optional (default=-1)
            Determines how a parameter and its gradient is temporarily reshaped prior
            to being passed to both `_pre_step_transform_` and `_post_step_transform_`.
            By default,the transformation broadcasts over the tensor's first dimension
            in a batch-like style. This can be specified per param-group

            - A positive number determines the dimensionality of the tensor that the transformation will act on.
            - A negative number indicates the 'offset' from the dimensionality of the tensor (see "Notes" for examples).
            - `None` means that the transformation will be applied directly to the tensor without any broadcasting.

            See `ParamTransformingOptimizer` for more details and examples.

        div_by_zero_eps : float, optional (default=`torch.finfo(torch.float32).tiny`)
            Prevents div-by-zero error in learning rate schedule.

        Examples
        --------
        Using `L2FrankWolfe`, we'll constrain the updated parameter to fall within a
        :math:`L^2`-ball of radius `1.8`.

        >>> import torch as tr
        >>> from rai_toolbox.optim import L2FrankWolfe

        Creating a parameter for our optimizer to update, and our optimizer. We
        specify `param_ndim=None` so that the constrain occurs on the parameter without any
        broadcasting.

        >>> x = tr.tensor([1.0, 1.0], requires_grad=True)
        >>> optim = L2FrankWolfe([x], epsilon=1.8, param_ndim=None)

        Performing a simple calculation with `x` and performing backprop to create
        a gradient.

        >>> (tr.tensor([1.0, 2.0]) * x).sum().backward()

        Performing a step with our optimizer uses the Frank-Wolfe algorithm to update
        its parameters. Note that the updated parameter falls within/on the
        :math:`L^2`-ball of radius `1.8`.

        >>> optim.step()
        >>> x
        tensor([-0.8050, -1.6100], requires_grad=True)
        """
        super().__init__(
            params,
            InnerOpt=FrankWolfe,
            lmo_scaling_factor=epsilon,
            lr=lr,
            use_default_lr_schedule=use_default_lr_schedule,
            param_ndim=param_ndim,
            div_by_zero_eps=div_by_zero_eps,
        )


class LinfFrankWolfe(SignedGradientOptim):
    r"""A Frank-Wolfe [1]_ optimizer that constrains each updated parameter to fall
    within an :math:`\epsilon`-sized ball in :math:`L^\infty` space, centered on the origin.

    Notes
    -----
    The method `SignedGradientOptim._pre_step_transform_` is responsible for
    computing the *negative* linear minimization oracle for a parameter and storing it
    on `param.grad`.

    See Also
    --------
    FrankWolfe
    L1FrankWolfe
    L2FrankWolfe
    L1qFrankWolfe

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm#Algorithm

    Examples
    --------
    Using `LinfFrankWolfe`, we'll constrain the updated parameter to fall within a
    :math:`L^\infty`-ball of radius `1.8`.

    >>> import torch as tr
    >>> from rai_toolbox.optim import LinfFrankWolfe

    Creating a parameter for our optimizer to update, and our optimizer. We
    specify `param_ndim=None` so that the constrain occurs on the parameter without any
    broadcasting.

    >>> x = tr.tensor([1.0, 1.0], requires_grad=True)
    >>> optim = L2FrankWolfe([x], epsilon=1.8, param_ndim=None)

    Performing a simple calculation with `x` and performing backprop to create
    a gradient.

    >>> (tr.tensor([1.0, 2.0]) * x).sum().backward()

    Performing a step with our optimizer uses the Frank-Wolfe algorithm to update
    its parameters. Note that the updated parameter falls within/on the
    :math:`L^\infty`-ball of radius `1.8`.

    >>> optim.step()
    >>> x
    tensor([-1.8000, -1.8000], requires_grad=True)"""

    def __init__(
        self,
        params: OptimParams,
        *,
        epsilon: float,
        lr: float = 2.0,
        use_default_lr_schedule: bool = True,
        param_ndim: Union[int, None] = -1,
        div_by_zero_eps: float = _TINY,
    ):
        r"""
        Parameters
        ----------
        params : Sequence[Tensor] | Iterable[Mapping[str, Any]]
            Iterable of parameters or dicts defining parameter groups.

        epsilon : float
            The radius of the of the L-inf ball to which each updated parameter will be
            constrained. Can be specified per parameter-group.

        lr :  float, optional (default=2.0)
            Indicates the weight with which the LMO contributes to the parameter
            update. See `use_default_lr_schedule` for additional details. If
            `use_default_lr_schedule=False` then `lr` must be be in the
            domain `[0, 1]`.

        use_default_lr_schedule : bool, optional (default=True)
            If ``True``, then the per-parameter "learning rate" is scaled
            by :math:`\hat{l_r} = l_r / (l_r + k)` where k is the update index
            for that parameter, which starts at 0.

        param_ndim : Union[int, None], optional (default=-1)
            Determines how a parameter and its gradient is temporarily reshaped prior
            to being passed to both `_pre_step_transform_` and `_post_step_transform_`.
            By default,the transformation broadcasts over the tensor's first dimension
            in a batch-like style. This can be specified per param-group

            - A positive number determines the dimensionality of the tensor that the transformation will act on.
            - A negative number indicates the 'offset' from the dimensionality of the tensor (see "Notes" for examples).
            - `None` means that the transformation will be applied directly to the tensor without any broadcasting.

            See `ParamTransformingOptimizer` for more details and examples.

        div_by_zero_eps : float, optional (default=`torch.finfo(torch.float32).tiny`)
            Prevents div-by-zero error in learning rate schedule.
        """
        super().__init__(
            params,
            InnerOpt=FrankWolfe,
            lmo_scaling_factor=epsilon,
            lr=lr,
            use_default_lr_schedule=use_default_lr_schedule,
            param_ndim=param_ndim,
            div_by_zero_eps=div_by_zero_eps,
        )
