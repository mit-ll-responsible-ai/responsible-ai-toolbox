# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from functools import partial
from typing import Any, Dict, Optional, Union

import torch
from torch import Generator, Tensor, default_generator
from torch.optim import SGD
from typing_extensions import Final

from rai_toolbox._typing import Optimizer as Opt
from rai_toolbox._typing import OptimizerType, OptimParams, Partial
from rai_toolbox._utils import check_param_group_value, value_check

from .misc import TopQGradientOptimizer
from .optimizer import (
    REQUIRED,
    ChainedParamTransformingOptimizer,
    DatumParamGroup,
    ParamTransformingOptimizer,
)

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


class _LpNormOptimizer(ParamTransformingOptimizer):
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
        defaults: Optional[Dict[str, Any]] = None,
        grad_scale: float = 1.0,
        grad_bias: float = 0.0,
        div_by_zero_eps: float = _TINY,
        **kwargs,
    ):
        """
        Parameters
        ----------
        params : Sequence[Tensor] | Iterable[Mapping[str, Any]]
            Iterable of parameters or dicts defining parameter groups.

        InnerOpt : Type[Optimizer] | Partial[Optimizer], optional (default=`torch.nn.optim.SGD`)
            The optimizer that updates the parameters after their gradients have
            been transformed.

        param_ndim : Optional[int]
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

        div_by_zero_eps : float, optional (default=`torch.finfo(torch.float32).tiny`)
            A lower bound used to clamp the normalization factor to prevent div-by-zero.

        **inner_opt_kwargs : Any
            Named arguments used to initialize `InnerOpt`.
        """
        if not hasattr(self, "_p"):
            raise TypeError(f"{type(self).__name__} must have the attribute `_p` set.")
        else:
            if not isinstance(self.p, (int, float)):
                raise TypeError(
                    f"{type(self).__name__}.p must be an int or float, got {self.p}"
                )

        super().__init__(
            params,
            InnerOpt=InnerOpt,
            param_ndim=param_ndim,
            defaults=defaults,
            grad_scale=grad_scale,
            grad_bias=grad_bias,
            **kwargs,
        )

        self.div_by_zero_eps = value_check("div_by_zero_eps", div_by_zero_eps, min_=0.0)

    @property
    def p(self) -> Union[float, int]:
        return self._p

    def per_datum_norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.norm(x, p=self.p, dim=1)  # type: ignore

    def _pre_step_transform_(self, param: Tensor, **_unused_kwargs) -> None:
        if param.grad is None:  # pragma: no cover
            return

        g = param.grad.flatten(1)
        g_norm = self.per_datum_norm(g).view(-1, *([1] * (param.ndim - 1)))
        param.grad /= torch.clamp(g_norm, self.div_by_zero_eps, None)


class SignedGradientOptim(ParamTransformingOptimizer):
    r"""A gradient-tranforming optimizer that takes the elementwise sign
    of a parameter's gradient prior to using `InnerOp.step` to update the
    corresponding parameter.

    See Also
    --------
    L1NormedGradientOptim
    L2NormedGradientOptim
    ParamTransformingOptimizer
    """

    def __init__(
        self,
        params: OptimParams,
        InnerOpt: Union[Partial[Opt], OptimizerType] = SGD,
        *,
        grad_scale: float = 1.0,
        grad_bias: float = 0.0,
        defaults: Optional[Dict[str, Any]] = None,
        param_ndim: Optional[int] = None,
        **inner_opt_kwargs,
    ) -> None:
        r"""
        Parameters
        ----------
        params : Sequence[Tensor] | Iterable[Mapping[str, Any]]
            Iterable of parameters or dicts defining parameter groups.

        InnerOpt : Type[Optimizer] | Partial[Optimizer], optional (default=`torch.nn.optim.SGD`)
            The optimizer that updates the parameters after their gradients have
            been transformed.

        grad_scale : float, optional (default=1.0)
            Multiplies each gradient in-place after the in-place transformation is
            performed. This can be specified per param-group.

        grad_bias : float, optional (default=0.0)
            Added to each gradient in-place after the in-place transformation is
            performed. This can be specified per param-group.

        defaults : Optional[Dict[str, Any]]
            Specifies default parameters for all parameter groups.

        param_ndim : Optional[int]
            Controls how `_pre_step_transform_` is broadcast onto the gradient
            of a given parameter. This has no effect for `SignedGradientOptim`.

        **inner_opt_kwargs : Any
            Named arguments used to initialize `InnerOpt`.

        Examples
        --------
        Let's create use `SignedGradientOptim` along with a SGD-step with a
        learning rate of `1.0`.

        >>> import torch as tr
        >>> from rai_toolbox.optim import SignedGradientOptim

        Creating a parameter for our optimizer to update, and our optimizer.

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
        super().__init__(
            params,
            InnerOpt,
            param_ndim=param_ndim,
            grad_scale=grad_scale,
            grad_bias=grad_bias,
            defaults=defaults,
            **inner_opt_kwargs,
        )

    def _pre_step_transform_(self, param: Tensor, **_unused_kwargs) -> None:
        if param.grad is None:  # pragma: no cover
            return

        torch.sign(param.grad, out=param.grad)


class L1NormedGradientOptim(_LpNormOptimizer):
    r"""A gradient-tranforming optimizer that normalizes the gradient by
    its :math:`L^1`-norm prior to using `InnerOp.step` to update the
    corresponding parameter.

    See Also
    --------
    L2NormedGradientOptim
    SignedGradientOptim
    ParamTransformingOptimizer

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
    r"""A gradient-tranforming optimizer that normalizes the gradient by
    its :math:`L^2`-norm prior to using `InnerOp.step` to update the
    corresponding parameter.

    The transformation is applied to the gradient in accordance with `param_ndim`.

    See Also
    --------
    L1NormedGradientOptim
    SignedGradientOptim
    ParamTransformingOptimizer

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


class L2ProjectedOptim(L2NormedGradientOptim):
    r"""A gradient-tranforming optimizer that constrains the updated parameters
    to lie within an :math:`\epsilon`-sized ball in :math:`L^2` space centered on the
    origin.

    A step with this optimizer normalizes the gradient by its :math:`L^2`-norm
    prior to using `InnerOp.step` to update the corresponding parameter. Each parameter
    is then projected into the constraint set.

    The transformation/projection is applied to the gradient/parameter in accordance
    with `param_ndim`.

    See Also
    --------
    L2NormedGradientOptim
    LinfProjectedOptim
    ParamTransformingOptimizer
    """

    def __init__(
        self,
        params: OptimParams,
        InnerOpt: Union[Partial[Opt], OptimizerType] = SGD,
        *,
        epsilon: float = REQUIRED,
        param_ndim: Union[int, None] = -1,
        grad_scale: float = 1.0,
        grad_bias: float = 0.0,
        defaults: Optional[Dict[str, Any]] = None,
        div_by_zero_eps: float = _TINY,
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

        epsilon : float
            Specifies the size of the L2-space ball that all parameters will be
            projected into, post optimization step.

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

        div_by_zero_eps : float, optional (default=`torch.finfo(torch.float32).tiny`)
            A lower bound used to clamp the normalization factor to prevent div-by-zero.

        **inner_opt_kwargs : Any
            Named arguments used to initialize `InnerOpt`.

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

        if defaults is None:
            defaults = {}
        defaults.setdefault("epsilon", epsilon)

        super().__init__(
            params,
            InnerOpt=InnerOpt,
            defaults=defaults,
            param_ndim=param_ndim,
            grad_scale=grad_scale,
            grad_bias=grad_bias,
            div_by_zero_eps=div_by_zero_eps,
            **inner_opt_kwargs,
        )
        check_param_group_value("epsilon", self.param_groups, min_=0.0)

    def _post_step_transform_(self, param: Tensor, optim_group: _HasEpsilon) -> None:
        """Applies an in-place projection on the given parameter"""
        param.renorm_(p=self.p, dim=0, maxnorm=optim_group["epsilon"])


class LinfProjectedOptim(SignedGradientOptim):
    r"""A gradient-tranforming optimizer that constrains the updated parameter values to fall within :math:`[-\epsilon, \epsilon]`.

    A step with this optimizer takes the elementwise sign of a parameter's gradient
    prior to using `InnerOp.step` to update the corresponding parameter. The updated
    parameter is then clamped elementwise to :math:`[-\epsilon, \epsilon]`.

    See Also
    --------
    L2NormedGradientOptim
    LinfProjectedOptim
    ParamTransformingOptimizer
    """

    def __init__(
        self,
        params: OptimParams,
        InnerOpt: Union[Partial[Opt], OptimizerType] = SGD,
        *,
        epsilon: float = REQUIRED,
        param_ndim=None,
        grad_scale: float = 1.0,
        grad_bias: float = 0.0,
        defaults: Optional[Dict[str, Any]] = None,
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

        epsilon : float
            Specifies the size of the L2-space ball that all parameters will be
            projected into, post optimization step.

        param_ndim : Optional[int]
            Clamp is performed elementwise, and thus `param_ndim` need not be adjusted.

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
        if defaults is None:
            defaults = {}
        defaults.setdefault("epsilon", epsilon)

        super().__init__(
            params,
            InnerOpt=InnerOpt,
            defaults=defaults,
            param_ndim=param_ndim,
            grad_scale=grad_scale,
            grad_bias=grad_bias,
            **inner_opt_kwargs,
        )

        check_param_group_value("epsilon", self.param_groups, min_=0.0)

    def _post_step_transform_(self, param: Tensor, optim_group: _HasEpsilon) -> None:
        epsilon = optim_group["epsilon"]
        param.clamp_(min=-epsilon, max=epsilon)


class L1qNormedGradientOptim(ChainedParamTransformingOptimizer):
    r"""A gradient-transforming optimizer that sparsifies a parameter's gradient and
    normalizes the gradient to have an :math:`L^1`-norm of `grad_scale`, prior to
    updating the parameter using `InnerOpt.step`.

    The sparsification process retains only the signs (i.e., :math:`\pm 1`) of the
    gradient's elements. The transformation is applied to the gradient in accordance
    with `param_ndim`.

    See Also
    --------
    L1NormedGradientOptim
    L2NormedGradientOptim
    TopQGradientOptimizer
    ParamTransformingOptimizer
    """

    def __init__(
        self,
        params: OptimParams,
        InnerOpt: Union[Partial[Opt], OptimizerType] = SGD,
        *,
        q: float = REQUIRED,
        dq: float = 0.0,
        param_ndim: Union[int, None] = -1,
        grad_scale: float = 1.0,
        grad_bias: float = 0.0,
        defaults: Optional[Dict[str, Any]] = None,
        div_by_zero_eps: float = _TINY,
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

        div_by_zero_eps : float, optional (default=`torch.finfo(torch.float32).tiny`)
            A lower bound used to clamp the normalization factor to prevent div-by-zero.

        generator : torch.Generator, optional (default=`torch.default_generator`)
            Controls the RNG source.

        **inner_opt_kwargs : Any
            Named arguments used to initialize `InnerOpt`.

        Examples
        --------
        Let's use `L1qNormedGradientOptim` along with a standard SGD-step with a learning
        rate of `1.0`. We'll sparsify the gradient to retain the top 70% elements of the
        tensor, and we'll normalize the sparse gradient to have a :math:`L^1`-norm of `1.8`.

        >>> import torch as tr
        >>> from rai_toolbox.optim import L1qNormedGradientOptim

        Creating a parameter for our optimizer to update, and our optimizer. We
        specify `param_ndim=None` so that the sparsification/normalization occurs on the
        gradient without any broadcasting.

        >>> x = tr.tensor([1.0, 1.0, 1.0], requires_grad=True)
        >>> optim = L1qNormedGradientOptim(
        ...     [x],
        ...     q=0.30,
        ...     grad_scale=1.8,
        ...     InnerOpt=tr.optim.SGD,
        ...     lr=1.0,
        ...     param_ndim=None,
        ... )

        Performing a simple calculation with `x` and performing backprop to create
        a gradient.

        >>> x.backward(gradient=tr.tensor([0.0, 1.0, 2.0]))
        >>> x.grad # the original gradient
        tensor([0., 1., 2.])

        Performing a step with our optimizer sparsifies and normalizes the gradient in-place, and then updates the parameter using `SGD([x], lr=1.0).step()`.

        >>> optim.step()
        >>> x.grad # the signed, sparsified, and normalized gradient
        tensor([0.0000, 0.9000, 0.9000])
        >>> x  # the updated parameter
        tensor([1.0000, 0.1000, 0.1000], requires_grad=True)
        """
        super().__init__(
            partial(TopQGradientOptimizer, q=q, dq=dq, generator=generator),
            SignedGradientOptim,
            partial(L1NormedGradientOptim, div_by_zero_eps=div_by_zero_eps),
            params=params,
            InnerOpt=InnerOpt,
            defaults=defaults,
            param_ndim=param_ndim,
            grad_scale=grad_scale,
            grad_bias=grad_bias,
            **inner_opt_kwargs,
        )
