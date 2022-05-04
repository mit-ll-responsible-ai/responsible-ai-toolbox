# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Optional, Union

import torch
from torch import Generator, Tensor, default_generator

from rai_toolbox._utils import to_batch as _to_batch
from rai_toolbox._utils import validate_param_ndim as _validate_param_ndim
from rai_toolbox._utils import value_check


@torch.no_grad()
def uniform_like_l1_n_ball_(
    x: Tensor,
    epsilon: float = 1,
    param_ndim: Union[int, None] = -1,
    generator: Generator = default_generator,
) -> None:
    r"""Uniform sampling of an :math:`\epsilon`-sized `n`-ball for :math:`L^1`-norm, where `n` is controlled by `x.shape` and `param_ndim`. The result overwrites `x` in-place.

    Parameters
    ----------
    x: Tensor, shape-(N, D, ...)
        The tensor from which to generate a new random tensor, i.e., returns a tensor of
        similar shape and on the same device.

        By default, each of the `N` shape-`(D, ...)` subtensors are initialized
        independently. See `param_ndim` to contol this behavior.

    epsilon : float, optional (default=1)
        Determines the radius of the ball.

    param_ndim : int | None, optional (default=-1)
        Determines the dimenionality of the subtensors that are sampled by this
        function.

        - A positive number determines the dimensionality of each subtensor to be drawn and packed into the shape-`(N, D, ...)` resulting tensor.
        - A negative number determines from the dimensionality of the subtensor in terms of the offset of `x.ndim`. E.g. -1 indicates that `x` is arranged in a batch-style, and that `N` independent shape-`(D, ...)` tensors will be sampled.
        - `None` means that a single tensor of shape-`(N, D, ...)` is sampled.

    Returns
    -------
    out: Tensor, shape-(N, D, ...)
        A new random tensor of the same shape and on the same device as the input.

    References
    ----------
    .. [1] Rauber et al., 2020, Foolbox Native: Fast adversarial attacks to benchmark the robustness of machine learning models in PyTorch, TensorFlow, and JAX https://doi.org/10.21105/joss.02607

    .. [2] https://mathoverflow.net/a/9188

    Examples
    --------
    >>> import torch as tr
    >>> from rai_toolbox.perturbations.init import uniform_like_l1_n_ball_

    Drawing two shape-`(3,)` tensors from an :math:`\epsilon=2` sized :math:`L^1` 3D-ball.

    >>> x = tr.zeros(2, 3)
    >>> uniform_like_l1_n_ball_(x, epsilon=2.0) # default: param_ndim=-1
    >>> x
    tensor([[0.3301, 0.8459, 0.7071],
            [0.3470, 0.5077, 0.0977]])
    >>> tr.linalg.norm(x, dim=1, ord=1) < 2.0
    tensor([True, True])

    Drawing one tensor shape-`(6,)` tensor from a :math:`\epsilon=2` sized :math:`L^1` 6D-ball, and
    storing it in `x` as a shape-`(2, 3)` tensor. We specify `param_ndim=2` (or
    `param_ndim=None`) to achieve this.

    >>> x = tr.zeros(2, 3)
    >>> uniform_like_l1_n_ball_(x, epsilon=2.0, param_ndim=2)
    >>> x
    tensor([[0.1413, 0.5270, 0.1570],
            [0.4817, 0.2760, 0.4076]])
    >>> tr.linalg.norm(x, ord=1) < 2.0
    tensor(True)
    """
    _validate_param_ndim(param_ndim=param_ndim, p=x)
    value_check("epsilon", epsilon, min_=0.0, incl_min=False)

    xflat = _to_batch(x, param_ndim=param_ndim).flatten(1)
    nbatch, ndim = xflat.shape
    u = xflat.new(nbatch, ndim).uniform_(generator=generator)
    v = u.sort(dim=1).values
    vp = torch.cat((xflat.new_zeros(nbatch, 1), v[:, : ndim - 1]), dim=1)
    assert v.shape == vp.shape
    z = v - vp
    sign = xflat.new(nbatch, ndim).uniform_().sign()
    x.copy_(epsilon * (sign * z).reshape(x.shape))


@torch.no_grad()
def uniform_like_l2_n_ball_(
    x: Tensor,
    epsilon: float = 1,
    param_ndim: Union[int, None] = -1,
    generator: Generator = default_generator,
) -> None:
    r"""Uniform sampling within an :math:`\epsilon`-sized `n`-ball for :math:`L^2`-norm, where `n`
    is controlled by `x.shape` and `param_ndim`. The result overwrites `x` in-place.

    Parameters
    ----------
    x: Tensor, shape-(N, D, ...)
        The tensor to generate a new random tensor from, i.e., returns a tensor of
        similar shape and on the same device.

        By default, each of the `N` shape-`(D, ...)` subtensors are initialized
        independently. See `param_ndim` to contol this behavior.

    epsilon : float, optional (default=1)
        Determines the radius of the ball.

    param_ndim : int | None, optional (default=-1)
        Determines the dimenionality of the subtensors that are sampled by this
        function.

        - A positive number determines the dimensionality of each subtensor to be drawn and packed into the shape-`(N, D, ...)` resulting tensor.
        - A negative number determines from the dimensionality of the subtensor in terms of the offset of `x.ndim`. E.g. -1 indicates that `x` is arranged in a batch-style, and that `N` independent shape-`(D, ...)` tensors will be sampled.
        - `None` means that a single tensor of shape-`(N, D, ...)` is sampled.

    generator : torch.Generator, optional (default=`torch.default_generator`)
        Controls the RNG source.

    Returns
    -------
    out: Tensor, shape-`(N, ...)`
        A new random tensor of the same shape and on the same device as the input.

    References
    ----------
    .. [1] Rauber et al., 2020, Foolbox Native: Fast adversarial attacks to benchmark the robustness of machine learning models in PyTorch, TensorFlow, and JAX https://doi.org/10.21105/joss.02607

    .. [2] Voelker et al., 2017, Efficiently sampling vectors and coordinates from the n-sphere and n-ball http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf

    .. [3] Roberts, Martin, 2020, How to generate uniformly random points on n-spheres and in n-balls http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/

    Examples
    --------
    >>> import torch as tr
    >>> from rai_toolbox.perturbations.init import uniform_like_l2_n_ball_

    Drawing two shape-`(3,)` tensors from an :math:`\epsilon=2`-sized :math:`L^2` 3D-ball.

    >>> x = tr.zeros(2, 3)
    >>> uniform_like_l2_n_ball_(x, epsilon=2.0, param_ndim=1)
    >>> x
    tensor([[0.3030, 1.4269, 0.3927],
            [1.4015, 0.4913, 1.3028]])
    >>> tr.linalg.norm(x, dim=1, ord=2) < 2.0
    tensor([True, True])

    Drawing one shape-`(6, )` tensor from a :math:`\epsilon=2`-sized :math:`L^2` 6D-ball, and
    storing it in `x` as a shape-`(2, 3)` tensor. We specify `param_ndim=2` (or
    `param_ndim=None`) to achieve this.

    >>> x = tr.zeros(2, 3)
    >>> uniform_like_l2_n_ball_(x, epsilon=2.0, param_ndim=2)
    >>> x
    tensor([[-0.6903, -0.8597,  0.0109],
            [ 0.0906, -0.2387, -0.3059]])
    >>> tr.linalg.norm(x, ord=2) < 2.0
    tensor(True)
    """
    _validate_param_ndim(param_ndim=param_ndim, p=x)
    value_check("epsilon", epsilon, min_=0.0, incl_min=False)

    xflat = _to_batch(x, param_ndim=param_ndim).flatten(1)
    nbatch, ndim = xflat.shape
    z = xflat.new(nbatch, ndim + 2).normal_(generator=generator)
    r = z.norm(p=2, dim=1, keepdim=True)  # type: ignore
    x.copy_(epsilon * (z / r)[:, :ndim].reshape(x.shape))


@torch.no_grad()
def uniform_like_linf_n_ball_(
    x: Tensor,
    epsilon: float = 1,
    param_ndim: Optional[int] = None,
    generator: Generator = default_generator,
) -> None:
    r"""Uniform sampling within an :math:`\epsilon`-sized `n`-ball for :math:`L^{\infty}`-norm.
    The result overwrites `x` in-place.

    Parameters
    ----------
    x: Tensor, shape-(N, D, ...)
        The tensor to generate a new random tensor from, i.e., returns a tensor of
        similar shape and on the same device.

    epsilon : float, optional (default=1)
        Determines the radius of the ball.

    param_ndim : Optional[int]
        Unused. Included for parity with other init functions.

    generator : torch.Generator, optional (default=`torch.default_generator`)
        Controls the RNG source.

    Returns
    -------
    out: Tensor, shape-(N, ...)
        A new random tensor of the same shape and on the same device as the input.

    Examples
    --------
    >>> import torch as tr
    >>> from rai_toolbox.perturbations.init import uniform_like_linf_n_ball_
    >>> x = tr.zeros(2, 3)
    >>> uniform_like_linf_n_ball_(x, epsilon=2.0)
    >>> x
    tensor([[ 1.7092, -1.8723, -0.0806],
            [-1.4680, -1.8782, -0.1998]])
    >>> x.abs() < 2.
    tensor([[True, True, True],
            [True, True, True]])
    """
    del param_ndim
    value_check("epsilon", epsilon, min_=0.0, incl_min=False)

    x.uniform_(-epsilon, epsilon, generator=generator)
