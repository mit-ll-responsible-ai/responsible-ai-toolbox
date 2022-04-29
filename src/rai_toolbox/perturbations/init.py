# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Union

import torch
from torch import Generator, Tensor, default_generator

from rai_toolbox._utils import to_batch as _to_batch
from rai_toolbox._utils import validate_param_ndim as _validate_param_ndim


@torch.no_grad()
def uniform_like_l1_n_ball_(
    x: Tensor,
    epsilon: float = 1,
    param_ndim: Union[int, None] = -1,
    generator: Generator = default_generator,
) -> None:
    r"""Uniform sampling of a n-ball for :math:`L^1:-norm, where `n` is determined by
    `param_ndim`. The result overwrites `x` in-place.

    Parameters
    ----------
    x: Tensor, shape-(N, D, ...)
        The tensor to generate a new random tensor from, i.e., returns a tensor of
        similar shape and on the same device.

        By default, each of the N shape-(D, ...) subtensors are initialized
        intependently. See `param_ndim` to contol this behavior.

    epsilon : float, optional (default=1)
        Determines the radius of the ball.

    param_ndim : int | None, optional (default=-1)
        Determines the dimenionality of the subtensors that are sampled by this
        function.

        - A positive number determines the dimensionality of each subtensor to be drawn and packed into the shape-(N, D, ...) resulting tensor.
        - A negative number determines from the dimensionality of the subtensor in terms of the offset of `x.ndim`. E.g. -1 indicates that `x` is arranged in a batch-style, and that N independent shape-(D, ...) tensors will be sampled.
        - `None` means that a single tensor of shape-(N, D, ...) is sampled.

    Returns
    -------
    out: Tensor, shape-(N, D, ...)
        A new random tensor of the same shape and on the same device as the input.

    References
    ----------
    .. [1] Rauber et al., 2020, Foolbox Native: Fast adversarial attacks to benchmark the robustness of machine learning models in PyTorch, TensorFlow, and JAX https://doi.org/10.21105/joss.02607

    .. [2] https://mathoverflow.net/a/9188
    """
    _validate_param_ndim(param_ndim=param_ndim, p=x)

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
    r"""Uniform sampling of am epsilon-sized n-ball for :math:`L^2:-norm, where `n` is
    determined by `param_ndim`. The result overwrites `x` in-place.

    Parameters
    ----------
    x: Tensor, shape-(N, D, ...)
        The tensor to generate a new random tensor from, i.e., returns a tensor of
        similar shape and on the same device.

        By default, each of the N shape-(D, ...) subtensors are initialized
        intependently. See `param_ndim` to contol this behavior.

    epsilon : float, optional (default=1)
        Determines the radius of the ball.

    param_ndim : int | None, optional (default=-1)
        Determines the dimenionality of the subtensors that are sampled by this
        function.

        - A positive number determines the dimensionality of each subtensor to be drawn and packed into the shape-(N, D, ...) resulting tensor.
        - A negative number determines from the dimensionality of the subtensor in terms of the offset of `x.ndim`. E.g. -1 indicates that `x` is arranged in a batch-style, and that N independent shape-(D, ...) tensors will be sampled.
        - `None` means that a single tensor of shape-(N, D, ...) is sampled.

    generator : torch.Generator, optional (default=`torch.default_generator`)
        Controls the RNG source.

    Returns
    -------
    out: Tensor, shape-(N, ...)
        A new random tensor of the same shape and on the same device as the input.

    References
    ----------
    .. [1] Rauber et al., 2020, Foolbox Native: Fast adversarial attacks to benchmark the robustness of machine learning models in PyTorch, TensorFlow, and JAX https://doi.org/10.21105/joss.02607

    .. [2] Voelker et al., 2017, Efficiently sampling vectors and coordinates from the n-sphere and n-ball http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf

    .. [3] Roberts, Martin, 2020, How to generate uniformly random points on n-spheres and in n-balls http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    """
    xflat = _to_batch(x, param_ndim=param_ndim).flatten(1)
    nbatch, ndim = xflat.shape
    z = xflat.new(nbatch, ndim + 2).normal_(generator=generator)
    r = z.norm(p=2, dim=1, keepdim=True)  # type: ignore
    x.copy_(epsilon * (z / r)[:, :ndim].reshape(x.shape))


@torch.no_grad()
def uniform_like_linf_n_ball_(
    x: Tensor, epsilon: float = 1, generator: Generator = default_generator
) -> None:
    r"""Uniform sampling of an epsilon-sized n-ball for :math:`L^{\infty}:-norm. The
    result overwrites `x` in-place.

    Parameters
    ----------
    x: Tensor, shape-(N, D, ...)
        The tensor to generate a new random tensor from, i.e., returns a tensor of
        similar shape and on the same device.

    epsilon : float, optional (default=1)
        Determines the radius of the ball.

    generator : torch.Generator, optional (default=`torch.default_generator`)
        Controls the RNG source.

    Returns
    -------
    out: Tensor, shape-(N, ...)
        A new random tensor of the same shape and on the same device as the input.
    """
    x.uniform_(-epsilon, epsilon, generator=generator)
