# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import torch
from torch import Tensor


@torch.no_grad()
def uniform_like_l1_n_ball_(x: Tensor, epsilon: float = 1) -> None:
    """Uniform sampling of a n-ball for L_1-norm

    Parameters
    -----------
    x: Tensor, shape-(N, D, ...)
        The tensor to generate a new random tensor from, i.e., returns a tensor of similar
        shape and on the same device.

    Returns
    -------
    out: Tensor, shape-(N, D, ...)
        A new random tensor of the same shape and on the same device as the input.

    References
    ----------
    .. [rauber2017foolboxnative] Rauber et al., 2020, Foolbox Native: Fast adversarial attacks to
       benchmark the robustness of machine learning models in PyTorch, TensorFlow, and JAX
       https://doi.org/10.21105/joss.02607
    .. https://mathoverflow.net/a/9188
    """
    xflat = x.flatten(1)
    nbatch, ndim = xflat.shape
    u = xflat.new(nbatch, ndim).uniform_()
    v = u.sort(dim=1).values
    vp = torch.cat((xflat.new_zeros(nbatch, 1), v[:, : ndim - 1]), dim=1)
    assert v.shape == vp.shape
    z = v - vp
    sign = xflat.new(nbatch, ndim).uniform_().sign()
    x.copy_(epsilon * (sign * z).reshape(x.shape))


@torch.no_grad()
def uniform_like_l2_n_ball_(x: Tensor, epsilon: float = 1) -> None:
    """Uniform sampling of a n-ball for L_2-norm

    Parameters
    ----------
    x: Tensor, shape-(N, D, ...)
        The tensor to generate a new random tensor from, i.e., returns a tensor of similar shape and on the same device.

    Returns
    -------
    out: Tensor, shape-(N, ...)
        A new random tensor of the same shape and on the same device as the input.

    References
    ----------
    .. [rauber2017foolboxnative] Rauber et al., 2020, Foolbox Native: Fast adversarial attacks
       to benchmark the robustness of machine learning models in PyTorch, TensorFlow, and JAX
       https://doi.org/10.21105/joss.02607

    .. [#Voel17] Voelker et al., 2017, Efficiently sampling vectors and coordinates
       from the n-sphere and n-ball
       http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf

    .. [#Roberts20] Roberts, Martin, 2020, How to generate uniformly random points on n-spheres and in n-balls
       http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    """
    xflat = x.flatten(1)
    nbatch, ndim = xflat.shape
    z = xflat.new(nbatch, ndim + 2).normal_()
    r = z.norm(p=2, dim=1, keepdim=True)
    x.copy_(epsilon * (z / r)[:, :ndim].reshape(x.shape))


@torch.no_grad()
def uniform_like_linf_n_ball_(x: Tensor, epsilon: float = 1) -> None:
    x.copy_(epsilon * 2 * (torch.rand_like(x) - 0.5))
