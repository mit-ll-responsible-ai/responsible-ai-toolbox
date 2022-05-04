# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""
An implementation of the AugMix augmentation method specified in:

    Hendrycks, Dan, et al. "Augmix: A simple data processing method to improve
    robustness and uncertainty." arXiv preprint arXiv:1912.02781 (2019).

with reference implementation from https://github.com/google-research/augmix
"""
from numbers import Real
from typing import Callable, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

from rai_toolbox._typing import ArrayLike

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2", bound=ArrayLike)

C1 = Callable[[_T1], _T2]
C2 = Sequence[Callable[[_T1], _T1]]


def augment_and_mix(
    datum: _T1,
    *,
    process_fn: Callable[[_T1], _T2],
    augmentations: Sequence[Callable[[_T1], _T1]],
    num_aug_chains: int = 3,
    aug_chain_depth: Union[int, Tuple[int, int]] = (1, 4),
    beta_params: Union[float, Tuple[float, float]] = (1.0, 1.0),
    dirichlet_params: Union[float, Sequence[float]] = 1.0,
    augmentation_choice_probs: Optional[Sequence[float]] = None,
) -> _T2:
    """
    Augments `datum` using a mixture of randomly-composed augmentations via a method
    called "AugMix" [1]_.

    Parameters
    ----------
    datum : T1
       The datum to be augmixed.

    num_aug_chains : int
        The number of independent "augmentation chains" that are used to produce the augmixed
        result.

    aug_chain_depth : Union[int, Tuple[int, int]]
        Determines that number of augmentations that are randomly sampled (with replacement) and
        composed to form each augmentation chain.

        If a tuple of values are provided, these are used as the lower (inclusive) and
        upper (exclusive) bounds on a uniform integer-valued distribution from which the depth
        will be sampled for each augmentation chain.

        E.g.,

        - `mixture_depth=2` means that each augmentation chain will consist of two (randomly sampled) augmentations composed together.
        - `aug_chain_depth=(1, 4)` means depth of any given augmentation chain is uniformly sampled from [1, 4).

    process_fn : Callable[[T1], T2]
        The preprocessing function applied to the both clean and augmixed datums before
        they are combined. The return type of `process_fn` determines the return type
        of `augment_and_mix`.

    augmentations : Sequence[Callable[[T1], T1]]
        The collection of datum augmentations that is sampled from (with replacement) to form each augmentation
        chain.

    beta_params : Tuple[float, float]
        The Beta distribution parameters to draw `m`, which weights that convex combination::

                (1 - m) * img_process_fn(datum) + m * img_process_fn(augment(datum))

        If a single value is specified, it is used as both parameters for the distribution.

    dirichlet_params : Union[float, Sequence[float]]
        The Dirichlet distribution parameters used to weight the `mixture_width` number of augmentation chains.
        If a sequence is provided, its length must match `num_aug_chains`.

    augmentation_choice_probs : Optional[Sequence[float]]
        The probabilities associated with sampling each respective entry in `augmentations`.
        If not specified, a uniform distribution over all entries of `augmentation`.

    Returns
    -------
    augmixed: T2
        The augmixed datum.

    Notes
    -----
    The following depicts AugMix with N augmentation chains. Each `augchain(...)` consists of
    composed augmentations, where the composition depth is determined by `mixture_depth`::

        (1 - m) * process_fn(img) + m * (w1 * (process_fn ∘ augchain1)(img) + ... + wN * (process_fn ∘ augchainN)(img))

    with

    - m ~ Beta
    - [w1, ..., wN] ~ Dirichlet

    Random values are drawn via NumPy's global random number generator. Thus `numpy.random.seed`
    must be set in order to obtain reproducible results. Note that, until
    PyTorch 1.9.0, there was an issue with using NumPy's global RNG in conjunction with
    DataLoaders that used multiple workers, where identical seeds were being used
    across workers and the same seed was being set at the outset of each epoch.

    https://github.com/pytorch/pytorch/pull/56488

    References
    ----------
    .. [1] Hendrycks, Dan, et al. "Augmix: A simple data processing method to improve robustness and uncertainty." arXiv preprint arXiv:1912.02781 (2019).
    """
    # See: https://numpy.org/doc/stable/reference/random/index.html?#module-numpy.random
    # for details about new rng methods in numpy

    if isinstance(beta_params, Real):
        beta_params = (float(beta_params), float(beta_params))

    if isinstance(dirichlet_params, (Real, float)):
        dirichlet_params = [float(dirichlet_params)] * num_aug_chains
    elif len(dirichlet_params) != num_aug_chains:
        raise ValueError(
            f"The number of specified dirichlet parameters ({len(dirichlet_params)}) "
            f"does not match the mixture width ({num_aug_chains})"
        )

    if num_aug_chains:
        mix_weight = float(np.random.beta(*beta_params))
    else:
        # no augmentations occur
        mix_weight = 0.0

    if augmentation_choice_probs is not None:
        augmentation_choice_probs = np.array(augmentation_choice_probs, dtype=float)
        augmentation_choice_probs /= augmentation_choice_probs.sum()

    width_weights = np.asarray(np.random.dirichlet(dirichlet_params), dtype=np.float32)

    # mixture_depths[i]: number of augmentations applied to branch-i
    # of mixture graph
    if isinstance(aug_chain_depth, Sequence):
        mixture_depths: np.ndarray = np.random.randint(
            *aug_chain_depth, size=num_aug_chains
        )
    else:
        mixture_depths = np.full(width_weights.shape, aug_chain_depth)

    out = (1.0 - mix_weight) * process_fn(datum)

    for depth, weight in zip(mixture_depths, width_weights):
        # augmentations must not mutate original datum
        augmented_datum = datum
        for op in np.random.choice(
            augmentations, size=depth, replace=True, p=augmentation_choice_probs
        ):
            op: Callable[[_T1], _T1]
            augmented_datum = op(augmented_datum)
        out += (mix_weight * weight) * process_fn(augmented_datum)
    out: _T2
    return out
