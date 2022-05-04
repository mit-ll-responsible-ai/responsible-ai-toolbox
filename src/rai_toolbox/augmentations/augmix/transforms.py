# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from numbers import Integral
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch as tr

from ._implementation import augment_and_mix

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2", np.ndarray, tr.Tensor, float)

__all__ = ["AugMix", "Fork", "augment_and_mix"]


def _check_non_neg_int(item, name):
    if not isinstance(item, (int, Integral)) or item < 0:
        raise ValueError(f"`{name}`` must be a non-negative integer. Got {item}")


class AugMix(tr.nn.Module):
    __doc__ = augment_and_mix.__doc__

    def __init__(
        self,
        process_fn: Callable[[_T1], _T2],
        augmentations: Sequence[Callable[[_T1], _T1]],
        *,
        num_aug_chains: int = 3,
        aug_chain_depth: Union[int, Tuple[int, int]] = (1, 4),
        beta_params: Union[float, Tuple[float, float]] = (1.0, 1.0),
        dirichlet_params: Union[float, Sequence[float]] = 1.0,
        augmentation_choice_probs: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        _check_non_neg_int(num_aug_chains, "num_aug_chains")

        if isinstance(aug_chain_depth, Sequence):
            assert len(aug_chain_depth) == 2, aug_chain_depth
            _check_non_neg_int(aug_chain_depth[0], name="aug_chain_depth[0]")
            _check_non_neg_int(aug_chain_depth[1], name="aug_chain_depth[1]")
            assert aug_chain_depth[0] <= aug_chain_depth[1], aug_chain_depth

        if isinstance(dirichlet_params, Sequence):
            assert len(dirichlet_params) == num_aug_chains

        self.process_fn = process_fn
        self.augmentations = augmentations
        self.num_aug_chains = num_aug_chains
        self.aug_chain_depth = aug_chain_depth
        self.beta_params = beta_params
        self.dirichlet_params = dirichlet_params
        self.augmentations = augmentations
        self.augmentation_choice_probs = augmentation_choice_probs

        if augmentation_choice_probs is not None and len(
            augmentation_choice_probs
        ) != len(augmentations):
            raise ValueError(
                f"`len(sample_probabilities)` ({len(augmentation_choice_probs)}) must match `len(augmentations)` ({len(augmentations)})"
            )

    def forward(self, datum):
        return augment_and_mix(
            datum=datum,
            process_fn=self.process_fn,
            augmentations=self.augmentations,
            num_aug_chains=self.num_aug_chains,
            aug_chain_depth=self.aug_chain_depth,
            beta_params=self.beta_params,
            dirichlet_params=self.dirichlet_params,
            augmentation_choice_probs=self.augmentation_choice_probs,
        )

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "(\naugmentations="
        for t in self.augmentations:
            format_string += "\n"
            format_string += f"    {t},"
        format_string += f"\nprocess_fn={self.process_fn},"
        format_string += f"\nnum_aug_chains={self.num_aug_chains},"
        format_string += f"\naug_chain_depth={self.aug_chain_depth},"
        format_string += f"\nbeta_params={self.beta_params},"
        format_string += f"\ndirichlet_params={self.dirichlet_params},"
        format_string += "\n)"

        return format_string


def _flat_repr(x) -> str:
    out = f"{x}".splitlines()
    if len(out) == 1:
        return out[0]
    else:
        return out[0] + "...)"


class Fork(tr.nn.Module):
    """
    Forks an input into an arbitrary number of transform-chains. This can
    be useful for doing consistency-loss workflows.

    Parameters
    ----------
    *forked_transforms: Callable[[Any], Any]
        One transform for each fork to create.

    Examples
    --------
    >>> from rai_toolbox.augmentations.augmix import Fork

    Here are some trivial examples:

    >>> two_fork = Fork(lambda x: x, lambda x: 2 * x)
    >>> two_fork(2)
    (2, 4)

    >>> three_fork = Fork(lambda x: x, lambda x: 2 * x, lambda x: 0 * x)
    >>> three_fork(-1.0)
    (-1.0, -2.0, -0.0)

    Here is a simplified version of the triple-processing used by the AugMix
    paper's consistency loss. It anticipates a PIL image and produces a triplet.

    >>> from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip
    >>> from rai_toolbox.augmentations.augmix import AugMix
    >>> augmix = AugMix(
    ...     augmentations=[RandomHorizontalFlip(), RandomVerticalFlip()],
    ...     process_fn=ToTensor(),
    ... )
    >>> Fork(augmix, augmix, ToTensor())
    Fork(
          - ToTensor() ->
    x --> - AugMix(...) ->
          - AugMix(...) ->
    )
    """

    def __init__(self, *forked_transforms: Callable[[Any], Any]):
        super().__init__()

        if not forked_transforms:
            raise ValueError("At least one transform must be passed")

        if not all(callable(t) for t in forked_transforms):
            raise TypeError(
                f"All forked transforms must be callable, got: {forked_transforms}"
            )

        self.forked_transforms = forked_transforms

    def forward(self, x) -> Tuple[Any, ...]:
        return tuple(f(x) for f in self.forked_transforms)

    def __repr__(self) -> str:
        out = "Fork(\n"

        num_forks = len(self.forked_transforms)

        for n, f in enumerate(self.forked_transforms):
            if num_forks // 2 == n:
                if num_forks % 2 == 1:
                    out += f"x --> - {_flat_repr(f)} ->\n"
                else:
                    out += "x -->\n"
                    out += f"      - {_flat_repr(f)} ->\n"
            else:
                out += f"      - {_flat_repr(f)} ->\n"

        return out + ")"
