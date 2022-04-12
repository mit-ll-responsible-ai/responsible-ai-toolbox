# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from numbers import Number
from typing import List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import torch as tr
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import ToPILImage, ToTensor

from ._fourier_basis import generate_fourier_bases

_T = TypeVar("_T", Tensor, Image)

_to_pil = ToPILImage()
_to_tensor = ToTensor()


class FourierPerturbation(tr.nn.Module):
    """
    Augments a shape-(C, H, W) image with a randomly-sampled Fourier-basis. The generated bases
    have values on [0.0. 1.0) and, by default, have a norm of 1.

    See A Fourier Perspective on Model Robustness in Computer Vision [1]_ for a detailed
    description of this augmentation and its implications.

    Notes
    -----
    All sources of randomness are derived from NumPy's global RNG. If you are using ``PyTorch < 1.9.0`` and
    ``num_workers > 0``, you must manually set NumPy's global seed in ``worker_init_fn``.

    For large images, this transform can generate many distinct bases and thus consume large amounts of memory.

    References
    ----------
    .. [1] https://arxiv.org/abs/1906.08988"""

    # number of random indices to be drawn at a given time
    _cache_size: int = 10_000

    def sample_basis(self) -> Tensor:
        """Returns shape-(H, W) basis tensor"""
        if not self._rand_index_cache:
            # It is slow to call into random.choice(..., p=...) individually.
            # Calling it once to draw many indices avoids overhead.
            self._rand_index_cache = np.random.choice(
                len(self.bases),
                size=self._cache_size,
                replace=True,
                p=self._sample_probs,
            ).tolist()

        index = self._rand_index_cache.pop()

        if isinstance(self._norm_bnds, tuple):
            norm = np.random.uniform(*self._norm_bnds, size=1).item()
            return self.bases[index] * norm
        else:
            # norm was already applied to all bases in __init__
            return self.bases[index]

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        *,
        norm_scale: Union[float, Tuple[float, float]] = 1.0,
        dtype: Union[str, tr.dtype] = "float32",
        rand_flip_per_channel: bool = False,
        freq_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        num_distinct_phases: int = 1,
    ):
        """
        Parameters
        ----------
        size : Union[int, Tuple[int, int]]
            The spatial-extent of the image that the basis will augment. E.g. for
            cifar10, this would be ``32`` or ``(32, 32)``.

        norm_scale : Union[float, Tuple[float, float]] (optional, default=1.0)
            The scaling factor applied to each basis, prior to augmenting the image.
            A tuple can be supplied to indicate the ``[low, high)`` bounds of a uniform
            distribution from which the norm will be drawn for each augmentation.

            By default, each basis has a norm of 1.

        dtype : Union[str, tr.dtype] (optional, default='float32')

        rand_flip_per_channel : bool (optional, default=False)
            If ``True`` the perturbation is applied to each image-channel with a randomly drawn sign.
            E.g. the basis may be subtracted from the image's red channel and added to its blue and
            green channels.

        freq_bounds : Tuple[Optional[float], Optional[float]]
            Specifies the range of frequencies – number of oscillations over the image's extent – that will
            be represented among the bases, as ``[low, high]``. E.g. specifying ``(1, 3)`` will only augment
            images with Fourier bases that oscillate between (inclusive) 1 to 3 times. Bounds of ``None``
            default to the lowest (excluding zero) and highest possible frequencies that can be sustained
            by the given image.

        num_distinct_phases : int (optional, default=1)
            Specifies the number of phase-shifts that will be included in generating each basis. E.g. specifying
            ``num_distinct_phases=3`` will increase the total number of bases that can be sampled from by 3x;
            this is achieved by using the phase-shifts of 0, pi/3, and 2*pi/3 to diversify the bases.
        """
        super().__init__()
        if isinstance(dtype, tr.dtype):
            dtype = str(dtype).split(".")[-1]

        self._rand_index_cache: List[int] = []
        self._low_norm = None
        self._hi_norm = None

        self.last_position: Optional[Tuple[int, int]] = None

        assert num_distinct_phases > 0
        assert isinstance(rand_flip_per_channel, bool)
        self.rand_flip_per_channel = rand_flip_per_channel

        if isinstance(size, (int, Number)):
            size = (int(size), int(size))

        if not isinstance(norm_scale, (float, Number)):
            lo, hi = norm_scale
            assert 0 <= lo <= hi, norm_scale
            self._norm_bnds = (lo, hi)
        else:
            self._norm_bnds = norm_scale

        lower, upper = freq_bounds

        if lower is None:
            lower = 0.0
        if upper is None:
            upper = np.inf

        # freq=0 is excluded
        lower = max(1e-6, lower)

        if upper < lower or lower < 0:
            raise ValueError(
                f"radii-bounds must provide ordered bounds within [0, inf), got {(lower, upper)}"
            )
        self.radii_bounds = (lower, upper)

        self.bases = []
        _positions = []
        for item in generate_fourier_bases(
            nrows=size[0], ncols=size[1], dtype=dtype, factor_2pi_phase_shift=0
        ):
            _positions.append(item.position)
            self.bases.append(item.basis)

        # fourier bases (without any phase shift)
        self.bases = np.array(self.bases)  # shape-(N, image-H, image-W)
        # Fourier-space positions associated with bases
        _positions = np.array(_positions)  # used to draw random key

        # filter bases that fall outside of annulus of frequencies
        _zero_centered = _positions - np.array(size[0]) / 2
        _radii = np.linalg.norm(_zero_centered, axis=1)
        in_bounds = np.logical_and(lower <= _radii, _radii <= upper)
        _radii = _radii[in_bounds]
        _positions = _positions[in_bounds]
        self.bases = self.bases[in_bounds]

        if not _radii.size:
            raise ValueError(
                f"The specified radii-bounds, {freq_bounds}, do not include any Fourier bases"
            )

        # Populate bases with additional phase-shifted versions.
        # Phase shiftes are sampled evenly on [0, 2*pi) according to num_distinct_phases
        self.bases = [self.bases]

        for n in range(1, num_distinct_phases):
            fill = np.zeros_like(self.bases[0])

            frac_2_pi = n / num_distinct_phases

            for idx, item in enumerate(
                generate_fourier_bases(
                    nrows=size[0],
                    ncols=size[1],
                    dtype=dtype,
                    row_col_coords=_positions,
                    factor_2pi_phase_shift=frac_2_pi,
                )
            ):
                fill[idx] = item.basis
            self.bases.append(fill)

        self.bases = tr.tensor(np.vstack(self.bases))

        if isinstance(self._norm_bnds, (float, Number)):
            # norm is constant; scale bases upfront
            self.bases = self._norm_bnds * self.bases

        # Sample probabilities are adjusted so that sampling occurs evenly
        # over basis-frequencies.
        self._sample_probs = np.concatenate([1 / _radii] * num_distinct_phases)
        self._sample_probs /= self._sample_probs.sum()

    def forward(self, img: _T) -> _T:
        """
        Parameters
        ----------
        img: Union[Image, Tensor], shape-(C, H, W)
            Tensors are expected to have shape-(C, H, W) and to be on the domain [0, 1].
            The post-perturbation tensor is clipped to [0, 1]. PIL images are scaled
            and transposed appropriately (255 -> 1) prior to augmentation.

        Returns
        -------
        perturbed_img: Union[Image, Tensor], shape-(C, H, W)
        """

        _was_pil = False
        if isinstance(img, Image):
            _was_pil = True
            img = _to_tensor(img)
        elif not isinstance(img, Tensor):
            raise TypeError(
                f"`FourierPerturbation` only supports PIL images and PyTorch tensors, got: {type(img)}"
            )

        img: Tensor
        # shape-(1, H, W)
        basis = self.sample_basis().type_as(img)[None]

        if self.rand_flip_per_channel:
            channel_flip = np.random.randint(0, 2, (img.shape[0],))
            channel_flip[channel_flip == 0] = -1
            basis = basis * channel_flip.reshape(-1, 1, 1)

        img = img + basis

        if _was_pil:
            img = tr.clip_(img, min=0.0, max=1.0)
            return cast(_T, _to_pil(img))

        return cast(_T, img)

    def __repr__(self) -> str:

        return (
            self.__class__.__name__
            + f"(norm_scale={self._norm_bnds}, rand_flip_per_channel={self.rand_flip_per_channel}, radii_bounds={self.radii_bounds})"
        )
