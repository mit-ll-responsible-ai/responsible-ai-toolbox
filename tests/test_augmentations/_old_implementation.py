# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from itertools import product
from typing import Iterable, Iterator, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import DTypeLike


class FourierBasis(NamedTuple):
    basis: np.ndarray
    position: Tuple[int, int]
    sym_position: Tuple[int, int]


def _get_symmetric_pos(*, dims: Tuple[int, int], pos: Tuple[int, int]) -> np.ndarray:
    x = np.array(dims)
    p = np.array(pos)
    return np.where(np.mod(x, 2) == 0, np.mod(x - p, x), x - 1 - p)


def generate_fourier_bases(
    nrows: int,
    ncols: int,
    dtype: DTypeLike = np.float32,
    *,
    factor_2pi_phase_shift: float = 0,
    row_col_coords: Optional[Iterable[Tuple[int, int]]] = None,
) -> Iterator[FourierBasis]:
    phase = np.exp(2j * np.pi * factor_2pi_phase_shift)
    # If Fourier basis at (i, j) is generated, set marker[i, j] = 1.
    marker = np.zeros([nrows, ncols], dtype=np.uint8)

    # this will be updated in place, but we will always re-zero
    freq = np.zeros([nrows, ncols], dtype=np.complex64)

    if row_col_coords is None:
        row_col_coords = product(range(nrows), range(ncols))

    for i, j in row_col_coords:
        if marker[i, j] > 0:
            continue

        sym_i, sym_j = _get_symmetric_pos(dims=(nrows, ncols), pos=(i, j))

        if (sym_i, sym_j) == (i, j):
            # bug! `phase` is not applied in these edge-cases
            freq[i, j] = 1.0
            marker[i, j] = 1
        else:
            freq[i, j] = (0.5 + 0.5j) * phase
            freq[sym_i, sym_j] = freq[i, j].conj()
            marker[i, j] = 1
            marker[sym_i, sym_j] = 1
        basis = np.fft.ifft2(np.fft.ifftshift(freq))
        # each basis has a (flattened) L2-norm of 1.0
        basis = np.sqrt(nrows * ncols) * np.real(basis)

        freq[i, j] = 0
        freq[sym_i, sym_j] = 0

        yield FourierBasis(
            basis=basis.astype(dtype), position=(i, j), sym_position=(sym_i, sym_j)
        )
