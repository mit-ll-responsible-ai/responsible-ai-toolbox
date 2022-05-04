# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from itertools import product
from typing import Iterable, Iterator, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import DTypeLike


class FourierBasis(NamedTuple):
    """
    Describes a 2D planewave generated via:

    A * cos(2pi * freq_vector @ x + phase_shift)

    Where `A` normalizes the 2D array to have an :math:`L^2`-norm of 1,
    and `x` is evaluated on a grid of positions on [0, H] x [0, W].

    Parameters
    ----------
    basis: np.ndarray, shape-(H, W)
        The amplitudes of the 2D planewave.

    position : Tuple[int, int]
        The (k-row, k-col) coordinate in shifted k-space, where
        the freq-0 component has been shifted to reside at the
        center of the shape-(H, W) "k-image".

    sym_position : Tuple[int, int]
        The k-space coordinate symmetric to `position`, about the
        center of the k-space image.

    phase_shift : float
        The total phase-shift applied to the cosine-generated
        planewave.
    """

    basis: np.ndarray
    position: Tuple[int, int]
    sym_position: Tuple[int, int]
    phase_shift: float


def generate_fourier_bases(
    nrows: int,
    ncols: int,
    dtype: DTypeLike = np.float32,
    *,
    factor_2pi_phase_shift: float = 0,
    row_col_coords: Optional[Iterable[Tuple[int, int]]] = None,
) -> Iterator[FourierBasis]:
    """Yields each unique, real-valued 2D array with unit norm, such that its Fourier
    transform is supported at each (i, j) and its symmetric position on a shape-(nrows, ncols)
    grid in Fourier space, where the lowest frequency component resides at the center of
    the grid.

    Parameters
    ----------
    nrows : int

    ncols : int

    dtype : DTypeLike

    factor_2pi_phase_shift : float (optional, default=0)
        E.g., specifying ``0.5`` will induce a phase-shift of :math:`\pi` on all bases.

    row_col_coords : Optional[Iterable[Tuple[int, int]]]
        If provided, specifies the specific Fourier-bases to generate in terms of their
        Fourier-space locations. Otherwise, all non-redundant entries on the nrows x ncols
        grip will be generated.

    Yields
    ------
    fourier_basis: FourierBasis
        Yields the array's position in Fourier space, (k_row, k_col), its symmetric position, and the
        shape-(nrows, ncols) real-valued basis array itself.

    Notes
    -----
    The iteration is performed in row-major order in 2D Fourier space, starting with the position (0, 0).

    Examples
    --------
    Here we will visualize the collection of bases in Fourier space that support the

    >>> import numpy as np
    >>> from rai_toolbox.augmentations.fourier import generate_fourier_bases
    >>> [np.round(out.basis, 2) for out in generate_fourier_bases(3, 3)]
    [array([[ 0.33,  0.12, -0.46],
        [ 0.12, -0.46,  0.33],
        [-0.46,  0.33,  0.12]]),
    array([[ 0.33,  0.33,  0.33],
            [ 0.12,  0.12,  0.12],
            [-0.46, -0.46, -0.46]]),
    array([[ 0.33, -0.46,  0.12],
            [ 0.12,  0.33, -0.46],
            [-0.46,  0.12,  0.33]]),
    array([[ 0.33,  0.12, -0.46],
            [ 0.33,  0.12, -0.46],
            [ 0.33,  0.12, -0.46]]),
    array([[0.33, 0.33, 0.33],
            [0.33, 0.33, 0.33],
            [0.33, 0.33, 0.33]])]

    We'll now inspect the corresponding support arrays in Fourier space.
    Note that only five basis arrays, not nine, are returned in association with the
    specified 3x3 grid. This is because the remaining four bases are redundant due
    to the symmetries in the support space.

    >>> f = lambda x: np.round(np.abs(np.fft.fftshift(np.fft.fft2(x))))
    >>> [f(out.basis) for out in generate_fourier_bases(3, 3)]
    [array([[2., 0., 0.],
            [0., 0., 0.],
            [0., 0., 2.]]),
    array([[0., 2., 0.],
            [0., 0., 0.],
            [0., 2., 0.]]),
    array([[0., 0., 2.],
            [0., 0., 0.],
            [2., 0., 0.]]),
    array([[0., 0., 0.],
            [2., 0., 2.],
            [0., 0., 0.]]),
    array([[0., 0., 0.],
            [0., 3., 0.],
            [0., 0., 0.]])]
    """
    pos = np.array(list(np.ndindex(nrows, ncols))).reshape(nrows, ncols, 2)
    shape = np.array((nrows, ncols))

    marker = np.zeros([nrows, ncols], dtype=np.uint8)

    if row_col_coords is None:
        row_col_coords = product(range(nrows), range(ncols))

    for i, j in row_col_coords:
        if marker[i, j] > 0:
            continue

        k = np.array([i, j])

        # Given a point, (kx, ky), in 2D Fourier space for real-valued data, compute
        # the symmetric point (kx', ky'). I.e. the Fourier coefficients at these points
        # are related by complex conjugation: c_{kx, ky} = c^*_{kx', ky'}
        sym_i, sym_j = np.where(
            np.mod(shape, 2) == 0, np.mod(shape - k, shape), shape - 1 - k
        )

        marker[i, j] = 1
        marker[sym_i, sym_j] = 1

        phase_shift = 1 / 4 * np.pi + (2 * np.pi * factor_2pi_phase_shift)

        # shift k so 0-freq resides at k=(Lx/2, Ly/2)
        k = k + np.ceil(shape / 2)

        # basis = A * np.cos((2 * np.pi) * (pos @ (k / shape)) + phase_shift)
        basis = pos @ (k / shape)
        basis *= 2 * np.pi
        basis += phase_shift
        np.cos(basis, out=basis)

        basis /= np.linalg.norm(basis, keepdims=True)

        # TODO: calculate frequency-vector to make it simple for users
        # to deduce per-image-length frequency of each basis
        #
        # This works for even-valued image-shapes:
        #
        # freq_vector=tuple(
        #     float(ele) if ele <= np.ceil(s / 2) else abs(s - ele)
        #     for ele, s in zip(k, shape)
        # )
        #

        yield FourierBasis(
            basis=basis.astype(dtype),
            position=(i, j),
            sym_position=(sym_i, sym_j),
            phase_shift=phase_shift,
        )
