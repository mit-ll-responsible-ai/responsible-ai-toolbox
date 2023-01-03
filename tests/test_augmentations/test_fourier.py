# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.testing import assert_allclose

from rai_toolbox.augmentations.fourier import generate_fourier_bases

from ._old_implementation import generate_fourier_bases as old_generate_bases


@given(
    nrows=st.integers(1, 15), ncols=st.integers(1, 15), phase_shift_factor=st.just(0.0)
)
def test_compare_implementations(nrows: int, ncols: int, phase_shift_factor: float):
    # note: there is an edge case in the old code that causes a discrepancy
    # for the phase-shift factors. Thus we do not include `phase_shift_factor`
    # int this test

    new_results = list(generate_fourier_bases(nrows, ncols))
    old_results = list(old_generate_bases(nrows, ncols))

    assert len(new_results) == len(old_results)

    for new, old in zip(new_results, old_results):
        assert new.position == old.position
        assert new.sym_position == old.sym_position
        assert_allclose(new.basis, old.basis, atol=1e-5, rtol=1e-5)


@given(
    nrows=st.integers(1, 15),
    ncols=st.integers(1, 15),
    phase_shift_factor=st.floats(0.0, 1.0),
)
def test_pi_phase_shift(nrows: int, ncols: int, phase_shift_factor: float):
    # Ensures that including a phase shift of pi is equivalent multiplying
    # the perturbations by -1.
    bases = np.stack(
        [
            x.basis
            for x in generate_fourier_bases(
                nrows, ncols, factor_2pi_phase_shift=phase_shift_factor
            )
        ]
    )

    assert_allclose(np.linalg.norm(bases, axis=(1, 2)), 1.0, rtol=1e-5)

    pi_shifted_bases = np.stack(
        [
            x.basis
            for x in generate_fourier_bases(
                nrows, ncols, factor_2pi_phase_shift=(phase_shift_factor + 0.5)
            )
        ]
    )

    assert_allclose(-bases, pi_shifted_bases, atol=1e-5, rtol=1e-5)

    two_pi_shifted_bases = np.stack(
        [
            x.basis
            for x in generate_fourier_bases(
                nrows, ncols, factor_2pi_phase_shift=(phase_shift_factor + 1.0)
            )
        ]
    )

    assert_allclose(bases, two_pi_shifted_bases, atol=1e-5, rtol=1e-5)
