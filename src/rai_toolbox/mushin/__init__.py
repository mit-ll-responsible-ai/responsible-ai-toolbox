# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import functools as _fn

from ..perturbations.models import AdditivePerturbation
from ._lightning import BaseMushinModule, MetricsCallback
from ._perturbations import random_restart, solve_perturbation
from ._trainer import trainer
from ._utils import load_experiment, load_from_checkpoint

additive_perturbation = _fn.update_wrapper(
    _fn.partial(solve_perturbation, perturbation_model=AdditivePerturbation),
    solve_perturbation,
)


__all__ = [
    "trainer",
    "BaseMushinModule",
    "load_experiment",
    "load_from_checkpoint",
    "additive_perturbation",
    "random_restart",
    "solve_perturbation",
    "MetricsCallback",
]
