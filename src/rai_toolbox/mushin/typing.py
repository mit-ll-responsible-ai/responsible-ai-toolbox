# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Callable, Iterable, List, Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing_extensions import TypedDict

# Types
Criterion = Callable[[Tensor, Tensor], Tensor]
Perturbation = Callable[[Module, Tensor, Tensor], Tuple[Tensor, Tensor]]
Predictor = Callable[[Optional[Tensor]], Tensor]

PartialOptimizer = Callable[[Iterable], Optimizer]
PartialLRScheduler = Callable[[Optimizer], _LRScheduler]


class PLOptim(TypedDict):
    optimizer: Optimizer
    lr_scheduler: _LRScheduler
    frequency: int


class PartialPLOptim(TypedDict):
    optimizer: PartialOptimizer
    lr_scheduler: PartialLRScheduler
    frequency: int


PartialLightningOptimizer = Union[
    PartialOptimizer, PartialPLOptim, List[PartialPLOptim]
]
