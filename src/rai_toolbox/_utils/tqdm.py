# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Iterable, Optional, TypeVar

_T = TypeVar("_T", bound=Iterable)

__all__ = ["tqdm"]

def _dummy_tqdm(
    iterable: _T,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    leave=True,
    file=None,
    ncols=None,
    mininterval=0.1,
    maxinterval=10.0,
    miniters=None,
    ascii=None,
    disable=False,
    unit="it",
    unit_scale=False,
    dynamic_ncols=False,
    smoothing=0.3,
    bar_format=None,
    initial=0,
    position=None,
    postfix=None,
    unit_divisor=1000,
) -> _T:
    return iterable
        
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = _dummy_tqdm

