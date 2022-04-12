# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Callable, Iterable, List, Mapping, TypeVar, Union

import torch as tr

T = TypeVar("T", bound=Callable)
NoneType = type(None)


def flatten_params(
    iter_params_or_groups: Union[
        tr.Tensor, Iterable[tr.Tensor], Iterable[Mapping[str, Iterable[tr.Tensor]]]
    ]
) -> List[tr.Tensor]:

    if isinstance(iter_params_or_groups, tr.Tensor):
        return [iter_params_or_groups]

    out = []
    for params in iter_params_or_groups:
        if isinstance(params, tr.Tensor):
            out.append(params)
        else:
            out.extend(params["params"])
    return out
