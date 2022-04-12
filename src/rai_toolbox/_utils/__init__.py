# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Union

import torch as tr


def get_device(obj: Union[tr.nn.Module, tr.Tensor]) -> tr.device:
    if isinstance(obj, tr.nn.Module):
        for p in obj.parameters():
            return p.device
        return tr.device("cpu")

    elif isinstance(obj, tr.Tensor):
        return obj.device

    else:  # pragma: no cover
        raise TypeError(f"Expected torch.nn.Module or torch.Tensor, got {obj}")
