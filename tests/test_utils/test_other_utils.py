# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import hypothesis.strategies as st
import torch as tr
from hypothesis import given, settings

from rai_toolbox._utils import get_device
from rai_toolbox._utils.tqdm import _dummy_tqdm


from torch.nn import Module

devices = [tr.device("cpu")]

if tr.cuda.is_available():
    devices.append(tr.device("cuda:0"))


class EmptyModule(Module):
    def parameters(self, recurse: bool = True):
        return []

@settings(deadline=None)
@given(
    device=st.sampled_from(devices),
    obj=st.sampled_from([tr.tensor([0.0]), tr.nn.Linear(1, 1)]),
)
def test_get_device(device, obj):
    assert get_device(obj.to(device)) == device


def test_get_device_for_empty_module():
    assert get_device(EmptyModule()) == tr.device("cpu")

@given(st.integers(0, 10))
def test_tqdm(n: int):
    assert list(_dummy_tqdm(range(n))) == list(range(n))
