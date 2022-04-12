# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
import pytest
from hydra_zen import builds, make_config
from hypothesis import given
from hypothesis import strategies as st

from rai_toolbox.mushin.hydra import zen


def function(x: int, y: int, z: int = 2):
    return x * y * z


class A:
    def f(self, x: int, y: int, z: int = 2):
        return x * y * z


method = A().f


@pytest.mark.parametrize("func", [function, method])
@pytest.mark.parametrize(
    "cfg",
    [
        make_config(),
        make_config(not_a_field=2),
        make_config(x=1),
        make_config(y=2, z=4),
    ],
)
def test_zen_validation(cfg, func):
    with pytest.raises(TypeError):
        zen(func).validate(cfg)


@pytest.mark.parametrize("func", [function, method])
@given(
    x=st.integers(-10, 10),
    y=st.integers(-10, 10),
    kwargs=st.dictionaries(st.sampled_from(["z", "not_a_field"]), st.integers()),
    instantiate_cfg=st.booleans(),
)
def test_zen_call(x: int, y: int, kwargs: dict, instantiate_cfg, func):

    cfg = make_config(x=x, y=y, **kwargs)
    if instantiate_cfg:
        cfg = cfg()

    kwargs.pop("not_a_field", None)
    expected = func(x, y, **kwargs)
    actual = zen(func)(cfg)
    assert expected == actual


def g(x: int):
    return x


def raises():
    raise AssertionError("shouldn't have been called!")


zen_g = zen(g)


@pytest.mark.parametrize(
    "call",
    [
        lambda x: zen_g(make_config(x=builds(int, x))),
        lambda x: zen_g(make_config(x=builds(raises), y=builds(raises)), x=x),
        lambda x: zen_g(
            make_config(x=builds(raises), y=builds(raises)), x=builds(int, x)
        ),
    ],
)
@given(x=st.integers(-10, 10))
def test_instantiation(call, x):
    assert call(x) == x


def test_zen_works_with_non_builds():
    bigger_cfg = make_config(super_conf=make_config(a=builds(int)))
    out = zen(lambda super_conf: super_conf)(bigger_cfg)
    assert out.a == 0
