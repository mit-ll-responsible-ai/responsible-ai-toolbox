# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
import pytest
from hydra_zen import builds, make_config
from hydra_zen.errors import HydraZenValidationError
from hypothesis import given, strategies as st

from rai_toolbox.mushin.hydra import zen


def function(x: int, y: int, z: int = 2):
    return x * y * z


def function_with_args(x: int, y: int, z: int = 2, *args):
    return x * y * z


def function_with_kwargs(x: int, y: int, z: int = 2, **kwargs):
    return x * y * z


def function_with_args_kwargs(x: int, y: int, z: int = 2, *args, **kwargs):
    return x * y * z


class A:
    def f(self, x: int, y: int, z: int = 2):
        return x * y * z


method = A().f


def test_zen_is_deprecated():
    with pytest.warns(FutureWarning):
        zen(lambda x: None)


@pytest.mark.parametrize(
    "func",
    [
        function,
        function_with_args,
        function_with_kwargs,
        function_with_args_kwargs,
        method,
    ],
)
@pytest.mark.parametrize(
    "cfg",
    [
        make_config(),
        make_config(not_a_field=2),
        make_config(x=1),
        make_config(y=2, z=4),
    ],
)
@pytest.mark.filterwarnings("ignore:rai_toolbox.mushin.zen will be removed")
def test_zen_validation(cfg, func):
    with pytest.raises(HydraZenValidationError):
        zen(func).validate(cfg)


@pytest.mark.parametrize(
    "func",
    [
        function,
        function_with_args,
        function_with_kwargs,
        function_with_args_kwargs,
        method,
    ],
)
@given(
    x=st.integers(-10, 10),
    y=st.integers(-10, 10),
    kwargs=st.dictionaries(st.sampled_from(["z", "not_a_field"]), st.integers()),
    instantiate_cfg=st.booleans(),
)
@pytest.mark.filterwarnings("ignore:rai_toolbox.mushin.zen will be removed")
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
@pytest.mark.filterwarnings("ignore:rai_toolbox.mushin.zen will be removed")
def test_instantiation(call, x):
    assert call(x) == x


@pytest.mark.filterwarnings("ignore:rai_toolbox.mushin.zen will be removed")
def test_zen_works_with_non_builds():
    bigger_cfg = make_config(super_conf=make_config(a=builds(int)))
    out = zen(lambda super_conf: super_conf)(bigger_cfg)
    assert out.a == 0


class Pre:
    record = []


class Post:
    record = []


pre_call_strat = st.just(lambda cfg: Pre.record.append(cfg.x))
post_call_strat = st.just(lambda cfg, result: Post.record.append((cfg.y, result)))


@given(
    pre_call=(pre_call_strat | st.lists(pre_call_strat)),
    post_call=(post_call_strat | st.lists(post_call_strat)),
)
@pytest.mark.filterwarnings("ignore:rai_toolbox.mushin.zen will be removed")
def test_pre_and_post_call(pre_call, post_call):
    Pre.record.clear()
    Post.record.clear()
    cfg = make_config(x=1, y="a")
    assert zen(pre_call=pre_call, post_call=post_call)(g)(cfg=cfg) == 1
    assert Pre.record == [1] * (len(pre_call) if isinstance(pre_call, list) else 1)
    assert Post.record == [("a", 1)] * (
        len(post_call) if isinstance(post_call, list) else 1
    )
