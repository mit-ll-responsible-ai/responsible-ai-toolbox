# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from dataclasses import is_dataclass
from inspect import Parameter, signature
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
    overload,
)

from hydra_zen import instantiate
from hydra_zen.errors import HydraZenValidationError
from hydra_zen.typing._implementations import DataClass
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing_extensions import Literal, ParamSpec, TypeGuard

T1 = TypeVar("T1")
P = ParamSpec("P")


def is_config(cfg: Any) -> TypeGuard[Union[DataClass, DictConfig, ListConfig]]:
    return is_dataclass(cfg) or OmegaConf.is_config(cfg)


SKIPPED_PARAM_KINDS = frozenset(
    (Parameter.POSITIONAL_ONLY, Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL)
)


PreCall = Optional[Union[Callable[[Any], Any], Iterable[Callable[[Any], Any]]]]
PostCall = Optional[Union[Callable[[Any, T1], Any], Iterable[Callable[[Any, T1], Any]]]]


def _flat_call(x: Iterable[Callable[P, Any]]):
    def f(*args: P.args, **kwargs: P.kwargs) -> None:
        for fn in x:
            fn(*args, **kwargs)

    return f


# TODO: - zen's instantiation should be memoized so that subsequent access
#         to an attribute returns the same instance
#       - zen should interface with a singelton that "recognizes"
#         the configs that it interacts with
class Zen(Generic[P, T1]):
    def __init__(
        self,
        func: Callable[P, T1],
        pre_call: PreCall = None,
        post_call: PostCall[T1] = None,
    ) -> None:
        self.func: Callable[P, T1] = func
        self.parameters = signature(self.func).parameters
        self.pre_call = (
            pre_call if not isinstance(pre_call, Iterable) else _flat_call(pre_call)
        )
        self.post_call = (
            post_call if not isinstance(post_call, Iterable) else _flat_call(post_call)
        )

    def validate(self, cfg: Any, excluded_params: Iterable[str] = ()):
        excluded_params = set(excluded_params)

        if callable(cfg):
            cfg = cfg()  # instantiate dataclass to result default-factory

        num_pos_only = sum(
            p.kind is p.POSITIONAL_ONLY for p in self.parameters.values()
        )
        if num_pos_only:
            assert hasattr(cfg, "_args_")
            assert len(cfg._args_) == num_pos_only

        missing_params: List[str] = []
        for name, param in self.parameters.items():
            if name in excluded_params:
                continue

            if param.kind in SKIPPED_PARAM_KINDS:
                continue
            if not hasattr(cfg, name) and param.default is param.empty:
                missing_params.append(name)

        if missing_params:
            raise HydraZenValidationError(
                f"`cfg` is missing the following fields: {', '.join(missing_params)}"
            )

    def __call__(self, cfg: Any, *args: Any, **kwargs: Any) -> T1:
        if callable(cfg):
            cfg = cfg()  # instantiate dataclass to resolve default-factory

        if self.pre_call is not None:
            self.pre_call(cfg)

        if not args:
            args_ = getattr(cfg, "_args_", [])
            assert isinstance(args_, Sequence)
            args_ = list(args_)
        else:
            args_ = list(args)

        del args

        cfg_kwargs = {
            name: (
                getattr(cfg, name, param.default)
                if param.default is not param.empty
                else getattr(cfg, name)
            )
            for name, param in self.parameters.items()
            if param.kind not in SKIPPED_PARAM_KINDS and name not in kwargs
        }

        # instantiate any configs
        args_ = [instantiate(x) if is_config(x) else x for x in args_]
        cfg_kwargs = {
            name: instantiate(val) if is_config(val) else val
            for name, val in cfg_kwargs.items()
        }
        kwargs = {
            name: instantiate(val) if is_config(val) else val
            for name, val in kwargs.items()
        }
        out = self.func(*args_, **cfg_kwargs, **kwargs)  # type: ignore

        if self.post_call is not None:
            self.post_call(cfg, out)

        return out


@overload
def zen(
    __func: Callable[P, T1],
    *,
    pre_call: PreCall = ...,
    post_call: PostCall = ...,
) -> Zen[P, T1]:  # pragma: no cover
    ...


@overload
def zen(
    __func: Literal[None] = None,
    *,
    pre_call: PreCall = ...,
    post_call: PostCall[T1] = ...,
) -> Callable[[Callable[P, T1]], Zen[P, T1]]:  # pragma: no cover
    ...


@overload
def zen(
    __func: Optional[Callable[P, T1]] = None,
    *,
    pre_call: PreCall = None,
    post_call: PostCall[T1] = None,
) -> Union[Zen[P, T1], Callable[[Callable[P, T1]], Zen[P, T1]]]:  # pragma: no cover
    ...


def zen(
    __func: Optional[Callable[P, T1]] = None,
    *,
    pre_call: PreCall = None,
    post_call: PostCall[T1] = None,
) -> Union[Zen[P, T1], Callable[[Callable[P, T1]], Zen[P, T1]]]:

    if __func is None:

        def wrap(f: Callable[P, T1]) -> Zen[P, T1]:
            return Zen(func=f, pre_call=pre_call, post_call=post_call)

        return wrap
    return Zen(__func, pre_call=pre_call, post_call=post_call)
