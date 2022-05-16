# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from dataclasses import is_dataclass
from inspect import Parameter, signature
from typing import Any, Callable, Generic, Iterable, List, Sequence, TypeVar

from hydra_zen import instantiate
from hydra_zen.errors import HydraZenValidationError
from hydra_zen.typing._implementations import DataClass_

from omegaconf import OmegaConf


def is_config(cfg: Any) -> bool:
    return is_dataclass(cfg) or OmegaConf.is_config(cfg)


SKIPPED_PARAM_KINDS = frozenset(
    (Parameter.POSITIONAL_ONLY, Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL)
)


T1 = TypeVar("T1")


# TODO: - zen's instantiation should be memoized so that subsequent access
#         to an attribute returns the same instance
#       - zen should interface with a singelton that "recognizes"
#         the configs that it interacts with
class zen(Generic[T1]):
    def __init__(self, func: Callable[..., T1]) -> None:
        self.func = func
        self.parameters = signature(self.func).parameters

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
        return self.func(*args_, **cfg_kwargs, **kwargs)
