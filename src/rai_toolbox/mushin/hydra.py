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
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
)

from hydra._internal.callbacks import Callbacks
from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra._internal.hydra import Hydra
from hydra._internal.utils import create_config_search_path
from hydra.core.config_store import ConfigStore
from hydra.core.utils import JobReturn
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, RunMode
from hydra_zen import instantiate
from hydra_zen.typing._implementations import DataClass_
from omegaconf import OmegaConf
from typing_extensions import Literal


def is_config(cfg: Any) -> bool:
    return is_dataclass(cfg) or OmegaConf.is_config(cfg)


@overload
def launch(
    config: Union[DataClass_, Type[DataClass_], Mapping[str, Any]],
    task_function: Callable[[Any], Any],
    overrides: Optional[List[str]] = None,
    config_name: str = "zen_launch",
    job_name: str = "zen_launch",
    with_log_configuration: bool = True,
    multirun: Literal[False] = False,
) -> JobReturn:
    ...


@overload
def launch(
    config: Union[DataClass_, Type[DataClass_], Mapping[str, Any]],
    task_function: Callable[[Any], Any],
    overrides: Optional[List[str]] = None,
    config_name: str = "zen_launch",
    job_name: str = "zen_launch",
    with_log_configuration: bool = True,
    multirun: Literal[True] = True,
) -> List[List[JobReturn]]:
    ...


def launch(
    config: Union[DataClass_, Type[DataClass_], Mapping[str, Any]],
    task_function: Callable[[Any], Any],
    overrides: Optional[List[str]] = None,
    config_name: str = "zen_launch",
    job_name: str = "zen_launch",
    with_log_configuration: bool = True,
    multirun: bool = False,
) -> Union[JobReturn, List[List[JobReturn]]]:

    cs = ConfigStore().instance()
    cs.store(name=config_name, node=config)
    search_path = create_config_search_path(None)

    # initiate Hydra without enforcing single instance
    config_loader = ConfigLoaderImpl(config_search_path=search_path)
    hydra = Hydra(task_name=job_name, config_loader=config_loader)

    if not multirun:
        # Here we can use Hydra's `run` method
        job = hydra.run(
            config_name=config_name,
            task_function=task_function,
            overrides=overrides if overrides is not None else [],
            with_log_configuration=with_log_configuration,
        )

    else:
        # Instead of running Hydra's `multirun` method we instantiate
        # and run the sweeper method.  This allows us to run local
        # sweepers and launchers without installing them in `hydra_plugins`
        # package directory.
        cfg = hydra.compose_config(
            config_name=config_name,
            overrides=overrides if overrides is not None else [],
            with_log_configuration=with_log_configuration,
            run_mode=RunMode.MULTIRUN,
        )

        callbacks = Callbacks(cfg)
        callbacks.on_multirun_start(config=cfg, config_name=config_name)

        # Instantiate sweeper without using Hydra's Plugin discovery (Zen!)
        sweeper = instantiate(cfg.hydra.sweeper)
        assert isinstance(sweeper, Sweeper)
        sweeper.setup(
            config=cfg,
            hydra_context=HydraContext(
                config_loader=hydra.config_loader, callbacks=callbacks
            ),
            task_function=task_function,
        )

        task_overrides = OmegaConf.to_container(cfg.hydra.overrides.task, resolve=False)
        assert isinstance(task_overrides, list)

        job = sweeper.sweep(arguments=task_overrides)
        callbacks.on_multirun_end(config=cfg, config_name=config_name)

    return job


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

            if param.kind is Parameter.POSITIONAL_ONLY:
                continue
            if not hasattr(cfg, name) and param.default is param.empty:
                missing_params.append(name)

        if missing_params:
            raise TypeError(
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
            if param.kind is not param.POSITIONAL_ONLY and name not in kwargs
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
