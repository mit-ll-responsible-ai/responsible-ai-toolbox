# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
from pathlib import Path
from time import sleep
from typing import Any, Callable, TypeVar

import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra_zen import load_from_yaml
from omegaconf.errors import ConfigAttributeError
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.states import TrainerFn
from torch import distributed

from .._compatibility import PL_VERSION, Version

R = TypeVar("R")


def _setup_environment() -> None:
    if distributed.is_initialized():
        distributed.destroy_process_group()


def _teardown() -> None:
    # Remove PL environments so next multirun starts fresh
    envs = (
        "LOCAL_RANK",
        "NODE_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "PL_GLOBAL_SEED",
    )

    for name in envs:
        os.environ.pop(name, None)


def _subprocess_call(local_rank: int, testing: bool) -> None:
    env_copy = os.environ.copy()
    env_copy["LOCAL_RANK"] = f"{local_rank}"
    # CWD is the Hydra working directory
    cwd = os.getcwd()
    os_cwd = (
        f'"{cwd}"'  # this is needed to handle characters like `=` in the directory name
    )

    command = [
        sys.executable,
        "-m",
        "rai_toolbox.mushin.lightning._pl_main",
    ]
    hydra_cfg = HydraConfig.get()

    hydra_output = (
        os.path.join(cwd, hydra_cfg.output_subdir)
        if hydra_cfg.output_subdir is not None
        else cwd
    )

    # Validate that minimal configuration requirements
    config = Path(hydra_output) / "config.yaml"
    assert config.exists()
    cfg = load_from_yaml(config)
    if "trainer" not in cfg or "module" not in cfg:
        raise ConfigAttributeError(
            "Missing configurations `trainer` and `module` are required for use with HydraDDP.  See documentation for further details."
        )

    # create the command for CLI
    command += ["-cp", hydra_output, "-cn", "config.yaml"]

    # Set flag to run Trainer.fit or Trainer.test in `_pl_main.py`
    command += ["++pl_testing=" + ("false" if not testing else "true")]

    # Set flag for local rank
    command += [f"++pl_local_rank={local_rank}"]

    command += [
        f"hydra.run.dir={os_cwd}",
        f"hydra.output_subdir=.pl_hydra_rank_{local_rank}",
        f"hydra.job.name={hydra_cfg.job.name}",
    ]
    subprocess.Popen(command, env=env_copy, cwd=cwd)


if PL_VERSION >= Version(1, 6, 0):
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.strategies.launchers.subprocess_script import (
        _SubprocessScriptLauncher,
    )

    class HydraDDP(DDPStrategy):  # type: ignore
        """DDP Strategy that supports Hydra run and multirun jobs.

        This strategy assumes a PyTorch Lightning `Trainer.fit` or `Trainer.test` has been configured
        to execute via Hydra.  It requires that Hydra saves a `config.yaml` in the current working directory with the following keys/properties set::

           ├── Config
           │    ├── trainer: A `pytorch_lightning.Trainer` configuration
           │    ├── module: A `pytorch_lightning.LightningModule` configuration
           │    ├── datamodule: [OPTIONAL] A `pytorch_lightning.LightningDataModule` configuration

        This strategy will launch a child subprocesses for additional GPU beyond the first using the following base command::

           python -m rai_toolbox.mushin.lightning._pl_main -cp <path to config.yaml> -cn config.yaml

        Examples
        --------

        First define a Hydra configuration using hydra-zen:

        >>> import pytorch_lightning as pl
        ... from hydra_zen import builds, make_config,
        ... from rai_toolbox.mushin import HydraDDP
        ... from rai_toolbox.mushin.testing.lightning import SimpleLightningModule
        ...
        ... TrainerConfig = builds(
        ...     pl.Trainer,
        ...     accelerator="auto",
        ...     gpus=2,
        ...     max_epochs=1,
        ...     fast_dev_run=True,
        ...     strategy=builds(HydraDDP),
        ...     populate_full_signature=True
        ... )
        ...
        ... ModuleConfig = builds(SimpleLightningModule)
        ...
        ... Config = make_config(
        ...     trainer=TrainerConfig,
        ...     module=ModuleConfig
        ... )

        Next, define a task function to execute the Hydra job:

        >>> from hydra_zen import instantiate
        >>> def task_function(cfg):
        ...     obj = instantiate(cfg)
        ...     obj.trainer.fit(obj.module)

        Launch the Hydra+Lightning DDP job

        >>> from hydra_zen import launch
        >>> job = launch(Config, task_function)

        ``HydraDDP`` also supports ``LightningDataModule`` configuration.

        >>> DataModuleConfig = ... # A LightningDataModule config
        >>> Config = make_config(
        ...     trainer=TrainerConfig,
        ...     module=ModuleConfig
        ...     datamodule=DataModuleconfig
        ... )

        Next define a task function to execute the Hydra job:

        >>> from hydra_zen import instantiate
        >>> def task_function(cfg):
        ...     obj = instantiate(cfg)
        ...     obj.trainer.fit(obj.module, datamodule=obj.datamodule)

        Launch the Hydra+Lightning DDP job:

        >>> from hydra_zen import launch
        >>> job = launch(Config, task_function)
        """

        def setup_environment(self) -> None:
            _setup_environment()
            super().setup_environment()

        def _configure_launcher(self) -> None:
            if self.cluster_environment is None:  # pragma: no cover
                raise TypeError("HydraDDP.cluster_environment is None")

            if not self.cluster_environment.creates_processes_externally:
                self._launcher = _HydraDDPLauncher(
                    self.cluster_environment, self.num_processes, self.num_nodes
                )
                self._rank_0_will_call_children_scripts = True

        def teardown(self) -> None:
            """Performs additional teardown steps for PL to allow for Hydra multirun jobs."""
            super().teardown()
            _teardown()

    class _HydraDDPLauncher(_SubprocessScriptLauncher):
        @property
        def is_interactive_compatible(self) -> bool:  # pragma: no cover
            return True

        def launch(
            self,
            function: Callable[..., R],
            *args: Any,
            trainer: Trainer,
            **kwargs: Any,
        ) -> R:
            """Creates new processes, then calls the given function.

            Parameters
            ----------
            function : Callable[[...], ReturnType]
                A callback function to execute after all processes have been created.
                It is up to the implementation of this function to synchronize the processes, e.g., with barriers.

            *args : Any
                Optional positional arguments to be passed to the given function.

            trainer : pytorch_lightning.Trainer
                Optional reference to the pytorch_lightning.Trainer`.

            **kwargs : Any
                Optional keyword arguments to be passed to the given function.

            Returns
            -------
            ReturnType
            """
            del trainer  # unused
            if not self.cluster_environment.creates_processes_externally:
                testing = function.__name__ == "_test_impl"
                self._call_children_scripts(testing=testing)

            return function(*args, **kwargs)

        def _call_children_scripts(self, testing: bool):
            # bookkeeping of spawned processes
            self._check_can_spawn_children()

            # DDP Environment variables
            os.environ["MASTER_ADDR"] = self.cluster_environment.main_address
            os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)

            # allow the user to pass the node rank
            os.environ["NODE_RANK"] = str(self.cluster_environment.node_rank())
            os.environ["LOCAL_RANK"] = str(self.cluster_environment.local_rank())
            os.environ["WORLD_SIZE"] = f"{self.num_processes * self.num_nodes}"

            for local_rank in range(1, self.num_processes):
                _subprocess_call(local_rank, testing)

                # starting all processes at once can cause issues
                # with dataloaders delay between 1-10 seconds
                delay = np.random.uniform(1, 5, 1)[0]
                sleep(delay)

else:  # pragma: no cover
    from pytorch_lightning.plugins.training_type.ddp import DDPPlugin

    class HydraDDP(DDPPlugin):
        """DDP Strategy that supports Hydra run and multirun jobs.

        This strategy assumes a PyTorch Lightning `Trainer.fit` or `Trainer.test` has been configured
        to execute via Hydra.  It requires that Hydra saves a `config.yaml` in the current working directory with the following keys/properties set::

           ├── Config
           │    ├── trainer: A `pytorch_lightning.Trainer` configuration
           │    ├── module: A `pytorch_lightning.LightningModule` configuration
           │    ├── datamodule: [OPTIONAL] A `pytorch_lightning.LightningDataModule` configuration

        This strategy will launch a child subprocesses for additional GPU beyond the first using the following base command::

           python -m rai_toolbox.mushin.lightning._pl_main -cp <path to config.yaml> -cn config.yaml

        Examples
        --------

        First define a Hydra configuration using hydra-zen:

        >>> import pytorch_lightning as pl
        ... from hydra_zen import builds, make_config,
        ... from rai_toolbox.mushin import HydraDDP
        ... from rai_toolbox.mushin.testing.lightning import SimpleLightningModule
        ...
        ... TrainerConfig = builds(
        ...     pl.Trainer,
        ...     accelerator="auto",
        ...     gpus=2,
        ...     max_epochs=1,
        ...     fast_dev_run=True,
        ...     strategy=builds(HydraDDP),
        ...     populate_full_signature=True
        ... )
        ...
        ... ModuleConfig = builds(SimpleLightningModule)
        ...
        ... Config = make_config(
        ...     trainer=TrainerConfig,
        ...     module=ModuleConfig
        ... )

        Next define a task function to execute the Hydra job:

        >>> from hydra_zen import instantiate
        >>> def task_function(cfg):
        ...     obj = instantiate(cfg)
        ...     obj.trainer.fit(obj.module)

        Launch the Hydra+Lightning DDP job:

        >>> from hydra_zen import launch
        >>> job = launch(Config, task_function)

        ``HydraDDP`` also supports ``LightningDataModule`` configuration.

        >>> DataModuleConfig = ... # A LightningDataModule config
        >>> Config = make_config(
        ...     trainer=TrainerConfig,
        ...     module=ModuleConfig
        ...     datamodule=DataModuleconfig
        ... )

        Next, define a task function to execute the Hydra job:

        >>> from hydra_zen import instantiate
        >>> def task_function(cfg):
        ...     obj = instantiate(cfg)
        ...     obj.trainer.fit(obj.module, datamodule=obj.datamodule)

        Launch the Hydra+Lightning DDP job:

        >>> from hydra_zen import launch
        >>> job = launch(Config, task_function)
        """

        def setup_environment(self) -> None:
            _setup_environment()
            super().setup_environment()

        def _call_children_scripts(self):
            if self.lightning_module is None:  # pragma: no cover
                raise TypeError("HydraDDP.lightning_module is None")

            if self.lightning_module.trainer is None:  # pragma: no cover
                raise TypeError("HydraDDP.lightning_module.trainer is None")

            if self.cluster_environment is None:  # pragma: no cover
                raise TypeError("HydraDDP.cluster_environment is None")

            # bookkeeping of spawned processes
            self._check_can_spawn_children()  # type: ignore

            # DDP Environment variables
            os.environ["MASTER_ADDR"] = self.cluster_environment.master_address()  # type: ignore
            os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())  # type: ignore

            # allow the user to pass the node rank
            os.environ["NODE_RANK"] = str(self.cluster_environment.node_rank())
            os.environ["LOCAL_RANK"] = str(self.cluster_environment.local_rank())
            os.environ["WORLD_SIZE"] = f"{self.num_processes * self.num_nodes}"

            self.interactive_ddp_procs = []
            for local_rank in range(1, self.num_processes):
                testing = self.lightning_module.trainer.state.fn == TrainerFn.TESTING
                _subprocess_call(local_rank, testing=testing)

                # starting all processes at once can cause issues
                # with dataloaders delay between 1-10 seconds
                delay = np.random.uniform(1, 5, 1)[0]
                sleep(delay)

            self._rank_0_has_called_call_children_scripts = True

        def teardown(self) -> None:
            """Performs additional teardown steps for PL to allow for Hydra multirun jobs."""
            super().teardown()
            _teardown()
