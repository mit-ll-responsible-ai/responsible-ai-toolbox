# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
from time import sleep
from typing import Any, Callable, TypeVar

import numpy as np
from torch import distributed
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.states import TrainerFn

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


def _subprocess_call(local_rank: int, trainer: Trainer) -> None:
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

    # create the command for CLI
    command += ["-cp", hydra_output, "-cn", "config.yaml"]

    if PL_VERSION < Version(1, 6, 0):
        # TODO: See about having PL 1.6 fix this behavior so we know
        # which function is being called
        trainer_fn = trainer.state.fn
        command += [
            "++pl_testing=" + ("false" if trainer_fn == TrainerFn.FITTING else "true")
        ]

    command += [
        f"hydra.output_subdir=.pl_hydra_{local_rank}",
        f"hydra.run.dir={os_cwd}",
        f"hydra.job.name=train_ddp_process_{local_rank}",
    ]
    subprocess.Popen(command, env=env_copy, cwd=cwd)


if PL_VERSION >= Version(1, 6, 0):
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.strategies.launchers.subprocess_script import (
        _SubprocessScriptLauncher,
    )

    class HydraDDP(DDPStrategy):  # type: ignore
        """DDP Strategy that supports Hydra run and multirun jobs.

        This strategy assumes a `Trainer.fit` or `Trainer.test` has been configured
        to execute via Hydra.  It requires that Hydra saves a `config.yaml` in the current working directory with the following keys/properties set:

        - trainer: A `pytorch_lightning.Trainer` configuration
        - module: A `pytorch_lightning.LightningModule` configuration
        - pl_testing: A boolean: True for `Trainer.test` and False (default) `Trainer.fit`

        This strategy will launch a child subprocesses for additional GPU beyond the first using the following base command::

           python -m rai_toolbox.mushin.lightning._pl_main -cp <path to config.yaml> -cn config.yaml

        Notes
        -----
        In order to execute a MULTIRUN Hydra job we must make sure to destroy an distributed
        processes on setup of this function.  This will lead to issues if running multiple jobs
        in the notebook or trying to do `Trainer.fit` followed by `Trainer.test`.


        Examples
        --------
        >>> from pytorch_lightning import Trainer
        >>> from rai_toolbox.mushin import HydraDDP
        >>> PLModule = # some LightningModule
        
        >>> trainer = Trainer(PLModule, accelerator="auto", devices=2, strategy=HydraDDP())
        >>> trainer.fit(module)
        """

        def setup_environment(self) -> None:
            _setup_environment()
            super().setup_environment()

        def _configure_launcher(self) -> None:
            if self.cluster_environment is None:  # pragma: no cover
                raise TypeError("HydraDDP.cluster_environment is None")

            if not self.cluster_environment.creates_processes_externally:
                self._launcher = HydraDDPLauncher(
                    self.cluster_environment, self.num_processes, self.num_nodes
                )
                self._rank_0_will_call_children_scripts = True

        def teardown(self) -> None:
            """Performs additional teardown steps for PL to allow for Hydra multirun jobs."""
            super().teardown()
            _teardown()

    class HydraDDPLauncher(_SubprocessScriptLauncher):
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
            if not self.cluster_environment.creates_processes_externally:
                self._call_children_scripts(trainer)

            return function(*args, **kwargs)

        def _call_children_scripts(self, trainer: Trainer):
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
                _subprocess_call(local_rank, trainer)

                # starting all processes at once can cause issues
                # with dataloaders delay between 1-10 seconds
                delay = np.random.uniform(1, 5, 1)[0]
                sleep(delay)

else:  # pragma: no cover
    from pytorch_lightning.plugins.training_type.ddp import DDPPlugin

    class HydraDDP(DDPPlugin):
        """DDP Strategy that supports Hydra run and multirun jobs.

        This strategy assumes a `Trainer.fit` or `Trainer.test` has been configured
        to execute via Hydra.  It requires that Hydra saves a `config.yaml` in the current working directory with the following keys/properties set:

        - trainer: A `pytorch_lightning.Trainer` configuration
        - module: A `pytorch_lightning.LightningModule` configuration
        - pl_testing: A boolean: True for `Trainer.test` and False (default) `Trainer.fit`

        This strategy will launch a child subprocesses for additional GPU beyond the first using the following base command::

           python -m rai_toolbox.mushin.lightning._pl_main -cp <path to config.yaml> -cn config.yaml


        Notes
        -----
        In order to execute a MULTIRUN Hydra job we must make sure to destroy an 
        distributed processes on setup of this function.  This will lead to issues if 
        running multiple jobs in the notebook or trying to do `Trainer.fit` followed by 
        `Trainer.test`.

        Examples
        --------
        >>> from pytorch_lightning import Trainer
        >>> from rai_toolbox.mushin import HydraDDP
        >>> PLModule = # some LightningModule
        
        >>> trainer = Trainer(PLModule, accelerator="auto", devices=2, strategy=HydraDDP())
        >>> trainer.fit(module)

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
                _subprocess_call(local_rank, self.lightning_module.trainer)

                # starting all processes at once can cause issues
                # with dataloaders delay between 1-10 seconds
                delay = np.random.uniform(1, 5, 1)[0]
                sleep(delay)

            self._rank_0_has_called_call_children_scripts = True

        def teardown(self) -> None:
            """Performs additional teardown steps for PL to allow for Hydra multirun jobs."""
            super().teardown()
            _teardown()
