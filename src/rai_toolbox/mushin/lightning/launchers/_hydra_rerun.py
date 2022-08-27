import os
import subprocess
import sys
from time import sleep
from typing import Any, Callable, Dict, Sequence, TypeVar

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.launchers.subprocess_script import (
    _SubprocessScriptLauncher,
)

from rai_toolbox.mushin._compatibility import PL_VERSION, Version

assert PL_VERSION >= Version(1, 6, 0)

R = TypeVar("R")


def _subprocess_call(pickled_config_fn: str, pickled_task_fn: str) -> Sequence[str]:
    from hydra.core.hydra_config import HydraConfig

    assert HydraConfig.initialized()

    # when user is using hydra find the absolute path
    command = [sys.executable, "-m", "rai_toolbox.mushin._pl_rerun_hydra"]

    # extract the hydra configu
    hydra_cfg = HydraConfig.get()

    # the location of the hydra configuration files saved for the current job
    hydra_output = hydra_cfg.runtime.output_dir
    if hydra_cfg.output_subdir is not None:
        hydra_output = os.path.join(hydra_output, hydra_cfg.output_subdir)

    # check if experimental re-run capability exists
    # otherwise use existing config.yaml which may have issues
    pickled_config = os.path.join(hydra_output, "config.pickle")
    pickled_task = os.path.join(hydra_output, "task_fn.pickle")
    command += [f"+config={pickled_config}", f"+task_fn={pickled_task}"]
    return command


class HydraRerunDDP(DDPStrategy):  # type: ignore
    """DDP Strategy that supports Hydra run and multirun jobs."""

    strategy_name = "hydra_rerun_ddp2"

    def setup_environment(self) -> None:
        self.setup_distributed()
        super().setup_environment()

    def _configure_launcher(self) -> None:
        if self.cluster_environment is None:  # pragma: no cover
            raise TypeError("HydraRerunDDP.cluster_environment is None")

        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _HydraDDPLauncher(
                self.cluster_environment, self.num_processes, self.num_nodes
            )
            self._rank_0_will_call_children_scripts = True

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            f"{cls.strategy_name}_find_unused_parameters_false",
            cls,
            description="DDP Strategy with `find_unused_parameters` as False",
            find_unused_parameters=False,
        )
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )

    def teardown(self) -> None:
        """Performs additional teardown steps for PL to allow for Hydra multirun jobs."""
        super().teardown()


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
        # del trainer  # unused
        if (
            not self.cluster_environment.creates_processes_externally
        ):  # pragma: no cover
            testing = function.__name__ == "_test_impl"
            predicting = function.__name__ == "_predict_impl"
            self._call_children_scripts(testing=testing, predicting=predicting)

        return function(*args, **kwargs)

    def _call_children_scripts(self, testing: bool, predicting: bool):
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
            env_copy = os.environ.copy()
            env_copy["LOCAL_RANK"] = f"{local_rank}"

            # remove env var if global seed not set
            if (
                os.environ.get("PL_GLOBAL_SEED") is None
                and "PL_GLOBAL_SEED" in env_copy
            ):
                del env_copy["PL_GLOBAL_SEED"]

            pickled_config_fn = "config.pickle"
            pickled_task_fn = "task_fn.pickle"
            command = _subprocess_call(pickled_config_fn, pickled_task_fn)
            subprocess.Popen(command, env=env_copy)

            # starting all processes at once can cause issues
            # with dataloaders delay between 1-10 seconds
            delay = np.random.uniform(1, 5, 1)[0]
            sleep(delay)
