# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from abc import ABC, abstractstaticmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import torch as tr
from hydra_zen import load_from_yaml, make_config

from .hydra import launch, zen


class BaseWorkflow(ABC):
    cfgs: Mapping[str, Any]
    "List of configurations for each Hydra job"

    metrics: Dict[str, Any]
    "Dictionary of metrics for across all jobs"

    metric_params: Dict[str, Any]
    "Workflow parameters defined additional arguments to ``run``"

    def __init__(self, eval_task_cfg=None) -> None:
        """Workflows and experiments using Hydra.

        Workflows are expected to contain subdirectories of Hydra runs
        containing the Hydra YAML configuration and any saved metrics
        file (defined by the evaulation task)::

            ├── working_dir
            │    ├── <experiment directory name: 0>
            │    |    ├── <hydra output subdirectory: (default: .hydra)>
            |    |    |    ├── config.yaml
            |    |    |    ├── hydra.yaml
            |    |    |    ├── overrides.yaml
            │    |    ├── <metrics_filename>
            │    ├── <experiment directory name: 1>
            |    |    ...


        Parameters
        ----------
        eval_task_cfg: Mapping | None (default: None)
            The workflow configuration object

        """
        # we can do validation checks here
        self.eval_task_cfg = (
            eval_task_cfg if eval_task_cfg is not None else make_config()
        )
        self.validate()

    @abstractstaticmethod
    def evaluation_task(*args: Any, **kwargs: Any) -> Any:
        """User defined evaluation task to run the workflow.

        Arguments should be instantiated configuration variables.  For example
        if the the workflow configuration is structured as::

            ├── eval_task_cfg
            │    ├── trainer
            |    ├── module
            |    ├── another_config

        The inputs to ``evaluation_task` can be any of the three configurations:
        ``trainer``, ``module``, or ``another_config`` such as::

        .. code-block:: python

            def evaluation_task(trainer: Trainer, module: LightningModule) -> None:
                trainer.fit(module)
        """

    def validate(self):
        """Valide the configuration will execute with the user defined evaluation task"""
        zen(self.evaluation_task).validate(self.eval_task_cfg)

    def run(
        self,
        *,
        working_dir: Optional[str] = None,
        sweeper: Optional[str] = None,
        launcher: Optional[str] = None,
        overrides: Optional[Sequence[str]] = None,
        **workflow_params: str,
    ):
        """Run the experiment.

        Individual workflows can expclitly define ``workflow_params`` to improve
        readability and undstanding of what parameters are expected for a particular
        workflow.

        Parameters
        ----------
        working_dir: str (default: None, the Hydra default will be used)
            The directory to run the experiment in.  This value is used for
            setting `hydra.sweep.dir`.

        sweeper: str | None (default: None)
            The configuration name of the Hydra Sweeper to use (i.e., the override for ``hydra/sweeper=sweeper``)

        launcher: str | None (default: None)
            The configuration name of the Hydra Launcher to use (i.e., the override for ``hydra/launcher=launcher``)

        overrides: Sequence[str] | None (default: None)
            Parameter overrides not considered part of the workflow parameter set.
            This is helpful for filtering out parameters stored in ``self.metric_params``.

        **workflow_params: str
            These parameters represent the values for configurations to use for the experiment.
            These values will be appeneded to the `overrides` for the Hydra job.
        """
        if overrides is None:
            overrides = []
        else:
            overrides = overrides

        if working_dir is not None:
            overrides += [f"hydra.sweep.dir={working_dir}"]

        if sweeper is not None:
            overrides += [f"hydra/sweeper={sweeper}"]

        if launcher is not None:
            overrides += [f"hydra/launcher={launcher}"]

        for k, v in workflow_params.items():
            overrides.append(f"{k}={v}")

        # Run a Multirun over epsilons
        (jobs,) = launch(
            self.eval_task_cfg,
            zen(self.evaluation_task),
            overrides=overrides,
            multirun=True,
        )

        if working_dir is not None:
            self.working_dir = working_dir

        else:
            # TODO: Is there a better way of getting the directory?
            # TODO: Raise error or warning if no jobs?
            if len(jobs) > 0:
                first_job_working_dir = jobs[0].working_dir
                workflow_base_dir = Path(first_job_working_dir).parent
                self.working_dir = workflow_base_dir

        self.jobs = jobs
        self.load_workflow_jobs_(
            self.working_dir, workflow_params=workflow_params.keys()
        )

    def load_workflow_jobs_(
        self,
        working_dir: Union[Path, str],
        config_dir: str = ".hydra",
        metrics_filename: str = "test_metrics.pt",
        workflow_params: Optional[Sequence[str]] = None,
    ) -> None:
        """Loading workflow job data from a given working directory.

        Parameters
        ----------
        working_dir: str | Path
            The base working directory of the experiment. It is expected
            that subdirectories within this working directory will contain
            individual Hydra jobs data (yaml configurations) and saved metrics files.

        config_dir: str (default: ".hydra")
            The directory in an experiment that stores Hydra configurations (``hydra.output_subdir``)

        metrics_filename: str (default: "test_metrics.pt")
            The filename used to save metrics for each individual Hydra job. This can
            be a search pattern as well since this is appended to

                ``Path(working_dir).glob("**/*/<metrics_filename")``

        workflow_params: Sequence[str] | None (default: None)
            A string of parameters to use for ``metric_params``.  If ``None`` it will
            default to all parameters saved in Hydra's ``overrides.yaml`` file.
        """

        # Load saved YAML configurations for each job (in hydra.job.output_subdir)
        job_cfgs = [
            load_from_yaml(f)
            for f in sorted(Path(working_dir).glob(f"**/*/{config_dir}/config.yaml"))
        ]

        # Load saved YAML overrides for each job (in hydra.job.output_subdir)
        job_overrides = [
            load_from_yaml(f)
            for f in sorted(Path(working_dir).glob(f"**/*/{config_dir}/overrides.yaml"))
        ]

        # Load metrics for each job
        job_metrics = [
            tr.load(f)
            for f in sorted(Path(working_dir).glob(f"**/*/{metrics_filename}"))
        ]

        self.cfgs = job_cfgs
        self.metric_params = defaultdict(list)
        self.metrics = defaultdict(list)
        for task_overrides, metrics in zip(job_overrides, job_metrics):
            for override in task_overrides:
                k, v = override.split("=")
                if workflow_params is None or k in workflow_params:
                    try:
                        v = float(v)
                    except ValueError:
                        pass

                    self.metric_params[k].append(v)

            for k, v in metrics.items():
                if isinstance(v, list) and len(v) == 1:
                    v = v[0]
                self.metrics[k].append(v)

    def plot(self, **kwargs) -> None:
        """Plot workflow metrics"""
        raise NotImplementedError()

    def to_dataframe(self):
        """Convert workflow data to Pandas DataFrame."""
        raise NotImplementedError()

    def to_xarray(self):
        """Convert workflow data to xArray Dataset or DataArray"""
        raise NotImplementedError()


class RobustnessCurve(BaseWorkflow):
    """Abstract class for workflows that measure performance for different perturbation.

    This workflow requires and uses parameter ``epsilon`` as the configuration option for varying
    the a perturbation.
    """

    def run(
        self,
        *,
        epsilon: Union[str, Sequence[float], range],
        working_dir: Optional[str] = None,
        sweeper: Optional[str] = None,
        launcher: Optional[str] = None,
        overrides: Optional[Sequence[str]] = None,
        **workflow_params: str,
    ):
        """Run the experiment for varying the perturbation value ``epsilon``.

        Parameters
        ----------
        epsilon: str | Sequence[float] | range
            The configuration parameter for the perturbation.  Unlike Hydra overrides this
            parameter can be a list of floats that will be conveted into a multirun sequence
            override for Hydra.

        working_dir: str (default: None, the Hydra default will be used)
            The directory to run the experiment in.  This value is used for
            setting `hydra.sweep.dir`.

        sweeper: str | None (default: None)
            The configuration name of the Hydra Sweeper to use (i.e., the override for ``hydra/sweeper=sweeper``)

        launcher: str | None (default: None)
            The configuration name of the Hydra Launcher to use (i.e., the override for ``hydra/launcher=launcher``)

        overrides: Sequence[str] | None (default: None)
            Parameter overrides not considered part of the workflow parameter set.
            This is helpful for filtering out parameters stored in ``self.metric_params``.

        **workflow_params: str
            These parameters represent the values for configurations to use for the experiment.
            These values will be appeneded to the `overrides` for the Hydra job.
        """

        if not isinstance(epsilon, str):
            eps_arg = ",".join(str(i) for i in epsilon)
        elif isinstance(epsilon, range):
            eps_arg = repr(epsilon)
        else:
            eps_arg = epsilon

        return super().run(
            epsilon=eps_arg,
            working_dir=working_dir,
            sweeper=sweeper,
            launcher=launcher,
            overrides=overrides,
            **workflow_params,
        )

    def to_dataframe(self):
        """Convert workflow data to Pandas DataFrame"""
        import pandas as pd

        d = {}
        d.update(self.metrics)
        d.update(self.metric_params)
        return pd.DataFrame(d).sort_values("epsilon")

    def to_xarray(self, dim: str = "x"):
        """Convert workflow data to xarray Dataset"""
        import xarray as xr

        return xr.Dataset(
            {k: ([dim], v) for k, v in self.metrics.items()},
            coords={k: ([dim], v) for k, v in self.metric_params.items()},
        ).sortby("epsilon")

    def plot(
        self,
        metric: str,
        ax=None,
        group: Optional[str] = None,
        dim: str = "x",
        save_fig: bool = False,
        **kwargs,
    ) -> None:
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        xdata = self.to_xarray(dim=dim)
        if group is None:
            plots = xdata[metric].plot(x="epsilon", ax=ax, **kwargs)

        else:
            dg = xdata.groupby(group)
            plots = [
                grp[metric].plot(x="epsilon", label=name, ax=ax, **kwargs)
                for name, grp in dg
            ]

        if save_fig:
            plt.savefig("robustness_curve.pdf")

        return plots
