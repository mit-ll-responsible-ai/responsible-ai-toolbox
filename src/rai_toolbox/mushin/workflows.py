# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod, abstractstaticmethod
from collections import UserList, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch as tr
from hydra.core.utils import JobReturn
from hydra_zen import load_from_yaml, make_config
from typing_extensions import Self, TypeAlias, TypeGuard

from rai_toolbox._utils import value_check

from .hydra import launch, zen

LoadedValue: TypeAlias = Union[str, int, float]

__all__ = [
    "BaseWorkflow",
    "RobustnessCurve",
    "MultiRunMetricsWorkflow",
    "multirun",
    "hydra_list",
]


class multirun(UserList):
    """Signals that a sequence is to be iterated over in a multirun"""

    pass


class hydra_list(UserList):
    """Signals that a sequence is provided as a single configured value (i.e. it is not
    to be iterated over during a multirun)"""

    pass


class BaseWorkflow(ABC):
    """Provides an interface for creating a reusable workflow: encapsulated
    "boilerplate" for running, aggregating, and analyzing one or more Hydra jobs.

    Attributes
    ----------
    cfgs : List[Any]
        List of configurations for each Hydra job.

    metrics : Dict[str, List[Any]]
        Dictionary of metrics for across all jobs.

    workflow_overrides : Dict[str, Any]
        Workflow parameters defined as additional arguments to `run`.

    jobs : List[Any]
        List of jobs returned for each experiment within the workflow.

    working_dir: pathlib.Path
        The working directory of the experiment defined by Hydra's sweep directory
        (`hydra.sweep.dir`).
    """

    cfgs: List[Any]
    metrics: Dict[str, List[Any]]
    workflow_overrides: Dict[str, Any]
    jobs: List[Any]
    working_dir: Path

    def __init__(self, eval_task_cfg=None) -> None:
        """Workflows and experiments using Hydra.

        Parameters
        ----------
        eval_task_cfg: Mapping | None (default: None)
            The workflow configuration object.

        """
        # we can do validation checks here
        self.eval_task_cfg = (
            eval_task_cfg if eval_task_cfg is not None else make_config()
        )

        # initialize attributes
        self.cfgs = []
        self.metrics = {}
        self.workflow_overrides = {}
        self._multirun_task_overrides = {}
        self.jobs = []
        self.working_dir = Path.cwd()  # TODO: I don't think we should assume this

    @property
    def multirun_task_overrides(
        self,
    ) -> Dict[str, Union[LoadedValue, Sequence[LoadedValue]]]:
        # e.g. {'epsilon': [1.0, 2.0, 3.0], "foo": "apple"}
        if not self._multirun_task_overrides:
            for entry in load_from_yaml(
                self.working_dir / "multirun.yaml"
            ).hydra.overrides.task:
                entry: str  # e.g. +epsilon=1.0,3.0,2.0
                param_name, val = entry.split("=", 1)
                if "," in val:
                    val = [_num_from_string(v) for v in val.split(",")]
                else:
                    val = _num_from_string(val)

                param_name = param_name.split("+")[-1]
                self._multirun_task_overrides[param_name] = val
        return self._multirun_task_overrides

    @abstractstaticmethod
    def evaluation_task(*args: Any, **kwargs: Any) -> Any:
        """User-defined evaluation task to run the workflow.

        Arguments should be instantiated configuration variables.  For example,
        if the the workflow configuration is structured as::

            ├── eval_task_cfg
            │    ├── trainer
            |    ├── module
            |    ├── another_config

        The inputs to `evaluation_task` can be any of the three configurations:
        `trainer`, `module`, or `another_config` such as::

            def evaluation_task(trainer: Trainer, module: LightningModule) -> None:
                trainer.fit(module)
        """
        raise NotImplementedError()

    def validate(self):
        """Valide the configuration will execute with the user defined evaluation task"""
        zen(self.evaluation_task).validate(self.eval_task_cfg)

    def run(
        self,
        *,
        working_dir: Optional[str] = None,
        sweeper: Optional[str] = None,
        launcher: Optional[str] = None,
        overrides: Optional[List[str]] = None,
        **workflow_overrides: Union[str, int, float, bool, dict, multirun, hydra_list],
    ):
        """Run the experiment.

        Individual workflows can expclitly define `workflow_overrides` to improve
        readability and undstanding of what parameters are expected for a particular
        workflow.

        Parameters
        ----------
        working_dir: str (default: None, the Hydra default will be used)
            The directory to run the experiment in.  This value is used for
            setting `hydra.sweep.dir`.

        sweeper: str | None (default: None)
            The configuration name of the Hydra Sweeper to use (i.e., the override for
            `hydra/sweeper=sweeper`)

        launcher: str | None (default: None)
            The configuration name of the Hydra Launcher to use (i.e., the override for
            `hydra/launcher=launcher`)

        overrides: List[str] | None (default: None)
            Parameter overrides not considered part of the workflow parameter set.
            This is helpful for filtering out parameters stored in
            `self.workflow_overrides`.

        **workflow_overrides: str | int | float | bool | multirun | hydra_list | dict
            These parameters represent the values for configurations to use for the
            experiment.

            Passing `param=multirun([1, 2, 3])` will perform a multirun over those
            three param values, whereas passing `param=hydra_list([1, 2, 3])` will
            pass the entire list as a single input.

            These values will be appended to the `overrides` for the Hydra job.
        """
        self._workflow_overrides = workflow_overrides

        if overrides is None:
            overrides = []

        if working_dir is not None:
            overrides.append(f"hydra.sweep.dir={working_dir}")

        if sweeper is not None:
            overrides.append(f"hydra/sweeper={sweeper}")

        if launcher is not None:
            overrides.append(f"hydra/launcher={launcher}")

        for k, v in workflow_overrides.items():
            value_check(k, v, type_=(int, float, bool, str, dict, multirun, hydra_list))
            if isinstance(v, multirun):
                v = ",".join(str(item) for item in v)

            prefix = ""
            if (
                not hasattr(self.eval_task_cfg, k)
                or getattr(self.eval_task_cfg, k) is None
            ):
                prefix = "+"

            overrides.append(f"{prefix}{k}={v}")

        # Run a Multirun over epsilons
        (jobs,) = launch(
            self.eval_task_cfg,
            zen(self.evaluation_task),
            overrides=overrides,
            multirun=True,
        )

        self.jobs = jobs
        self.jobs_post_process()

    @abstractmethod
    def jobs_post_process(self):
        """Method to extract attributes and metrics relevant to the workflow."""

    def plot(self, **kwargs) -> None:
        """Plot workflow metrics."""
        raise NotImplementedError()

    def to_dataframe(self):
        """Convert workflow data to Pandas DataFrame."""
        raise NotImplementedError()

    def to_xarray(self):
        """Convert workflow data to xArray Dataset or DataArray."""
        raise NotImplementedError()


def _num_from_string(str_input: str) -> LoadedValue:
    try:
        val = float(str_input)
        if val.is_integer() and "." not in str_input:
            # v is e.g., 1 or -2. Not 1.0 or -2.0
            val = int(str_input)
        return val
    except ValueError:
        return str_input


def _non_str_sequence(x: Any) -> TypeGuard[Sequence[Any]]:
    return isinstance(x, Sequence) and not isinstance(x, str)


class MultiRunMetricsWorkflow(BaseWorkflow):
    """Abstract class for workflows that record metrics using Hydra multirun.

    This workflow creates subdirectories of multirun experiments using Hydra.  These directories
    contain the Hydra YAML configuration and any saved metrics file (defined by the evaulation task)::

        ├── working_dir
        │    ├── <experiment directory name: 0>
        │    |    ├── <hydra output subdirectory: (default: .hydra)>
        |    |    |    ├── config.yaml
        |    |    |    ├── hydra.yaml
        |    |    |    ├── overrides.yaml
        │    |    ├── <metrics_filename>
        │    ├── <experiment directory name: 1>
        |    |    ...

    The evaluation task is expected to return a dictionary that maps
    `metric-name (str) -> value (number | Sequence[number])`

    Examples
    --------
    Let's create a simple workflow where we perform a multirun over a parameter,
    `epsilon`, and evaluate a task function that computes an accuracy and loss based on
    that `epsilon` value and a specified `scale`.

    >>> from rai_toolbox.mushin.workflows import MultiRunMetricsWorkflow
    >>> from rai_toolbox.mushin import multirun
    >>> import torch as tr
    ... class LocalRobustness(MultiRunMetricsWorkflow):
    ...     @staticmethod
    ...     def evaluation_task(epsilon: float, scale: float) -> dict:
    ...         epsilon *= scale
    ...         val = 100 - epsilon**2
    ...         result = dict(accuracies=val+2, loss=epsilon**2)
    ...         tr.save(result, "test_metrics.pt")
    ...         return result

    We'll run this workflow for six total configurations of three `epsilon` values and two `scale` values. This will launch a Hydra multirun job and aggregate the results.

    >>> wf = LocalRobustness()
    >>> wf.run(epsilon=multirun([1.0, 2.0, 3.0]), scale=multirun([0.1, 1.0]))
    [2022-05-02 11:57:59,219][HYDRA] Launching 6 jobs locally
    [2022-05-02 11:57:59,220][HYDRA] 	#0 : +epsilon=1.0 +scale=0.1
    [2022-05-02 11:57:59,312][HYDRA] 	#1 : +epsilon=1.0 +scale=1.0
    [2022-05-02 11:57:59,405][HYDRA] 	#2 : +epsilon=2.0 +scale=0.1
    [2022-05-02 11:57:59,498][HYDRA] 	#3 : +epsilon=2.0 +scale=1.0
    [2022-05-02 11:57:59,590][HYDRA] 	#4 : +epsilon=3.0 +scale=0.1
    [2022-05-02 11:57:59,683][HYDRA] 	#5 : +epsilon=3.0 +scale=1.0

    Now that this workflow has run, we can view the results as an xarray-dataset whose
    coordinates reflect the multirun parameters that were varied, and whose data-variables are our recorded metrics: "accuracies" and "loss".

    >>> ds = wf.to_xarray()
    >>> ds
    <xarray.Dataset>
    Dimensions:     (epsilon: 3, scale: 2)
    Coordinates:
    * epsilon     (epsilon) float64 1.0 2.0 3.0
    * scale       (scale) float64 0.1 1.0
    Data variables:
        accuracies  (epsilon, scale) float64 102.0 101.0 102.0 98.0 101.9 93.0
        loss        (epsilon, scale) float64 0.01 1.0 0.04 4.0 0.09 9.0

    We can also load this workflow by providing the working directory where it was run.

    >>> loaded = LocalRobustness().load_from_dir(wf.working_dir)
    >>> loaded.to_xarray()
    <xarray.Dataset>
    Dimensions:     (epsilon: 3, scale: 2)
    Coordinates:
    * epsilon     (epsilon) float64 1.0 2.0 3.0
    * scale       (scale) float64 0.1 1.0
    Data variables:
        accuracies  (epsilon, scale) float64 102.0 101.0 102.0 98.0 101.9 93.0
        loss        (epsilon, scale) float64 0.01 1.0 0.04 4.0 0.09 9.0
    """

    @abstractstaticmethod
    def evaluation_task(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError()

    def jobs_post_process(self):
        assert len(self.jobs) > 0
        # TODO: Make protocol type for JobReturn
        assert isinstance(self.jobs[0], JobReturn)

        # set working directory of this workflow
        first_job_working_dir = self.jobs[0].working_dir
        assert first_job_working_dir is not None
        self.working_dir = Path(first_job_working_dir).parent

        # extract configs, overrides, and metrics
        self.cfgs = [j.cfg for j in self.jobs]
        job_overrides = [j.hydra_cfg.hydra.overrides.task for j in self.jobs]
        job_metrics = [j.return_value for j in self.jobs]
        workflow_params = list(self._workflow_overrides.keys())

        self.metrics, self.workflow_overrides = _load_metrics(
            job_overrides, job_metrics, workflow_params
        )

    def load_from_dir(
        self: Self,
        working_dir: Union[Path, str],
        config_dir: str = ".hydra",
        metrics_filename: str = "test_metrics.pt",
        workflow_params: Optional[Sequence[str]] = None,
    ) -> Self:
        """Loading workflow job data from a given working directory. The workflow
        is loaded in-place and "self" is returned by this method.

        Parameters
        ----------
        working_dir: str | Path
            The base working directory of the experiment. It is expected
            that subdirectories within this working directory will contain
            individual Hydra jobs data (yaml configurations) and saved metrics files.

        config_dir: str (default: ".hydra")
            The directory in an experiment that stores Hydra configurations (`hydra.output_subdir`)

        metrics_filename: str (default: "test_metrics.pt")
            The filename used to save metrics for each individual Hydra job. This can
            be a search pattern as well since this is appended to

                `Path(working_dir).glob("**/*/<metrics_filename")`

        workflow_params: Sequence[str] | None (default: None)
            A string of parameters to use for `workflow_params`.  If `None` it will
            default to all parameters saved in Hydra's `overrides.yaml` file.

        Returns
        -------
        loaded_workflow : Self
        """
        self.working_dir = Path(working_dir).resolve()

        multirun_cfg = self.working_dir / "multirun.yaml"
        assert (
            multirun_cfg.exists()
        ), "Working directory does not contain `multirun.yaml` file.  Be sure to use the value of the Hydra sweep directory for the workflow"

        # Load saved YAML configurations for each job (in hydra.job.output_subdir)
        job_cfgs = [
            load_from_yaml(f)
            for f in sorted(self.working_dir.glob(f"**/*/{config_dir}/config.yaml"))
        ]

        # Load saved YAML overrides for each job (in hydra.job.output_subdir)
        job_overrides = [
            list(load_from_yaml(f))
            for f in sorted(self.working_dir.glob(f"**/*/{config_dir}/overrides.yaml"))
        ]

        # Load metrics for each job
        job_metrics = [
            tr.load(f)
            for f in sorted(self.working_dir.glob(f"**/*/{metrics_filename}"))
        ]

        self.cfgs = job_cfgs
        self.metrics, self.workflow_overrides = _load_metrics(
            job_overrides, job_metrics, workflow_params
        )
        return self

    def to_dataframe(self):
        """Convert workflow data to Pandas DataFrame."""
        import pandas as pd

        d = {}
        d.update(self.metrics)
        d.update(self.workflow_overrides)
        return pd.DataFrame(d)

    def to_xarray(self, non_multirun_params_as_singleton_dims: bool = False):
        """Convert workflow data to xarray Dataset.

        Parameters
        ----------
        non_multirun_params_as_singleton_dims : bool, optional (default=False)
            If `True` then non-multirun entries from `workflow_overrides` will be
            included as length-1 dimensions in the xarray. Useful for merging/
            concatenation with other Datasets

        Returns
        -------
        results : xarray.Dataset
            A dataset whose dimensions and coordinate-values are determined by the
            quantities over which the multi-run was performed. The data variables correspond to the named results returned by the jobs."""
        import xarray as xr

        orig_coords = {
            k: (v if _non_str_sequence(v) else [v])
            for k, v in self.multirun_task_overrides.items()
            if non_multirun_params_as_singleton_dims or _non_str_sequence(v)
        }

        # non-multirun overrides
        attrs = {
            k: v
            for k, v in self.multirun_task_overrides.items()
            if not _non_str_sequence(v)
        }

        # we will add additional coordinates as-needed for multi-dim metrics
        coords: Dict[str, Any] = orig_coords.copy()
        shape = tuple(len(v) for v in coords.values())

        data = {}
        for k, v in self.metrics.items():
            datum = np.asarray(v).reshape(shape + np.asarray(v[0]).shape)
            n = 0
            for n in range(datum.ndim - len(coords)):
                # Create additional arbitrary coordinates as-needed for non-scalar
                # metrics
                coords[f"{k}_dim{n}"] = np.arange(datum.shape[len(orig_coords) + n])

            data[k] = (
                list(orig_coords)
                + [f"{k}_dim{i}" for i in range(datum.ndim - len(shape))],
                datum,
            )
        return xr.Dataset(coords=coords, data_vars=data, attrs=attrs)


class RobustnessCurve(MultiRunMetricsWorkflow):
    """Abstract class for workflows that measure performance for different perturbation values.

    This workflow requires and uses parameter `epsilon` as the configuration option for varying the perturbation.

    See Also
    --------
    MultiRunMetricsWorkflow
    """

    def run(
        self,
        *,
        epsilon: Union[str, Sequence[float]],
        working_dir: Optional[str] = None,
        sweeper: Optional[str] = None,
        launcher: Optional[str] = None,
        overrides: Optional[List[str]] = None,
        **workflow_overrides: Union[str, int, float, bool, multirun, hydra_list],
    ):
        """Run the experiment for varying value `epsilon`.

        Parameters
        ----------
        epsilon: str | Sequence[float]
            The configuration parameter for the perturbation.  Unlike Hydra overrides,
            this parameter can be a list of floats that will be converted into a
            multirun sequence override for Hydra.

        working_dir: str (default: None, the Hydra default will be used)
            The directory to run the experiment in.  This value is used for
            setting `hydra.sweep.dir`.

        sweeper: str | None (default: None)
            The configuration name of the Hydra Sweeper to use (i.e., the override for
            `hydra/sweeper=sweeper`)

        launcher: str | None (default: None)
            The configuration name of the Hydra Launcher to use (i.e., the override for
            `hydra/launcher=launcher`)

        overrides: List[str] | None (default: None)
            Parameter overrides not considered part of the workflow parameter set.
            This is helpful for filtering out parameters stored in
            `self.workflow_overrides`.

        **workflow_overrides: dict | str | int | float | bool | multirun | hydra_list
            These parameters represent the values for configurations to use for the
            experiment.

            These values will be appeneded to the `overrides` for the Hydra job.
        """

        if not isinstance(epsilon, str):
            epsilon = multirun(epsilon)

        return super().run(
            working_dir=working_dir,
            sweeper=sweeper,
            launcher=launcher,
            overrides=overrides,
            **workflow_overrides,
            # for multiple multi-run params, epsilon should fastest-varying param;
            # i.e. epsilon should be the trailing dim in the multi-dim array of results
            epsilon=epsilon,
        )

    def to_dataframe(self):
        """Convert workflow data to Pandas DataFrame."""
        return super().to_dataframe().sort_values("epsilon")

    def to_xarray(self, non_multirun_params_as_singleton_dims: bool = False):
        """Convert workflow data to xarray Dataset.

        Parameters
        ----------
        non_multirun_params_as_singleton_dims : bool, optional (default=False)
            If `True` then non-multirun entries from `workflow_overrides` will be
            included as length-1 dimensions in the xarray. Useful for merging/
            concatenation with other Datasets

        Returns
        -------
        results : xarray.Dataset
            A dataset whose dimensions and coordinate-values are determined by the
            quantities over which the multi-run was performed. The data variables correspond to the named results returned by the jobs."""
        return (
            super()
            .to_xarray(
                non_multirun_params_as_singleton_dims=non_multirun_params_as_singleton_dims
            )
            .sortby("epsilon")
        )

    def plot(
        self,
        metric: str,
        ax=None,
        group: Optional[str] = None,
        save_filename: Optional[str] = None,
        non_multirun_params_as_singleton_dims: bool = False,
        **kwargs,
    ) -> None:
        """Plot metrics versus `epsilon`.

        Using the `xarray.Dataset` from `to_xarray`, plot the metrics
        against the workflow perturbation parameters.

        Parameters
        ----------
        metric: str
            The metric saved

        ax: Axes | None (default: None)
            If not `None`, the matplotlib.Axes to use for plotting.

        group: str | None (default: None)
            Needed if other parameters besides `epsilon` were varied.

        save_filename: str | None (default: None)
            If not `None` save figure to the filename provided.

        non_multirun_params_as_singleton_dims : bool, optional (default=False)
            If `True` then non-multirun entries from `workflow_overrides` will be
            included as length-1 dimensions in the xarray. Useful for merging/
            concatenation with other Datasets

        **kwargs: Any
            Additional arguments passed to `xarray.plot`.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        xdata = self.to_xarray(
            non_multirun_params_as_singleton_dims=non_multirun_params_as_singleton_dims
        )
        if group is None:
            plots = xdata[metric].plot.line(x="epsilon", ax=ax, **kwargs)

        else:
            # TODO: xarray.groupby doesn't support multidimensional grouping
            dg = xdata.groupby(group)
            plots = [
                grp[metric].plot(x="epsilon", label=name, ax=ax, **kwargs)
                for name, grp in dg
            ]

        if save_filename is not None:
            plt.savefig(save_filename)

        return plots


def _load_metrics(
    job_overrides: List[List[str]],
    job_metrics: List[Dict[str, Any]],
    workflow_params: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
    workflow_overrides = defaultdict(list)
    metrics = defaultdict(list)
    for task_overrides, task_metrics in zip(job_overrides, job_metrics):
        for override in task_overrides:
            k, v = override.split("=")
            # remove any hydra override prefix
            param = k.split("+")[-1]
            if workflow_params is None or param in workflow_params:
                v = _num_from_string(v)
                workflow_overrides[param].append(v)

        for k, v in task_metrics.items():
            # get item if it's a single element array
            if isinstance(v, list) and len(v) == 1:
                v = v[0]

            metrics[k].append(v)

    return metrics, workflow_overrides
