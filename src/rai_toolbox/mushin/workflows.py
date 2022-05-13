# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod, abstractstaticmethod
from collections import UserList, defaultdict
from pathlib import Path
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import torch as tr
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.utils import JobReturn
from hydra_zen import launch, load_from_yaml, make_config
from hydra_zen._launch import _NotSet
from typing_extensions import Self, TypeAlias, TypeGuard

from rai_toolbox._utils import value_check

from .hydra import zen

LoadedValue: TypeAlias = Union[str, int, float]

__all__ = [
    "BaseWorkflow",
    "RobustnessCurve",
    "MultiRunMetricsWorkflow",
    "multirun",
    "hydra_list",
]


T = TypeVar("T", List[Any], Tuple[Any])


class multirun(UserList):
    """Signals that a sequence is to be iterated over in a multirun"""

    pass


class hydra_list(UserList):
    """Signals that a sequence is provided as a single configured value (i.e. it is not
    to be iterated over during a multirun)"""

    pass


def _sort_x_by_k(x: T, k: Iterable[Any]) -> T:
    k = tuple(k)
    assert len(x) == len(k)
    sorted_, _ = zip(*sorted(zip(x, k), key=lambda x: x[1]))
    return type(x)(sorted_)


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
    jobs: Union[List[JobReturn], List[Any], JobReturn]
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

    def _parse_overrides(self, overrides: List[str]) -> Dict[str, Any]:
        parser = OverridesParser.create()
        parsed_overrides = parser.parse_overrides(overrides=overrides)

        output = {}
        for override in parsed_overrides:
            if override.is_sweep_override():
                param_name = override.get_key_element()
                val = [
                    _num_from_string(val) for val in override.sweep_string_iterator()
                ]
            else:
                param_name = override.get_key_element()
                val = override.get_value_element_as_str()
                val = _num_from_string(val)

            param_name = param_name.split("+")[-1]
            output[param_name] = val

        return output

    @property
    def multirun_task_overrides(
        self,
    ) -> Dict[str, Union[LoadedValue, Sequence[LoadedValue]]]:
        # e.g. {'epsilon': [1.0, 2.0, 3.0], "foo": "apple"}
        if not self._multirun_task_overrides:
            overrides = load_from_yaml(
                self.working_dir / "multirun.yaml"
            ).hydra.overrides.task

            output = self._parse_overrides(overrides)
            self._multirun_task_overrides.update(output)

        return self._multirun_task_overrides

    @abstractstaticmethod
    def evaluation_task(*args: Any, **kwargs: Any) -> Any:
        """User-defined evaluation task to run the workflow.

        Arguments will be instantiated configuration variables.  For example,
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
        version_base: Optional[Union[str, Type[_NotSet]]] = _NotSet,
        to_dictconfig: bool = False,
        config_name: str = "rai_workflow",
        job_name: str = "rai_workflow",
        with_log_configuration: bool = True,
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

        version_base : Optional[str], optional (default=_NotSet)
            Available starting with Hydra 1.2.0.
            - If the `version_base parameter` is not specified, Hydra 1.x will use defaults compatible with version 1.1. Also in this case, a warning is issued to indicate an explicit version_base is preferred.
            - If the `version_base parameter` is `None`, then the defaults are chosen for the current minor Hydra version. For example for Hydra 1.2, then would imply `config_path=None` and `hydra.job.chdir=False`.
            - If the `version_base` parameter is an explicit version string like "1.1", then the defaults appropriate to that version are used.

        to_dictconfig: bool (default: False)
            If ``True``, convert a ``dataclasses.dataclass`` to a ``omegaconf.DictConfig``. Note, this
            will remove Hydra's cabability for validation with structured configurations.

        config_name : str (default: "rai_workflow")
            Name of the stored configuration in Hydra's ConfigStore API.

        job_name : str (default: "rai_workflow")
            Name of job for logging.

        with_log_configuration : bool (default: True)
            If ``True``, enables the configuration of the logging subsystem from the loaded config.

        **workflow_overrides: str | int | float | bool | multirun | hydra_list | dict
            These parameters represent the values for configurations to use for the
            experiment.

            Passing `param=multirun([1, 2, 3])` will perform a multirun over those
            three param values, whereas passing `param=hydra_list([1, 2, 3])` will
            pass the entire list as a single input.

            These values will be appended to the `overrides` for the Hydra job.
        """
        launch_overrides = []

        if overrides is not None:
            launch_overrides.extend(overrides)

        if working_dir is not None:
            launch_overrides.append(f"hydra.sweep.dir={working_dir}")
            self.working_dir = Path(working_dir).resolve()

        if sweeper is not None:
            launch_overrides.append(f"hydra/sweeper={sweeper}")

        if launcher is not None:
            launch_overrides.append(f"hydra/launcher={launcher}")

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

            launch_overrides.append(f"{prefix}{k}={v}")

        # Run a Multirun over epsilons
        jobs = launch(
            self.eval_task_cfg,
            zen(self.evaluation_task),
            overrides=launch_overrides,
            multirun=True,
            version_base=version_base,
            to_dictconfig=to_dictconfig,
            config_name=config_name,
            job_name=job_name,
            with_log_configuration=with_log_configuration,
        )

        if isinstance(jobs, List) and len(jobs) == 1:
            # hydra returns [jobs]
            jobs = jobs[0]
            _job_nums = [j.hydra_cfg.hydra.job.num for j in jobs]
            jobs = _sort_x_by_k(
                jobs, _job_nums
            )  # ensure jobs are always sorted by job-num

        self.jobs = jobs
        self.jobs_post_process()

    @abstractmethod
    def jobs_post_process(self):
        """Method to extract attributes and metrics relevant to the workflow."""

    def plot(self, **kwargs) -> None:
        """Plot workflow metrics."""
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

    >>> class LocalRobustness(MultiRunMetricsWorkflow):
    ...     @staticmethod
    ...     def evaluation_task(epsilon: float, scale: float) -> dict:
    ...         epsilon *= scale
    ...         val = 100 - epsilon**2
    ...         result = dict(accuracies=val+2, loss=epsilon**2)
    ...         tr.save(result, "test_metrics.pt")
    ...         return result

    We'll run this workflow for six total configurations of three `epsilon` values and
    two `scale` values. This will launch a Hydra multirun job and aggregate the results.

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
    coordinates reflect the multirun parameters that were varied, and whose
    data-variables are our recorded metrics: "accuracies" and "loss".

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

    # TODO: add target_job_dirs example
    #      Document .swap_dims({"job_dir": <...>}) and .set_index(job_dir=[...]).unstack("job_dir")
    #      for re-indexing based on overrides values

    _JOBDIR_NAME: str = "job_dir"
    _target_dir_multirun_overrides: Optional[DefaultDict[str, List[Any]]] = None
    output_subdir: Optional[str] = None

    # List of all the dirs that the multirun writes to; sorted by job-num
    multirun_working_dirs: Optional[List[Path]] = None

    @abstractstaticmethod
    def evaluation_task(*args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore
        """Abstract `staticmethod` for users to define the evalultion task"""

    def run(
        self,
        *,
        working_dir: Optional[str] = None,
        sweeper: Optional[str] = None,
        launcher: Optional[str] = None,
        overrides: Optional[List[str]] = None,
        version_base: Optional[Union[str, Type[_NotSet]]] = _NotSet,
        target_job_dirs: Optional[Sequence[Union[str, Path]]] = None,
        to_dictconfig: bool = False,
        config_name: str = "rai_workflow",
        job_name: str = "rai_workflow",
        with_log_configuration: bool = True,
        **workflow_overrides: Union[str, int, float, bool, dict, multirun, hydra_list],
    ):
        # TODO: add docs

        if target_job_dirs is not None:
            if isinstance(target_job_dirs, str):
                raise TypeError(
                    f"`target_job_dirs` must be a sequence of pathlike objects, got: {target_job_dirs}"
                )
            value_check("target_job_dirs", target_job_dirs, type_=Sequence)

            target_job_dirs = [Path(s).resolve() for s in target_job_dirs]
            for d in target_job_dirs:
                if not d.is_dir() or not d.exists():
                    # TODO: check not only that dir exists but that it contains
                    #       correct config file (e.g. .hydra)
                    raise FileNotFoundError()
            target_job_dirs = multirun([str(s) for s in target_job_dirs])
            workflow_overrides[self._JOBDIR_NAME] = target_job_dirs

        return super().run(
            working_dir=working_dir,
            sweeper=sweeper,
            launcher=launcher,
            overrides=overrides,
            version_base=version_base,
            to_dictconfig=to_dictconfig,
            config_name=config_name,
            job_name=job_name,
            with_log_configuration=with_log_configuration,
            **workflow_overrides,
        )

    @property
    def target_dir_multirun_overrides(self) -> Dict[str, List[Any]]:
        """
        For a multirun that sweeps over the target directories of a
        previous multirun, `target_dir_multirun_overrides` provides
        the flattened overrides for that previous run.

        Examples
        --------
        >>> class A(MultiRunMetricsWorkflow):
        ...     @staticmethod
        ...     def evaluation_task(value: float, scale: float):
        ...         pass
        ...

        >>> class B(MultiRunMetricsWorkflow):
        ...     @staticmethod
        ...     def evaluation_task():
        ...         pass

        >>> a = A()
        >>> a.run(value=multirun([-1.0, 0.0, 1.0]), scale=multirun([11.0, 9.0]))
        [2022-05-13 17:19:51,497][HYDRA] Launching 6 jobs locally
        [2022-05-13 17:19:51,497][HYDRA] 	#0 : +value=-1.0 +scale=11.0
        [2022-05-13 17:19:51,555][HYDRA] 	#1 : +value=-1.0 +scale=9.0
        [2022-05-13 17:19:51,613][HYDRA] 	#2 : +value=0.0 +scale=11.0
        [2022-05-13 17:19:51,671][HYDRA] 	#3 : +value=0.0 +scale=9.0
        [2022-05-13 17:19:51,729][HYDRA] 	#4 : +value=1.0 +scale=11.0
        [2022-05-13 17:19:51,787][HYDRA] 	#5 : +value=1.0 +scale=9.0

        >>> b = B()
        >>> b.run(target_job_dirs=a.multirun_working_dirs)
        [2022-05-13 17:19:59,900][HYDRA] Launching 6 jobs locally
        [2022-05-13 17:19:59,900][HYDRA] 	#0 : +job_dir=/home/scratch/multirun/0
        [2022-05-13 17:19:59,958][HYDRA] 	#1 : +job_dir=/home/scratch/multirun/1
        [2022-05-13 17:20:00,015][HYDRA] 	#2 : +job_dir=/home/scratch/multirun/2
        [2022-05-13 17:20:00,073][HYDRA] 	#3 : +job_dir=/home/scratch/multirun/3
        [2022-05-13 17:20:00,130][HYDRA] 	#4 : +job_dir=/home/scratch/multirun/4
        [2022-05-13 17:20:00,188][HYDRA] 	#5 : +job_dir=/home/scratch/multirun/5

        >>> b.target_dir_multirun_overrides
        {'value': [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0],
         'scale': [11.0, 9.0, 11.0, 9.0, 11.0, 9.0]}"""
        if self._target_dir_multirun_overrides is not None:
            return dict(self._target_dir_multirun_overrides)
        assert self.output_subdir is not None

        multirun_cfg = self.working_dir / "multirun.yaml"
        self._target_dir_multirun_overrides = defaultdict(list)

        overrides = load_from_yaml(multirun_cfg).hydra.overrides.task
        self.overrides = overrides

        dirs = []

        for o in overrides:
            k, v = o.split("=")
            k = k.replace("+", "")
            if k == self._JOBDIR_NAME:
                dirs = v.split(",")
                break

        for d in dirs:
            overrides: List[str] = list(
                load_from_yaml(Path(d) / f"{self.output_subdir}/overrides.yaml")
            )
            output = self._parse_overrides(overrides)

            for ko, vo in output.items():
                self._target_dir_multirun_overrides[ko].append(vo)
        return dict(self._target_dir_multirun_overrides)

    def jobs_post_process(self):
        assert len(self.jobs) > 0
        # TODO: Make protocol type for JobReturn
        assert isinstance(self.jobs[0], JobReturn)
        self.jobs: List[JobReturn]

        self.multirun_working_dirs = []

        for job in self.jobs:
            _hydra_cfg = job.hydra_cfg
            assert _hydra_cfg is not None
            assert job.working_dir is not None
            _cwd = _hydra_cfg.hydra.runtime.cwd
            working_dir = Path(_cwd) / job.working_dir
            self.multirun_working_dirs.append(working_dir)

        # set working directory of this workflow
        self.working_dir = self.multirun_working_dirs[0].parent

        hydra_cfg = self.jobs[0].hydra_cfg
        assert hydra_cfg is not None
        self.output_subdir = hydra_cfg.hydra.output_subdir

        # extract configs, overrides, and metrics
        self.cfgs = [j.cfg for j in self.jobs]
        job_metrics = [j.return_value for j in self.jobs]
        self.metrics = self._process_metrics(job_metrics)

    def _process_metrics(self, job_metrics) -> Dict[str, Any]:
        metrics = defaultdict(list)
        for task_metrics in job_metrics:
            if task_metrics is None:
                continue
            for k, v in task_metrics.items():
                # get item if it's a single element array
                if isinstance(v, list) and len(v) == 1:
                    v = v[0]

                metrics[k].append(v)
        return metrics

    def load_from_dir(
        self: Self,
        working_dir: Union[Path, str],
        metrics_filename: Union[str, None],
    ) -> Self:
        """Loading workflow job data from a given working directory. The workflow
        is loaded in-place and "self" is returned by this method.

        Parameters
        ----------
        working_dir: str | Path
            The base working directory of the experiment. It is expected
            that subdirectories within this working directory will contain
            individual Hydra jobs data (yaml configurations) and saved metrics files.

        metrics_filename: Union[str, None]
            The filename used to save metrics for each individual Hydra job. This can
            be a search pattern as well since this is appended to

                `Path(working_dir).glob("**/*/<metrics_filename")`

            If `None`, no metrics are loaded.

        Returns
        -------
        loaded_workflow : Self
        """
        self.working_dir = Path(working_dir).resolve()
        self.output_subdir = load_from_yaml(
            self.working_dir / "multirun.yaml"
        ).hydra.output_subdir

        self.multirun_working_dirs = list(
            (x.parent for x in self.working_dir.glob(f"**/*/{self.output_subdir}"))
        )

        # ensure working dirs are sorted by job num
        _job_nums = (
            load_from_yaml(dir_ / f"{self.output_subdir}/hydra.yaml").hydra.job.num
            for dir_ in self.multirun_working_dirs
        )

        self.multirun_working_dirs = _sort_x_by_k(self.multirun_working_dirs, _job_nums)

        self.cfgs = []
        job_metrics = []
        for dir_ in self.multirun_working_dirs:
            if metrics_filename is None:
                break
            # Ensure we load saved YAML configurations for each job (in hydra.job.output_subdir)
            cfg_file = dir_ / f"{self.output_subdir}/config.yaml"
            assert cfg_file.exists(), cfg_file
            self.cfgs.append(load_from_yaml(cfg_file))
            job_metrics.append(tr.load(dir_ / metrics_filename))

        self.metrics = self._process_metrics(job_metrics)

        return self

    def to_xarray(
        self,
        coord_from_metrics: Optional[str] = None,
        non_multirun_params_as_singleton_dims: bool = False,
    ):
        """Convert workflow data to xarray Dataset.

        Parameters
        ----------
        coord_from_metrics: str | None (default: None)
            If not `None` defines the metric key to use as a coordinate
            in the `Dataset`.  This function assumes that this coordinate
            represents the leading dimension for all data-variables.

        non_multirun_params_as_singleton_dims : bool, optional (default=False)
            If `True` then non-multirun entries from `workflow_overrides` will be
            included as length-1 dimensions in the xarray. Useful for merging/
            concatenation with other Datasets

        Returns
        -------
        results : xarray.Dataset
            A dataset whose dimensions and coordinate-values are determined by the
            quantities over which the multi-run was performed. The data variables
            correspond to the named results returned by the jobs."""
        import xarray as xr

        orig_coords = {
            k: (v if _non_str_sequence(v) else [v])
            for k, v in self.multirun_task_overrides.items()
            if non_multirun_params_as_singleton_dims or _non_str_sequence(v)
        }

        metric_coords = {}
        if coord_from_metrics:
            if coord_from_metrics not in self.metrics:
                raise ValueError(
                    f"key `{coord_from_metrics}` not in metrics (available: {list(self.metrics.keys())})"
                )

            v = self.metrics[coord_from_metrics]
            if np.asarray(v).ndim > 1:  # pragma: no cover
                # assume this coord was repeated across experiments, e.g., "epochs"
                v = v[0]
            metric_coords[coord_from_metrics] = v

        # non-multirun overrides
        attrs = {
            k: v
            for k, v in self.multirun_task_overrides.items()
            if not _non_str_sequence(v)
        }

        # we will add additional coordinates as-needed for multi-dim metrics
        coords: Dict[str, Any] = orig_coords.copy()
        shape = tuple(len(v) for k, v in coords.items())

        data = {}
        for k, v in self.metrics.items():
            if coord_from_metrics and k == coord_from_metrics:
                continue

            datum = np.asarray(v).reshape(shape + np.asarray(v[0]).shape)

            k_coords = list(orig_coords)
            for n in range(datum.ndim - len(orig_coords)):

                if coord_from_metrics and n < len(metric_coords):
                    # Assume the first coordinate of the metric is the metric coordinate dimension
                    k_coords += list(metric_coords.keys())
                    for mk, mv in metric_coords.items():
                        coords[mk] = mv
                else:
                    # Create additional arbitrary coordinates as-needed for non-scalar
                    # metrics
                    k_coords += [f"{k}_dim{n}"]
                    coords[f"{k}_dim{n}"] = np.arange(datum.shape[len(orig_coords) + n])

            data[k] = (k_coords, datum)

        coords.update(metric_coords)
        out = xr.Dataset(coords=coords, data_vars=data, attrs=attrs)

        if self._JOBDIR_NAME in set(out.coords):
            exp_dir = out.coords[self._JOBDIR_NAME]
            coords = {}
            for k, v in self.target_dir_multirun_overrides.items():
                if len(v) == len(exp_dir):
                    uv = list(set(np.unique(v)))
                    if len(uv) > 1 or non_multirun_params_as_singleton_dims:
                        coords[k] = ([self._JOBDIR_NAME], v)
            out = out.assign_coords(coords)
        return out


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
        target_job_dirs: Optional[Sequence[Union[str, Path]]] = None,  # TODO: add docs
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
            target_job_dirs=target_job_dirs,
            epsilon=epsilon,
        )

    def to_xarray(
        self,
        coord_from_metrics: Optional[str] = None,
        non_multirun_params_as_singleton_dims: bool = False,
    ):
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
                coord_from_metrics=coord_from_metrics,
                non_multirun_params_as_singleton_dims=non_multirun_params_as_singleton_dims,
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
