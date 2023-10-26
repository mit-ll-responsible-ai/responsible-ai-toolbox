# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from collections import defaultdict
from inspect import getattr_static
from pathlib import Path
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Mapping,
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
from hydra_zen import hydra_list, launch, load_from_yaml, make_config, multirun, zen
from hydra_zen._compatibility import HYDRA_VERSION
from hydra_zen._launch import _NotSet
from typing_extensions import Self, TypeAlias, TypeGuard

from rai_toolbox._utils import value_check

LoadedValue: TypeAlias = Union[str, int, float, bool, List[Any], Dict[str, Any]]

__all__ = [
    "BaseWorkflow",
    "RobustnessCurve",
    "MultiRunMetricsWorkflow",
]


T = TypeVar("T", List[Any], Tuple[Any])
T1 = TypeVar("T1")


_VERSION_BASE_DEFAULT = _NotSet if HYDRA_VERSION < (1, 2, 0) else "1.1"


def _sort_x_by_k(x: T, k: Iterable[Any]) -> T:
    k = tuple(k)
    assert len(x) == len(k)
    sorted_, _ = zip(*sorted(zip(x, k), key=lambda x: x[1]))
    return type(x)(sorted_)


def _identity(x: T1) -> T1:
    return x


def _task_calls(
    pre_task: Callable[[Any], None], task: Callable[[Any], T1]
) -> Callable[[Any], T1]:
    def wrapped(cfg: Any) -> T1:
        pre_task(cfg)
        return task(cfg)

    return wrapped


class BaseWorkflow:
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

    working_dir: Optional[pathlib.Path]
        The working directory of the experiment defined by Hydra's sweep directory
        (`hydra.sweep.dir`).
    """

    _REQUIRED_STATIC_METHODS = ("task", "pre_task")

    cfgs: List[Any]
    metrics: Dict[str, List[Any]]
    workflow_overrides: Dict[str, Any]
    jobs: Union[List[JobReturn], List[Any], JobReturn]

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
        self._working_dir = None

    @property
    def working_dir(self) -> Path:
        if self._working_dir is None:
            raise ValueError("`self.working_dir` must be set.")

        return self._working_dir

    @working_dir.setter
    def working_dir(self, path: Union[str, Path]):
        if isinstance(path, str):
            path = Path(path)
        value_check("path", path, type_=Path)
        path = path.resolve()

        if not path.is_dir():
            raise FileNotFoundError(
                f"`path` point to an existing directory, got {path}"
            )

        self._working_dir = path

    @staticmethod
    def _parse_overrides(
        overrides,
    ) -> Dict[str, Union[LoadedValue, Sequence[LoadedValue]]]:
        parser = OverridesParser.create()
        parsed_overrides = parser.parse_overrides(overrides=overrides)

        output = {}

        for override in parsed_overrides:
            param_name = override.get_key_element()
            val = override.value()
            if override.is_sweep_override():
                val = multirun(val.list)  # type: ignore

            param_name = param_name.split("+")[-1]
            output[param_name] = val

        return output

    @property
    def multirun_task_overrides(
        self,
    ) -> Dict[str, Union[LoadedValue, Sequence[LoadedValue]]]:
        """Returns override param-name -> value.

        A sequence of overrides associated with a multirun will
        be stored in a `rai_toolbox.mushin.multirun` list. This
        enables one to distinguish this from an override whose sole
        value was a list of values.

        Returns
        -------
        multirun_task_overrides: Dict[str, LoadedValue | Sequence[LoadedValue]]

        Examples
        --------
        >>> from rai_toolbox.mushin import multirun, hydra_list
        >>>
        >>> class WorkFlow(MultiRunMetricsWorkflow):
        ...     @staticmethod
        ...     def task(*args, **kwargs):
        ...         return None
        >>>
        >>> wf = WorkFlow()
        >>> wf.run(foo=hydra_list(["val"]), bar=multirun(["a", "b"]), apple=1)
        >>> wf.multirun_task_overrides
        {'foo': ['val'], 'bar': multirun(['a', 'b']), 'apple': 1}
        """
        if not self._multirun_task_overrides:
            overrides = load_from_yaml(
                self.working_dir / "multirun.yaml"
            ).hydra.overrides.task

            output = self._parse_overrides(overrides)
            self._multirun_task_overrides = output

        return self._multirun_task_overrides

    @staticmethod
    def pre_task(*args: Any, **kwargs: Any) -> None:
        """Called prior to `task`

        This can be useful for doing things like setting random seeds,
        which must occur prior to instantiating objects for the evaluation
        task.

        Notes
        -----
        This function is automatically wrapped by `zen`, which is responsible
        for parsing the function's signature and then extracting and instantiating
        the corresponding fields from a Hydra config object – passing them to the
        function. This behavior can be modified by `self.run(pre_task_fn_wrapper=...)`
        """

    @staticmethod
    def task(*args: Any, **kwargs: Any) -> Any:
        """User-defined task that is run by the workflow. This should be
        a static method.

        Arguments will be instantiated configuration variables.  For example,
        if the the workflow configuration is structured as::

            ├── eval_task_cfg
            │    ├── trainer
            |    ├── module
            |    ├── another_config

        The inputs to `task` can be any of the three configurations:
        `trainer`, `module`, or `another_config` such as::

            @staticmethod
            def task(trainer: Trainer, module: LightningModule) -> None:
                trainer.fit(module)

        Notes
        -----
        This function is automatically wrapped by `zen`, which is responsible
        for parsing the function's signature and then extracting and instantiating
        the corresponding fields from a Hydra config object – passing them to the
        function. This behavior can be modified by `self.run(task_fn_wrapper=...)`
        """
        raise NotImplementedError()

    def validate(self, include_pre_task: bool = True):
        """Validates that the configuration will execute with the user-defined evaluation task"""
        if include_pre_task:
            zen(self.pre_task).validate(self.eval_task_cfg)

        zen(self.task).validate(self.eval_task_cfg)

    def run(
        self,
        *,
        working_dir: Optional[str] = None,
        sweeper: Optional[str] = None,
        launcher: Optional[str] = None,
        overrides: Optional[List[str]] = None,
        task_fn_wrapper: Union[
            Callable[[Callable[..., T1]], Callable[[Any], T1]], None
        ] = zen,
        pre_task_fn_wrapper: Union[
            Callable[[Callable[..., None]], Callable[[Any], None]], None
        ] = zen,
        version_base: Optional[Union[str, Type[_NotSet]]] = _VERSION_BASE_DEFAULT,
        to_dictconfig: bool = False,
        config_name: str = "rai_workflow",
        job_name: str = "rai_workflow",
        with_log_configuration: bool = True,
        **workflow_overrides: Union[str, int, float, bool, dict, multirun, hydra_list],
    ):
        """Run the experiment.

        Individual workflows can explicitly define `workflow_overrides` to improve
        readability and undstanding of what parameters are expected for a particular
        workflow.

        Parameters
        ----------
        task_fn_wrapper: Callable[[Callable[..., T1]], Callable[[Any], T1]] | None, optional (default=rai_toolbox.mushin.zen)
            A wrapper applied to `self.task` prior to launching the task.
            The default wrapper is `rai_toolbox.mushin.zen`. Specify `None` for no
            wrapper to be applied.

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

        version_base : Optional[str], optional (default=1.1)
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

        for _name in self._REQUIRED_STATIC_METHODS:
            if _name == "task" and hasattr(self, "evaluation_task"):
                # TODO: remove when evaluation_task support is removed
                _name = "evaluation_task"
            if not isinstance(getattr_static(self, _name), staticmethod):
                raise TypeError(
                    f"{type(self).__name__}.{_name} must be a static method"
                )

        if task_fn_wrapper is None:
            task_fn_wrapper = _identity

        if pre_task_fn_wrapper is None:
            pre_task_fn_wrapper = _identity

        # Run a Multirun over epsilons
        jobs = launch(
            self.eval_task_cfg,
            _task_calls(
                pre_task=pre_task_fn_wrapper(self.pre_task),
                task=task_fn_wrapper(self.task),
            ),
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
            # ensure jobs are always sorted by job-num
            jobs = _sort_x_by_k(jobs, _job_nums)

        self.jobs = jobs
        self.jobs_post_process()

    def jobs_post_process(self):  # pragma: no cover
        """Method to extract attributes and metrics relevant to the workflow."""
        raise NotImplementedError()

    def plot(self, **kwargs) -> None:  # pragma: no cover
        """Plot workflow metrics."""
        raise NotImplementedError()

    def to_xarray(self):  # pragma: no cover
        """Convert workflow data to xArray Dataset or DataArray."""
        raise NotImplementedError()


def _non_str_sequence(x: Any) -> TypeGuard[Sequence[Any]]:
    return isinstance(x, Sequence) and not isinstance(x, str)


def _coerce_list_of_arraylikes(v: List[Any]):
    if v and hasattr(v[0], "__array__"):
        return [np.asarray(i) for i in v]
    return v


class MultiRunMetricsWorkflow(BaseWorkflow):
    """Abstract class for workflows that record metrics using Hydra multirun.

    This workflow creates subdirectories of multirun experiments using Hydra.  These directories
    contain the Hydra YAML configuration and any saved metrics file (defined by the evaluationf task)::

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
    ...     def task(epsilon: float, scale: float) -> dict:
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

    def __init__(self, eval_task_cfg=None, working_dir: Optional[Path] = None) -> None:
        super().__init__(eval_task_cfg)
        self._working_dir = working_dir

        if self._working_dir is not None:
            self.load_from_dir(self.working_dir, metrics_filename=None)

    # TODO: add target_job_dirs example
    #      Document .swap_dims({"job_dir": <...>}) and .set_index(job_dir=[...]).unstack("job_dir")
    #      for re-indexing based on overrides values

    _JOBDIR_NAME: str = "job_dir"
    _target_dir_multirun_overrides: Optional[DefaultDict[str, List[Any]]] = None
    output_subdir: Optional[str] = None

    # List of all the dirs that the multirun writes to; sorted by job-num
    multirun_working_dirs: Optional[List[Path]] = None

    @staticmethod
    def task(*args: Any, **kwargs: Any) -> Mapping[str, Any]:  # pragma: no cover
        """Abstract `staticmethod` for users to define the task that is configured and
        launched by the workflow"""
        raise NotImplementedError()

    @staticmethod
    def metric_load_fn(file_path: Path) -> Mapping[str, Any]:
        """Loads a metric file and returns a dictionary of metric-name -> metric-value
        mappings.

        The default metric load function is `torch.load`.

        Parameters
        ----------
        file_path : Path

        Returns
        -------
        named_metrics : Mapping[str, Any]
            metric-name -> metric-value(s)

        Examples
        --------
        Designing a workflow that uses the `pickle` module to save and load
        metrics

        >>> from rai_toolbox.mushin import MultiRunMetricsWorkflow, multirun
        >>> import pickle
        >>>
        >>> class PickledWorkFlow(MultiRunMetricsWorkflow):
        ...     @staticmethod
        ...     def metric_load_fn(file_path: Path):
        ...         with file_path.open("rb") as f:
        ...             return pickle.load(f)
        ...
        ...     @staticmethod
        ...     def task(a, b):
        ...         with open("./metrics.pkl", "wb") as f:
        ...             pickle.dump(dict(a=a, b=b), f)
        >>>
        >>> wf = PickleWorkFlow()
        >>> wf.run(a=multirun([1, 2, 3]), b=False)
        >>> wf.load_metrics("metrics.pkl")
        >>> wf.metrics
        dict(a=[1, 2, 3], b=[False, False, False])"""
        return tr.load(file_path)

    def run(
        self,
        *,
        task_fn_wrapper: Union[
            Callable[[Callable[..., T1]], Callable[[Any], T1]], None
        ] = zen,
        pre_task_fn_wrapper: Union[
            Callable[[Callable[..., None]], Callable[[Any], None]], None
        ] = zen,
        working_dir: Optional[str] = None,
        sweeper: Optional[str] = None,
        launcher: Optional[str] = None,
        overrides: Optional[List[str]] = None,
        version_base: Optional[Union[str, Type[_NotSet]]] = _VERSION_BASE_DEFAULT,
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
                    raise FileNotFoundError(
                        f"The specified target directory – {d} – does not exist."
                    )
            target_job_dirs = multirun([str(s) for s in target_job_dirs])
            workflow_overrides[self._JOBDIR_NAME] = target_job_dirs

        return super().run(
            working_dir=working_dir,
            sweeper=sweeper,
            launcher=launcher,
            overrides=overrides,
            task_fn_wrapper=task_fn_wrapper,
            pre_task_fn_wrapper=pre_task_fn_wrapper,
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
        ...     def task(value: float, scale: float):
        ...         pass
        ...

        >>> class B(MultiRunMetricsWorkflow):
        ...     @staticmethod
        ...     def task():
        ...         pass

        >>> a = A()
        >>> a.run(value=multirun([-1.0, 0.0, 1.0]), scale=multirun([11.0, 9.0]))
        [2022-05-13 17:19:51,497][HYDRA] Launching 6 jobs locally
        [2022-05-13 17:19:51,497][HYDRA] 	#0 : +value=-1.0 +scale=11.0
        [2022-05-13 17:19:51,555][HYDRA] 	#1 : +value=-1.0 +scale=9.0
        [2022-05-13 17:19:51,729][HYDRA] 	#2 : +value=1.0 +scale=11.0
        [2022-05-13 17:19:51,787][HYDRA] 	#3 : +value=1.0 +scale=9.0

        >>> b = B()
        >>> b.run(target_job_dirs=a.multirun_working_dirs)
        [2022-05-13 17:19:59,900][HYDRA] Launching 6 jobs locally
        [2022-05-13 17:19:59,900][HYDRA] 	#0 : +job_dir=/home/scratch/multirun/0
        [2022-05-13 17:19:59,958][HYDRA] 	#1 : +job_dir=/home/scratch/multirun/1
        [2022-05-13 17:20:00,015][HYDRA] 	#2 : +job_dir=/home/scratch/multirun/2
        [2022-05-13 17:20:00,073][HYDRA] 	#3 : +job_dir=/home/scratch/multirun/3

        >>> b.target_dir_multirun_overrides
        {'value': [-1.0, -1.0, 1.0, 1.0],
         'scale': [11.0, 9.0, 11.0, 9.0]}"""
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

    @staticmethod
    def _process_metrics(job_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        metrics_filename: Union[str, Sequence[str], None],
    ) -> Self:
        """Loading workflow job data from a given working directory. The workflow
        is loaded in-place and "self" is returned by this method.

        Parameters
        ----------
        working_dir: str | Path
            The base working directory of the experiment. It is expected
            that subdirectories within this working directory will contain
            individual Hydra jobs data (yaml configurations) and saved metrics files.

        metrics_filename: str | Sequence[str] | None
            The filename(s) or glob-pattern(s) uses to load the metrics.
            If `None`, the metrics stored in `self.metrics` is used.

        Returns
        -------
        loaded_workflow : Self
        """
        self.working_dir = Path(working_dir)
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

        for dir_ in self.multirun_working_dirs:
            # Ensure we load saved YAML configurations for each job (in hydra.job.output_subdir)
            cfg_file = dir_ / f"{self.output_subdir}/config.yaml"
            assert cfg_file.exists(), cfg_file
            self.cfgs.append(load_from_yaml(cfg_file))

        if metrics_filename is not None:
            self.load_metrics(metrics_filename)

        return self

    def load_metrics(
        self, metrics_filename: Union[str, Sequence[str]]
    ) -> Dict[str, List[Any]]:
        """Loads and aggregates across all multirun working dirs, and stores
        the metrics in `self.metrics`.

        `self.metric_load_fn` is used to load each job's metric file(s).

        Parameters
        ----------
        metrics_filename : str | Sequence[str]
            The filename(s) or glob-pattern(s) uses to load the metrics.
            If `None`, the metrics stored in `self.metrics` is used.

        Returns
        -------
        metrics : Dict[str, List[Any]]

        Examples
        --------
        Creating a workflow that saves named metrics using `torch.save`

        >>> from rai_toolbox.mushin.workflows import MultiRunMetricsWorkflow, multirun
        >>> import torch as tr
        >>>
        ... class TorchWorkFlow(MultiRunMetricsWorkflow):
        ...     @staticmethod
        ...     def task(a, b):
        ...         tr.save(dict(a=a, b=b), "metrics.pt")
        ...
        >>> wf = TorchWorkFlow()
        >>> wf.run(a=multirun([1, 2, 3]), b=False)
        [2022-06-01 12:35:51,650][HYDRA] Launching 3 jobs locally
        [2022-06-01 12:35:51,650][HYDRA] 	#0 : +a=1 +b=False
        [2022-06-01 12:35:51,715][HYDRA] 	#1 : +a=2 +b=False
        [2022-06-01 12:35:51,780][HYDRA] 	#2 : +a=3 +b=False

        `~MultiRunMetricsWorkflow` uses `torch.load` by default to load metrics files
        (refer to `~MultiRunMetricsWorkflow.metric_load_fn` to change this behavior).

        >>> wf.load_metrics("metrics.pt")
        defaultdict(list, {'a': [1, 2, 3], 'b': [False, False, False]})
        >>> wf.metrics
        defaultdict(list, {'a': [1, 2, 3], 'b': [False, False, False]})
        """
        if self.multirun_working_dirs is None:
            self.load_from_dir(self.working_dir, metrics_filename=None)
            assert self.multirun_working_dirs is not None

        if isinstance(metrics_filename, str):
            metrics_filename = [metrics_filename]

        job_metrics = []
        for dir_ in self.multirun_working_dirs:
            _metrics = {}
            for name in metrics_filename:
                files = sorted(dir_.glob(name))
                if not files:
                    raise FileNotFoundError(
                        f"No files with the path/pattern {dir_/name} were found"
                    )

                for f_ in files:
                    _metrics.update(self.metric_load_fn(f_))
            job_metrics.append(_metrics)

        self.metrics = self._process_metrics(job_metrics)

        return self.metrics

    @staticmethod
    def _sanitize_coordinate_for_xarray(
        value: Union[LoadedValue, Sequence[LoadedValue]]
    ) -> Union[str, int, float, bool, List[Union[str, int, float, bool]]]:
        """Nested sequences are not permitted for xarray coordinates. This
        Returns a list of scalars when `value` is a multi-run or a scalar.

        Inner sequences are converted to strings"""
        if _non_str_sequence(value):
            if isinstance(value, multirun):
                _seq: Sequence[LoadedValue] = value
                return [str(_v) if _non_str_sequence(_v) else _v for _v in _seq]
            return str(value)
        return value  # type: ignore

    def to_xarray(
        self,
        include_working_subdirs_as_data_var: bool = False,
        coord_from_metrics: Optional[str] = None,
        non_multirun_params_as_singleton_dims: bool = False,
        metrics_filename: Union[str, Sequence[str], None] = None,
    ):
        """Convert workflow data to xarray Dataset.

        Parameters
        ----------
        include_working_subdirs_as_data_var : bool, optional (default=False)
            If `True` then the data-variable "working_subdir" will be included in the
            xarray. This data variable is used to lookup the working sub-dir path
            (a string) by multirun coordinate.

        coord_from_metrics : str | None (default: None)
            If not `None` defines the metric key to use as a coordinate
            in the `Dataset`.  This function assumes that this coordinate
            represents the leading dimension for all data-variables.

        non_multirun_params_as_singleton_dims : bool, optional (default=False)
            If `True` then non-multirun entries from `workflow_overrides` will be
            included as length-1 dimensions in the xarray. Useful for merging/
            concatenation with other Datasets

        metrics_filename: Optional[str]
            The filename or glob-pattern uses to load the metrics.
            If `None`, the metrics stored in `self.metrics` is used.

        Returns
        -------
        results : xarray.Dataset
            A dataset whose dimensions and coordinate-values are determined by the
            quantities over which the multi-run was performed. The data variables
            correspond to the named results returned by the jobs."""
        import xarray as xr

        if metrics_filename is not None:
            if self.multirun_working_dirs is None:
                self.load_from_dir(self.working_dir, metrics_filename=metrics_filename)
            else:
                self.load_metrics(metrics_filename)

        # all overrides containing non-multirun lists must be converted to
        # strings so that xarray treats that list value as a "scalar"
        #
        # stores: override-name -> value
        # where value is either a scalar (i.e. int|float|bool|str) or a list of scalars
        # A list of scalars indicates a multirun
        cast_overrides = {
            k: self._sanitize_coordinate_for_xarray(value)
            for k, value in self.multirun_task_overrides.items()
        }

        orig_coords = {
            k: (v if _non_str_sequence(v) else [v])
            for k, v in cast_overrides.items()
            if non_multirun_params_as_singleton_dims or _non_str_sequence(v)
        }

        metric_coords = {}
        if coord_from_metrics:
            if coord_from_metrics not in self.metrics:
                raise ValueError(
                    f"key `{coord_from_metrics}` not in metrics (available: "
                    f"{list(self.metrics.keys())})"
                )

            v = _coerce_list_of_arraylikes(self.metrics[coord_from_metrics])
            v = np.asarray(v)

            if v.ndim > 1:  # pragma: no cover
                # assume this coord was repeated across experiments, e.g., "epochs"
                v = v[0]
            metric_coords[coord_from_metrics] = v

        attrs = {k: v for k, v in cast_overrides.items() if not _non_str_sequence(v)}

        # we will add additional coordinates as-needed for multi-dim metrics
        coords: Dict[str, Any] = orig_coords.copy()
        shape = tuple(len(v) for v in coords.values())

        metrics_to_add = self.metrics.copy()
        if (
            include_working_subdirs_as_data_var
            and self.multirun_working_dirs is not None
        ):
            metrics_to_add["working_subdir"] = [
                str(p) for p in self.multirun_working_dirs
            ]

        data = {}
        for k, v in metrics_to_add.items():
            if coord_from_metrics and k == coord_from_metrics:
                continue

            v = _coerce_list_of_arraylikes(v)

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
                    if (
                        len(set(np.unique(v))) > 1
                        or non_multirun_params_as_singleton_dims
                    ):
                        coords[k] = (
                            [self._JOBDIR_NAME],
                            [self._sanitize_coordinate_for_xarray(item) for item in v],
                        )
            out = out.assign_coords(coords)
        return out


class RobustnessCurve(MultiRunMetricsWorkflow):
    """Abstract class for workflows that measure performance for different perturbation
    values.

    This workflow requires and uses parameter `epsilon` as the configuration option for
    varying the perturbation.

    See Also
    --------
    MultiRunMetricsWorkflow
    """

    def run(
        self,
        *,
        epsilon: Union[str, Sequence[float]],
        task_fn_wrapper: Union[
            Callable[[Callable[..., T1]], Callable[[Any], T1]], None
        ] = zen,
        pre_task_fn_wrapper: Union[
            Callable[[Callable[..., None]], Callable[[Any], None]], None
        ] = zen,
        target_job_dirs: Optional[Sequence[Union[str, Path]]] = None,  # TODO: add docs
        version_base: Optional[Union[str, Type[_NotSet]]] = _VERSION_BASE_DEFAULT,
        working_dir: Optional[str] = None,
        sweeper: Optional[str] = None,
        launcher: Optional[str] = None,
        overrides: Optional[List[str]] = None,
        to_dictconfig: bool = False,
        config_name: str = "rai_workflow",
        job_name: str = "rai_workflow",
        with_log_configuration: bool = True,
        **workflow_overrides: Union[str, int, float, bool, multirun, hydra_list],
    ):
        """Run the experiment for varying value `epsilon`.

        Parameters
        ----------
        epsilon: str | Sequence[float]
            The configuration parameter for the perturbation.  Unlike Hydra overrides,
            this parameter can be a list of floats that will be converted into a
            multirun sequence override for Hydra.

        task_fn_wrapper: Callable[[Callable[..., T1]], Callable[[Any], T1]] | None, optional (default=rai_toolbox.mushin.zen)
            A wrapper applied to `self.task` prior to launching the task.
            The default wrapper is `rai_toolbox.mushin.zen`. Specify `None` for no
            wrapper to be applied.

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

            These values will be appended to the `overrides` for the Hydra job.
        """

        if not isinstance(epsilon, str):
            epsilon = multirun(epsilon)

        return super().run(
            task_fn_wrapper=task_fn_wrapper,
            pre_task_fn_wrapper=pre_task_fn_wrapper,
            working_dir=working_dir,
            sweeper=sweeper,
            launcher=launcher,
            version_base=version_base,
            overrides=overrides,
            to_dictconfig=to_dictconfig,
            config_name=config_name,
            job_name=job_name,
            with_log_configuration=with_log_configuration,
            **workflow_overrides,
            # for multiple multi-run params, epsilon should fastest-varying param;
            # i.e. epsilon should be the trailing dim in the multi-dim array of results
            target_job_dirs=target_job_dirs,
            epsilon=epsilon,
        )

    def to_xarray(
        self,
        include_working_subdirs_as_data_var: bool = False,
        coord_from_metrics: Optional[str] = None,
        non_multirun_params_as_singleton_dims: bool = False,
        metrics_filename: Union[str, Sequence[str], None] = None,
    ):
        """Convert workflow data to xarray Dataset.

        Parameters
        ----------
        include_working_subdirs_as_data_var : bool, optional (default=False)
            If `True` then the data-variable "working_subdir" will be included in the
            xarray. This data variable is used to lookup the working sub-dir path
            (a string) by multirun coordinate.

        coord_from_metrics : str | None (default: None)
            If not `None` defines the metric key to use as a coordinate
            in the `Dataset`.  This function assumes that this coordinate
            represents the leading dimension for all data-variables.

        non_multirun_params_as_singleton_dims : bool, optional (default=False)
            If `True` then non-multirun entries from `workflow_overrides` will be
            included as length-1 dimensions in the xarray. Useful for merging/
            concatenation with other Datasets

        metrics_filename: Optional[str]
            The filename or glob-pattern uses to load the metrics.
            If `None`, the metrics stored in `self.metrics` is used.

        Returns
        -------
        results : xarray.Dataset
            A dataset whose dimensions and coordinate-values are determined by the
            quantities over which the multi-run was performed. The data variables
            correspond to the named results returned by the jobs."""
        return (
            super()
            .to_xarray(
                include_working_subdirs_as_data_var=include_working_subdirs_as_data_var,
                coord_from_metrics=coord_from_metrics,
                non_multirun_params_as_singleton_dims=non_multirun_params_as_singleton_dims,
                metrics_filename=metrics_filename,
            )
            .sortby("epsilon")
        )

    def plot(
        self,
        metric: str,
        ax: Any = None,
        group: Optional[str] = None,
        save_filename: Optional[str] = None,
        non_multirun_params_as_singleton_dims: bool = False,
        **kwargs,
    ) -> Any:
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
