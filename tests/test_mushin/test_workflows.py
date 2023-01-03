# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
import string
from pathlib import Path
from typing import Optional, Sequence

import hypothesis.strategies as st
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch as tr
import xarray as xr
from hydra.core.config_store import ConfigStore
from hydra.plugins.sweeper import Sweeper
from hydra_zen import builds, load_from_yaml, make_config
from hydra_zen.errors import HydraZenValidationError
from hypothesis import given, settings
from hypothesis.extra.numpy import array_shapes, arrays
from numpy.testing import assert_allclose
from xarray.testing import assert_duckarray_equal, assert_identical

from rai_toolbox.mushin import multirun
from rai_toolbox.mushin.workflows import (
    BaseWorkflow,
    MultiRunMetricsWorkflow,
    RobustnessCurve,
    hydra_list,
)

common_shape = array_shapes(min_dims=2, max_dims=2)

epsilons = arrays(
    shape=st.integers(1, 5),
    dtype="float",
    elements=st.floats(-1000, 1000),
)


class MyWorkflow(BaseWorkflow):
    @staticmethod
    def task():
        return dict(result=1)

    def jobs_post_process(self):
        return


def create_workflow(as_array=False):
    class LocalRobustness(RobustnessCurve):
        @staticmethod
        def task(epsilon):
            val = 100 - epsilon**2
            if as_array:
                val = [val]

            result = dict(result=val)

            tr.save(result, "test_metrics.pt")
            return result

    return LocalRobustness


def test_robustnesscurve_raises_notimplemented():
    task = MyWorkflow()

    with pytest.raises(NotImplementedError):
        task.plot()

    with pytest.raises(NotImplementedError):
        task.to_xarray()


@pytest.mark.usefixtures("cleandir")
def test_robustnesscurve_validate():
    LocalRobustness = create_workflow()

    task = LocalRobustness(make_config(epsilon=0))
    task.validate()

    task = LocalRobustness()
    with pytest.raises(HydraZenValidationError):
        task.validate()


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "config", [None, make_config(epsilon=0), make_config(epsilon=None)]
)
@pytest.mark.parametrize("as_array", [True, False])
def test_robustnesscurve_run(config, as_array):
    epsilon = [0, 1, 2, 3.0]
    LocalRobustness = create_workflow(as_array)
    task = LocalRobustness(config)
    task.run(epsilon=epsilon)

    assert "result" in task.metrics
    assert len(task.metrics["result"]) == len(epsilon)

    multirun_task_overrides = task.multirun_task_overrides
    assert "epsilon" in multirun_task_overrides

    extracted_epsilon = multirun_task_overrides["epsilon"]
    assert isinstance(extracted_epsilon, Sequence)
    assert len(extracted_epsilon) == len(epsilon)

    # will raise if not set correctly
    task.plot("result")


@pytest.mark.usefixtures("cleandir")
def test_robustnesscurve_working_dir():
    LocalRobustness = create_workflow()
    task = LocalRobustness(make_config(epsilon=0))
    task.run(epsilon="0,1,2,3", working_dir="test_dir")

    assert str(task.working_dir).endswith("test_dir")
    assert Path("test_dir").exists()


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "sweeper,launcher", [(None, None), (None, "basic"), ("basic", None)]
)
def test_robustnesscurve_hydra(sweeper, launcher):
    LocalRobustness = create_workflow()
    task = LocalRobustness(make_config(epsilon=0))
    task.run(epsilon=[0, 1, 2, 3], sweeper=sweeper, launcher=launcher)


@pytest.mark.usefixtures("cleandir")
def test_robustnesscurve_override():
    LocalRobustness = create_workflow()
    task = LocalRobustness(make_config(epsilon=0))

    overrides = ["hydra.sweep.dir=test_sweep_dir"]
    task.run(epsilon=[0, 1, 2, 3], overrides=overrides)
    assert Path("test_sweep_dir").exists()

    # make sure overrides is not modified
    assert len(overrides) == 1


@pytest.mark.usefixtures("cleandir")
@settings(deadline=None, max_examples=10)
@given(epsilon=epsilons)
def test_robustnesscurve_to_data(epsilon):
    LocalRobustness = create_workflow()
    task = LocalRobustness(make_config(epsilon=0))
    task.run(epsilon=epsilon)

    xd = task.to_xarray(non_multirun_params_as_singleton_dims=True)

    assert isinstance(xd, xr.Dataset)
    assert len(xd["epsilon"]) == len(epsilon)


@pytest.mark.usefixtures("cleandir")
def test_robustnesscurve_load_from_dir():
    LocalRobustness = create_workflow()
    task = LocalRobustness()
    task.run(epsilon=[0, 1, 2, 3], working_dir="test_dir")

    working_dir = "test_dir"
    LocalRobustness = create_workflow()
    load_task = LocalRobustness()
    load_task.load_from_dir(working_dir, "test_metrics.pt")

    for k, v in task.multirun_task_overrides.items():
        assert k in load_task.multirun_task_overrides
        assert v == load_task.multirun_task_overrides[k]

    for k, v in task.metrics.items():
        assert k in load_task.metrics
        assert v == load_task.metrics[k]

    for cfg in task.cfgs:
        assert "epsilon" in cfg


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("fake_param_string", [True, False])
def test_robustnesscurve_extra_param(fake_param_string: bool):
    class BadVal:
        pass

    LocalRobustness = create_workflow()
    task = LocalRobustness(make_config(epsilon=0))

    if fake_param_string:
        task.run(epsilon=[0, 1, 2, 3], fake_param="some_value")
    else:
        with pytest.raises(TypeError):
            task.run(epsilon=[0, 1, 2, 3], fake_param=BadVal)  # type: ignore
        return

    assert "fake_param" in task.multirun_task_overrides
    assert task.multirun_task_overrides["fake_param"] == "some_value"
    task.plot("result", group="fake_param", non_multirun_params_as_singleton_dims=True)


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("fake_param_string", [True, False])
def test_robustnesscurve_extra_param_multirun(fake_param_string: bool):
    LocalRobustness = create_workflow()
    task = LocalRobustness(make_config(epsilon=0))

    if fake_param_string:
        task.run(epsilon=[0, 1, 2, 3], fake_param="1,2")
        task.plot("result")
    else:
        with pytest.raises(TypeError):
            task.run(epsilon=[0, 1, 2, 3], fake_param=[1, 2])  # type: ignore
        return

    multirun_task_overrides = task.multirun_task_overrides
    assert "fake_param" in multirun_task_overrides

    extracted_fake_param = multirun_task_overrides["fake_param"]
    assert isinstance(extracted_fake_param, Sequence)
    assert len(extracted_fake_param) == 2


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("ax", [None, plt.subplots()[1]])
def test_robustnesscurve_plot_save(ax):
    LocalRobustness = create_workflow()
    task = LocalRobustness()
    task.run(epsilon=[0, 1, 2, 3])
    task.plot("result", save_filename="test_save.png", ax=ax)
    assert Path("test_save.png").exists()


class MultiDimMetrics(RobustnessCurve):
    # returns     "images" -> shape-(4, 1)
    #         "accuracies" -> scalar
    @staticmethod
    def task(epsilon):
        val = 100 - epsilon**2
        result = dict(images=[[val] * 1] * 4, accuracies=val + 2)
        tr.save(result, "test_metrics.pt")
        return result


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "foo, foo_expected", [("val", "val"), (hydra_list(["val"]), "['val']")]
)
@pytest.mark.parametrize("bar", [multirun(["a", "b"]), multirun(["[a,b]", "[c,d]"])])
def test_robustness_with_multidim_metrics(foo, foo_expected, bar):
    wf = MultiDimMetrics()
    wf.run(epsilon=[1.0, 3.0, 2.0], foo=foo, bar=bar)
    xarray = wf.to_xarray()
    assert list(xarray.data_vars.keys()) == ["images", "accuracies"]
    assert list(xarray.coords.keys()) == [
        "bar",
        "epsilon",
        "images_dim0",
        "images_dim1",
    ]
    assert xarray.accuracies.shape == (2, 3)
    assert xarray.images.shape == (2, 3, 4, 1)
    assert xarray.attrs == {"foo": foo_expected}

    for eps, expected in zip([1.0, 2.0, 3.0], [99.0, 96.0, 91.0]):
        # test that results were organized as-expected
        sub_xray = xarray.sel(epsilon=eps)
        assert np.all(sub_xray.accuracies == expected + 2).item()
        assert np.all(sub_xray.images == expected).item()


class MultiDimIterationMetrics(MultiRunMetricsWorkflow):
    # returns "images" -> shape-(N, 4, 4)
    #         "accuracies" -> N
    #         "epochs" -> (N, 10)

    @staticmethod
    def task(epsilon, as_tensor: bool):
        assert isinstance(as_tensor, bool)
        backend = np if as_tensor else tr
        val = 100 * backend.ones(10) - epsilon**2
        epochs = backend.arange(10)
        images = 100 * backend.ones((10, 4, 4)) - epsilon**2
        result = dict(images=images, accuracies=val + 2, epochs=epochs)
        tr.save(result, "test_metrics.pt")
        return result


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("as_tensor", [False, True])
def test_robustness_with_multidim_metrics_with_iteration(as_tensor: bool):
    wf = MultiDimIterationMetrics()
    wf.run(
        epsilon=multirun([1.0, 3.0, 2.0]),
        foo="val",
        bar=multirun(["a", "b"]),
        as_tensor=as_tensor,
    )
    xarray = wf.to_xarray(coord_from_metrics="epochs")
    assert list(xarray.data_vars.keys()) == ["images", "accuracies"]
    assert list(xarray.coords.keys()) == [
        "epsilon",
        "bar",
        "epochs",
        "images_dim1",
        "images_dim2",
    ]
    assert xarray.accuracies.shape == (3, 2, 10)
    assert xarray.images.shape == (3, 2, 10, 4, 4)
    assert xarray.attrs == {"foo": "val", "as_tensor": as_tensor}

    for eps, expected in zip([1.0, 2.0, 3.0], [99.0, 96.0, 91.0]):
        # test that results were organized as-expected
        sub_xray = xarray.sel(epsilon=eps)
        assert np.all(sub_xray.accuracies == expected + 2).item()
        assert np.all(sub_xray.images == expected).item()

    with pytest.raises(ValueError):
        wf.to_xarray(coord_from_metrics="key_not_in_metrics")


@pytest.mark.usefixtures("cleandir")
def test_xarray_from_loaded_workflow():
    wf = MultiDimMetrics()
    wf.run(epsilon=[1.0, 3.0, 2.0], foo="val", bar=multirun(["a", "b"]))
    xarray1 = wf.to_xarray()

    wf2 = MultiDimMetrics().load_from_dir(wf.working_dir, "test_metrics.pt")
    xarray2 = wf2.to_xarray()
    assert_identical(xarray1, xarray2)

    wf3 = MultiDimMetrics(working_dir=wf.working_dir)
    xarray3 = wf3.to_xarray(metrics_filename="test_metrics.pt")
    assert_identical(xarray1, xarray3)

    wf4 = MultiDimMetrics().load_from_dir(wf.working_dir, metrics_filename=None)
    xarray4 = wf4.to_xarray(metrics_filename="test_metrics.pt")
    assert_identical(xarray1, xarray4)


class LocalBasicSweeper(Sweeper):
    def setup(self, *, hydra_context, task_function, config):
        pass

    def sweep(self, arguments):
        return dict(hi=1)


@pytest.mark.usefixtures("cleandir")
def test_return_not_list_jobreturn():
    cs = ConfigStore.instance()
    cs.store(group="hydra/sweeper", name="local_test", node=builds(LocalBasicSweeper))

    wf = MyWorkflow()
    wf.run(epsilon=multirun([1.0, 3.0, 2.0]), overrides=["hydra/sweeper=local_test"])
    assert wf.jobs == dict(hi=1)


class FirstMultiRun(RobustnessCurve):
    # returns     "images" -> shape-(4, 1)
    #         "accuracies" -> scalar
    @staticmethod
    def task(epsilon, acc):
        val = 100 - epsilon**2
        result = dict(images=np.array([[val] * 1] * 4), accuracies=acc)
        tr.save(result, "test_metrics.pt")
        return result


class ScndMultiRun(MultiRunMetricsWorkflow):
    # loads test metrics, multiplies each by `val` and saves
    @staticmethod
    def task(job_dir, val):
        result = tr.load(f"{job_dir}/test_metrics.pt")

        # val multiplies each metric
        result = {k: v * val for k, v in result.items()}
        tr.save(result, "test_metrics.pt")
        return result


@pytest.mark.parametrize("load_from_working_dir", [False, True])
@pytest.mark.usefixtures("cleandir")
def test_multirun_over_jobdir(load_from_working_dir):
    # Runs a standard multirun workflow and then runs
    # a multirun over the resulting folders, loading in
    # their metrics and re-returning them
    wf = FirstMultiRun()
    wf.run(
        epsilon=multirun([1.0, 2.0, 3.0]),
        acc=multirun([1, 2]),
        list_vals=multirun([[0, 1]]),  # ensure that multiruns over lists work
        working_dir="first",
    )

    snd_wf = ScndMultiRun()
    # runs over a total of epsilon-3 x acc-2 -> 6 job-dirs and 2 val
    snd_wf.run(
        target_job_dirs=wf.multirun_working_dirs,
        val=multirun([1, 2]),
        working_dir="second",
    )

    if load_from_working_dir:
        snd_wf = ScndMultiRun().load_from_dir(snd_wf.working_dir, "test_metrics.pt")

    assert wf.target_dir_multirun_overrides == {}
    assert snd_wf.target_dir_multirun_overrides == {
        "acc": [1, 1, 1, 2, 2, 2],
        "epsilon": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "list_vals": [[0, 1]] * 6,
    }
    xr1 = wf.to_xarray()
    xr2 = snd_wf.to_xarray()

    assert xr1.dims == {"acc": 2, "epsilon": 3, "images_dim0": 4, "images_dim1": 1}
    assert xr2.dims == {"val": 2, "job_dir": 6, "images_dim0": 4, "images_dim1": 1}

    xr2 = xr2.set_index(job_dir=["epsilon", "acc", "list_vals"]).unstack("job_dir")
    xr2 = xr2.transpose(
        "list_vals", "val", "acc", "epsilon", "images_dim0", "images_dim1"
    )

    assert_identical(xr1.epsilon, xr2.epsilon)
    assert_identical(xr1.acc, xr2.acc)

    assert_duckarray_equal(xr1.images, xr2.images.sel(val=1, list_vals="[0, 1]"))
    assert_duckarray_equal(2 * xr1.images, xr2.images.sel(val=2, list_vals="[0, 1]"))
    assert_duckarray_equal(
        xr1.accuracies, xr2.accuracies.sel(val=1, list_vals="[0, 1]")
    )
    assert_duckarray_equal(
        2 * xr1.accuracies, xr2.accuracies.sel(val=2, list_vals="[0, 1]")
    )


class NoMetrics(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x: int, y: int):
        pass


@pytest.mark.parametrize("load_from_working_dir", [False, True])
@pytest.mark.usefixtures("cleandir")
def test_multirun_metrics_workflow_no_metrics(load_from_working_dir):
    wf = NoMetrics()
    wf.run(x=multirun([-1, 0, 1]), y=multirun([-10, 10]))
    assert wf.multirun_task_overrides == {"x": [-1, 0, 1], "y": [-10, 10]}

    if load_from_working_dir:
        wf = NoMetrics().load_from_dir(wf.working_dir, metrics_filename=None)

    xdata = wf.to_xarray()
    assert xdata.dims == {"x": 3, "y": 2}
    assert_allclose(xdata.coords["x"].data, [-1, 0, 1])
    assert_allclose(xdata.coords["y"].data, [-10, 10])
    assert len(xdata.data_vars) == 0


class GridMetrics(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x: int, y: int):
        results = dict(xx=x, yy=y)
        tr.save(results, "test_metrics.pt")
        return results


@pytest.mark.parametrize("hydra_sweep_dir", [None, "cross_validation/", "."])
@pytest.mark.parametrize("hydra_sweep_subdir", [None, "x_${x}_y_${y}"])
@pytest.mark.parametrize("load_from_working_dir", [False, True])
@pytest.mark.usefixtures("cleandir")
def test_working_subdirs(
    hydra_sweep_dir: Optional[str],
    hydra_sweep_subdir: Optional[str],
    load_from_working_dir: bool,
):
    overrides = [
        f"{k.replace('_', '.')}={v}"
        for k, v in locals().items()
        if k.startswith("hydra") and v is not None
    ]

    wf = GridMetrics()
    wf.run(x=multirun([-1, 0, 1]), y=multirun([-10, 10]), overrides=overrides)

    if load_from_working_dir:
        wf = GridMetrics().load_from_dir(
            wf.working_dir, metrics_filename="test_metrics.pt"
        )

    xdata = wf.to_xarray(include_working_subdirs_as_data_var=True)

    # ensure data variables are set appropriately
    yy, xx = np.meshgrid([-10, 10], [-1, 0, 1])
    assert_allclose(actual=xdata.xx.data, desired=xx)
    assert_allclose(actual=xdata.yy.data, desired=yy)

    # ensure working_subdir points to correct dir
    for x_coord in xdata.x:
        for y_coord in xdata.y:
            dd = Path(xdata.working_subdir.sel(x=x_coord, y=y_coord).item())
            cfg = load_from_yaml(dd / ".hydra" / "config.yaml")
            assert cfg == dict(x=x_coord.item(), y=y_coord.item())

    # ensure working_subdir is serializable
    xdata.to_netcdf("tmp.nc")


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "file_pattern", ["*.pt", ("*.pt",), ("images.pt", "acc.pt"), ("images.*", "acc.*")]
)
def test_globbed_xarray(file_pattern):
    class MultiSaveFile(MultiRunMetricsWorkflow):
        # returns     "images" -> shape-(4, 1)
        #         "accuracies" -> scalar
        @staticmethod
        def task(epsilon, acc):
            val = 100 - epsilon**2
            result = dict(images=np.array([[val] * 1] * 4), accuracies=acc)
            tr.save(dict(images=result["images"]), "images.pt")
            tr.save(dict(accuracies=result["accuracies"]), "acc.pt")

            return result

    # saves multiple metrics files that we load/merge via glob pattern
    wf = MultiSaveFile()
    wf.run(epsilon=multirun([1, 2, 3]), acc=multirun([0.9, 0.95, 0.99]))
    xdata1 = wf.to_xarray()

    wf2 = MultiSaveFile(working_dir=wf.working_dir)
    xdata2 = wf2.to_xarray(metrics_filename=file_pattern)

    assert_identical(xdata1, xdata2)


@pytest.mark.parametrize("seed", [0, 123])
@pytest.mark.usefixtures("cleandir")
def test_pre_task_seeding(seed: int):
    class HasPreTask(MultiRunMetricsWorkflow):
        @staticmethod
        def pre_task(seed: int):
            np.random.seed(seed)

        @staticmethod
        def task(rand_val: int):
            return {"rand_val": rand_val}

    wf = HasPreTask(make_config(rand_val=builds(np.random.rand)))
    wf.run(seed=seed)
    actual = wf.jobs[0].return_value["rand_val"]

    np.random.seed(seed)
    expected = np.random.rand()
    assert expected == actual


def test_raises_on_non_static_method():
    class NonStaticEvalTask(MultiRunMetricsWorkflow):
        def task(self):
            pass

    class NonStaticPreTask(MultiRunMetricsWorkflow):
        def pre_task(self):
            pass

    with pytest.raises(TypeError, match="task must be a static method"):
        NonStaticEvalTask().run()

    with pytest.raises(TypeError, match="pre_task must be a static method"):
        NonStaticPreTask().run()


@pytest.mark.usefixtures("cleandir")
@settings(max_examples=10, deadline=None)
@given(
    int_=st.integers(),
    bool_=st.booleans(),
    float_=st.floats(-10, 10),
    list_=st.lists(st.integers()),
    str_=st.text(alphabet=string.ascii_lowercase).filter(
        lambda x: x != "true" and x != "false"
    ),
    mrun=st.lists(
        st.booleans() | st.lists(st.integers()),
        min_size=2,
        max_size=5,
    ).map(multirun),
)
def test_overrides_roundtrip(
    int_,
    bool_,
    float_,
    str_,
    list_,
    mrun,
):
    class WorkFlow(MultiDimIterationMetrics):
        @staticmethod
        def task():
            pass

    wf = WorkFlow()
    overrides = dict(
        int_=int_,
        float_=float_,
        str_=str_,
        bool_=bool_,
        list_=hydra_list(list_),
        mrun=mrun,
    )
    wf.run(**overrides)

    assert wf.multirun_task_overrides == overrides
    xdata = wf.to_xarray(non_multirun_params_as_singleton_dims=True)
    assert xdata.int_.item() == int_
    assert xdata.bool_.item() == bool_
    assert xdata.str_.item() == str_
    if not isinstance(mrun[0], list) and all(
        isinstance(i, type(mrun[0])) for i in mrun
    ):
        assert xdata.mrun.data.tolist() == mrun
    else:
        assert xdata.mrun.data.tolist() == [str(i) for i in mrun]


@pytest.mark.usefixtures("cleandir")
def test_evaluation_task_is_deprecated():
    class OldWorkflow(MultiRunMetricsWorkflow):
        @staticmethod
        def evaluation_task(a: int):
            return dict(a=a)

    with pytest.warns(FutureWarning):
        wf = OldWorkflow()

    wf.run(a=10)
    out = wf.jobs[0]
    assert out.return_value == dict(a=10)


@pytest.mark.usefixtures("cleandir")
def test_custom_metric_load_fn():
    import pickle

    class PickleWorkFlow(MultiRunMetricsWorkflow):
        def metric_load_fn(self, file_path: Path):
            with file_path.open("rb") as f:
                return pickle.load(f)

        @staticmethod
        def task(a, b):
            with open("./metrics.pkl", "wb") as f:
                pickle.dump(dict(a=[[a] * 2], b=b), f)

    wf = PickleWorkFlow()
    wf.run(a=multirun([1, 2, 3]), b=False)
    wf.load_metrics("metrics.pkl")
    assert wf.metrics == dict(a=[[1] * 2, [2] * 2, [3] * 2], b=[False] * 3)


@pytest.mark.usefixtures("cleandir")
def test_regression_68():
    # https://github.com/mit-ll-responsible-ai/responsible-ai-toolbox/pull/68
    class Blank(MultiRunMetricsWorkflow):
        @staticmethod
        def task():
            pass

    wf1 = Blank()
    wf1.run(
        list_vals=multirun([[0, 1], [2, 3]]),  # note: multi-run over list-values
        working_dir="first",
    )

    wf2 = Blank()
    wf2.run(
        target_job_dirs=wf1.multirun_working_dirs,
        val=multirun([1, 2]),
        working_dir="second",
    )

    xr1_coords = wf1.to_xarray().list_vals.data
    xr2_coords = wf2.to_xarray().list_vals.data
    assert np.all(xr1_coords == xr2_coords)
