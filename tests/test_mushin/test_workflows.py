# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Sequence

import hypothesis.strategies as st
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch as tr
import xarray as xr
from hydra_zen import make_config
from hypothesis import given, settings
from hypothesis.extra.numpy import array_shapes, arrays
from xarray.testing import assert_identical

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
    def evaluation_task():
        return dict(result=1)

    def jobs_post_process(self):
        return


def create_workflow(as_array=False):
    class LocalRobustness(RobustnessCurve):
        @staticmethod
        def evaluation_task(epsilon):
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
    with pytest.raises(TypeError):
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

    assert str(task.working_dir) == "test_dir"
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
    load_task.load_from_dir(working_dir)

    for k, v in task.multirun_task_overrides.items():
        assert k in load_task.multirun_task_overrides
        assert v == load_task.multirun_task_overrides[k]

    for k, v in task.metrics.items():
        assert k in load_task.metrics
        assert v == load_task.metrics[k]


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("fake_param_string", [True, False])
def test_robustnesscurve_extra_param(fake_param_string):
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
def test_robustnesscurve_extra_param_multirun(fake_param_string):
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
    def evaluation_task(epsilon):
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
    @staticmethod
    def evaluation_task(epsilon):
        val = 100 * np.ones(10) - epsilon**2
        epochs = np.arange(10)
        images = 100 * np.ones((10, 4, 4)) - epsilon**2
        result = dict(images=images, accuracies=val + 2, epochs=epochs)
        tr.save(result, "test_metrics.pt")
        return result


@pytest.mark.usefixtures("cleandir")
def test_robustness_with_multidim_metrics_with_iteration():
    wf = MultiDimIterationMetrics()
    wf.run(epsilon=multirun([1.0, 3.0, 2.0]), foo="val", bar=multirun(["a", "b"]))
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
    assert xarray.attrs == {"foo": "val"}

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

    wf2 = MultiDimMetrics().load_from_dir(wf.working_dir)
    xarray2 = wf2.to_xarray()
    assert_identical(xarray1, xarray2)
