# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch as tr
import xarray as xr
from hydra_zen import make_config

from rai_toolbox.mushin.workflows import BaseWorkflow, RobustnessCurve, _load_metrics


class MyWorkflow(BaseWorkflow):
    @staticmethod
    def evaluation_task():
        return dict(result=1)

    def jobs_post_process(self, workflow_params):
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
        task.to_dataframe()

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
@pytest.mark.parametrize("config", [None, make_config(epsilon=0)])
@pytest.mark.parametrize("as_array", [True, False])
def test_robustnesscurve_run(config, as_array):
    LocalRobustness = create_workflow(as_array)
    task = LocalRobustness(config)
    task.run(epsilon=[0, 1, 2, 3])

    assert "result" in task.metrics
    assert len(task.metrics["result"]) == 4
    assert "epsilon" in task.workflow_overrides
    assert len(task.workflow_overrides["epsilon"]) == 4

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
def test_robustnesscurve_hydra():
    LocalRobustness = create_workflow()
    task = LocalRobustness(make_config(epsilon=0))
    task.run(epsilon=[0, 1, 2, 3], sweeper="basic", launcher="basic")


@pytest.mark.usefixtures("cleandir")
def test_robustnesscurve_override():
    LocalRobustness = create_workflow()
    task = LocalRobustness(make_config(epsilon=0))
    task.run(epsilon=[0, 1, 2, 3], overrides=["hydra.sweep.dir=test_sweep_dir"])
    assert Path("test_sweep_dir").exists()


@pytest.mark.usefixtures("cleandir")
def test_robustnesscurve_to_data():
    LocalRobustness = create_workflow()
    task = LocalRobustness(make_config(epsilon=0))
    task.run(epsilon=[0, 1, 2, 3])

    df = task.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df["epsilon"]) == 4

    xd = task.to_xarray()

    assert isinstance(xd, xr.Dataset)
    assert len(xd["epsilon"]) == 4


@pytest.mark.usefixtures("cleandir")
def test_robustnesscurve_load_from_dir():
    LocalRobustness = create_workflow()
    task = LocalRobustness()
    task.run(epsilon=[0, 1, 2, 3], working_dir="test_dir")

    working_dir = "test_dir"
    LocalRobustness = create_workflow()
    load_task = LocalRobustness()
    load_task.load_from_dir(working_dir, workflow_params=["epsilon"])

    for k, v in task.workflow_overrides.items():
        assert k in load_task.workflow_overrides
        assert v == load_task.workflow_overrides[k]

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
        task.plot("result", group="fake_param")
    else:
        with pytest.raises(TypeError):
            task.run(epsilon=[0, 1, 2, 3], fake_param=BadVal)  # type: ignore


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


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("ax", [None, plt.subplots()[1]])
def test_robustnesscurve_plot_save(ax):
    LocalRobustness = create_workflow()
    task = LocalRobustness()
    task.run(epsilon=[0, 1, 2, 3])
    task.plot("result", save_filename="test_save.png", ax=ax)
    assert Path("test_save.png").exists()


@pytest.mark.parametrize("workflow_params", [None, "hi"])
def test_load_metrics(workflow_params):
    job_overrides = [["hi=1"], ["by=2"]]
    job_metrics = [dict(val=1), dict(val=2)]
    mets, overs = _load_metrics(job_overrides, job_metrics, workflow_params)

    assert "val" in mets
    assert len(mets["val"]) == 2
    assert "hi" in overs

    if workflow_params is None:
        assert "by" in overs
