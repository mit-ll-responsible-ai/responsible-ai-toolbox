# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any, List

import pytest
import torch
from hydra.core.utils import JobReturn
from hydra_zen import builds, instantiate, launch, load_from_yaml, make_config
from omegaconf.errors import ConfigAttributeError
from pytorch_lightning import Trainer

from rai_toolbox.mushin.lightning import HydraDDP
from rai_toolbox.mushin.lightning._pl_main import task as pl_main_task
from rai_toolbox.mushin.lightning.callbacks import MetricsCallback
from rai_toolbox.mushin.testing.lightning import (
    SimpleDataModule,
    SimpleLightningModule,
    SimpleLightningModuleNoData,
)

if torch.cuda.device_count() < 2:
    pytest.skip("Need at least 2 GPUs to test", allow_module_level=True)


TrainerConfig = builds(
    Trainer,
    max_epochs=1,
    accelerator="auto",
    devices="${devices}",
    strategy=builds(HydraDDP),
    fast_dev_run=True,
)


def task_fn(cfg: Any):
    trainer: Trainer = instantiate(cfg.trainer)
    module = instantiate(cfg.module)
    if "run_test" in cfg and cfg.run_test:
        trainer.test(module)
    elif "run_predict" in cfg and cfg.run_predict:
        trainer.predict(module)
    else:
        trainer.fit(module)


def task_fn_with_datamodule(cfg: Any):
    trainer: Trainer = instantiate(cfg.trainer)
    module = instantiate(cfg.module)
    datamodule = instantiate(cfg.datamodule)
    if "run_test" in cfg and cfg.run_test:
        trainer.test(module, datamodule=datamodule)
    elif "run_predict" in cfg and cfg.run_predict:
        trainer.predict(module, datamodule=datamodule)
    else:
        trainer.fit(module, datamodule=datamodule)


def task_fn_raises(cfg: Any):
    trainer: Trainer = instantiate(cfg.trainer)
    module = instantiate(cfg.wrong_config_name)
    if "run_test" in cfg and cfg.run_test:
        trainer.test(module)
    else:
        trainer.fit(module)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Need at least 2 GPUs to test"
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "testing, predicting", [(True, False), (False, True), (False, False)]
)
@pytest.mark.usefixtures("cleandir")
def test_ddp_with_hydra_pl_main(testing, predicting):
    trainer = Trainer(max_epochs=1, fast_dev_run=True, callbacks=[MetricsCallback()])
    module = SimpleLightningModule()
    pl_main_task(trainer, module, None, testing, predicting)

    if not testing and not predicting:
        # makes sure Trainer.fit ws executed
        assert len(list(Path.cwd().glob("**/fit_metrics.pt"))) == 1
    elif testing:
        # makes sure Trainer.test ws executed
        assert len(list(Path.cwd().glob("**/test_metrics.pt"))) == 1
    else:
        # makes sure Trainer.predict ws executed
        assert len(list(Path.cwd().glob("**/fit_metrics.pt"))) == 0
        assert len(list(Path.cwd().glob("**/test_metrics.pt"))) == 0


@pytest.mark.usefixtures("cleandir")
def test_ddp_with_hydra_raises_misconfiguration():
    module = builds(SimpleLightningModule)
    Config = make_config(trainer=TrainerConfig, wrong_config_name=module, devices=2)
    with pytest.raises(ConfigAttributeError):
        launch(Config, task_fn_raises, version_base="1.1")


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Need at least 2 GPUs to test"
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("subdir", [None, ".hydra", "dksa"])
@pytest.mark.usefixtures("cleandir")
def test_ddp_with_hydra_output_subdir(subdir):
    overrides = []

    if subdir is not None:
        overrides += [f"hydra.output_subdir={subdir}"]

    module = builds(SimpleLightningModule)
    Config = make_config(trainer=TrainerConfig, module=module, devices=2)
    job = launch(Config, task_fn, overrides)

    if subdir is None:
        subdir = ".hydra"

    assert isinstance(job.working_dir, str)
    cfg_file = Path(job.working_dir) / subdir / "config.yaml"
    assert cfg_file.exists()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Need at least 2 GPUs to test"
)
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "testing, predicting", [(True, False), (False, True), (False, False)]
)
@pytest.mark.usefixtures("cleandir")
def test_ddp_with_hydra_subprocess_runs_correct_mode(testing, predicting):
    overrides = [f"+run_test={testing}", f"+run_predict={predicting}"]
    module = builds(SimpleLightningModule)
    Config = make_config(trainer=TrainerConfig, module=module, devices=2)
    job = launch(Config, task_fn, overrides=overrides)

    assert isinstance(job.working_dir, str)
    cfg_file_run = Path(job.working_dir) / ".hydra/config.yaml"
    assert cfg_file_run.exists()
    cfg_run = load_from_yaml(cfg_file_run)

    assert "run_test" in cfg_run
    assert cfg_run.run_test == testing
    assert "run_predict" in cfg_run
    assert cfg_run.run_predict == predicting

    assert isinstance(job.working_dir, str)
    cfg_file_subprocess = Path(job.working_dir) / ".pl_hydra_rank_1/config.yaml"
    assert cfg_file_subprocess.exists()
    cfg_subprocess = load_from_yaml(cfg_file_subprocess)

    assert "pl_testing" in cfg_subprocess
    assert cfg_subprocess.pl_testing == testing
    assert cfg_subprocess.pl_testing == cfg_run.run_test
    assert "pl_predicting" in cfg_subprocess
    assert cfg_subprocess.pl_predicting == predicting
    assert cfg_subprocess.pl_predicting == cfg_run.run_predict


@pytest.mark.usefixtures("cleandir")
@pytest.mark.filterwarnings("ignore:The dataloader")
@pytest.mark.filterwarnings("ignore:It is recommended to use")
@pytest.mark.filterwarnings("ignore:The number of training batches")
def test_ddp_with_hydra_with_datamodule():
    module = builds(SimpleLightningModuleNoData)
    datamodule = builds(SimpleDataModule)
    Config = make_config(
        trainer=TrainerConfig, module=module, datamodule=datamodule, devices=2
    )
    launch(Config, task_fn_with_datamodule, version_base="1.1")


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Need at least 2 GPUs to test"
)
@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_ddp_with_hydra_runjob(num_jobs):
    overrides = []
    multirun = False
    if num_jobs > 1:
        multirun = True
        # create fake multirun params based on `num_jobs`
        fake_param = "+foo="
        for i in range(num_jobs):
            fake_param += f"{i}"
            if i < num_jobs - 1:
                fake_param += ","

        overrides += [fake_param]

    module = builds(SimpleLightningModule)
    Config = make_config(trainer=TrainerConfig, module=module, devices=2)
    launch_job = launch(
        Config, task_fn, overrides, multirun=multirun, version_base="1.1"
    )

    if multirun:
        assert isinstance(launch_job, list)
        assert len(launch_job) == 1
        assert isinstance(launch_job[0], list)
        assert isinstance(launch_job[0][0], JobReturn)
        jobs: List[JobReturn] = launch_job[0]
    else:
        assert isinstance(launch_job, JobReturn)
        jobs: List[JobReturn] = [launch_job]

    # Make sure the parameter was set and used
    for job in jobs:
        assert job.cfg is not None
        assert "devices" in job.cfg
        assert job.cfg.devices == 2
        if num_jobs > 1:
            assert "foo" in job.cfg
            assert job.hydra_cfg is not None
            assert job.cfg.foo == job.hydra_cfg.hydra.job.num

    # Make sure config.yaml was created for each job
    yamls = list(Path.cwd().glob("**/.hydra/config.yaml"))
    assert len(yamls) == num_jobs

    subrocess_yamls = list(Path.cwd().glob("**/.pl_hydra_rank_1/config.yaml"))
    assert len(subrocess_yamls) == num_jobs
