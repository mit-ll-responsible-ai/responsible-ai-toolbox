# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any

import pytest
import torch
from hydra_zen import builds, instantiate, launch, load_from_yaml, make_config
from omegaconf.errors import ConfigAttributeError
from pytorch_lightning import Trainer

from rai_toolbox.mushin.lightning import HydraDDP
from rai_toolbox.mushin.testing.lightning import SimpleDataModule, SimpleLightningModule


def task_fn(cfg: Any):
    trainer: Trainer = instantiate(cfg.trainer)
    module = instantiate(cfg.module)
    if "run_test" in cfg and cfg.run_test:
        trainer.test(module)
    else:
        trainer.fit(module)


def task_fn_with_datamodule(cfg: Any):
    trainer: Trainer = instantiate(cfg.trainer)
    module = instantiate(cfg.module)
    datamodule = instantiate(cfg.datamodule)
    if "run_test" in cfg and cfg.run_test:
        trainer.test(module, datamodule=datamodule)
    else:
        trainer.fit(module, datamodule=datamodule)


def task_fn_raises(cfg: Any):
    trainer: Trainer = instantiate(cfg.trainer)
    module = instantiate(cfg.wrong_config_name)
    if "run_test" in cfg and cfg.run_test:
        trainer.test(module)
    else:
        trainer.fit(module)


@pytest.mark.usefixtures("cleandir")
def test_ddp_with_hydra_raises_misconfiguration():
    trainer = builds(
        Trainer,
        max_epochs=1,
        accelerator="auto",
        devices="${devices}",
        strategy=builds(HydraDDP),
        fast_dev_run=True,
    )
    module = builds(SimpleLightningModule)
    Config = make_config(trainer=trainer, wrong_config_name=module, devices=2)
    with pytest.raises(ConfigAttributeError):
        launch(Config, task_fn_raises)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Need at least 2 GPUs to test"
)
@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("subdir", [None, ".hydra", "dksa"])
def test_ddp_with_hydra_output_subdir(subdir):
    overrides = []

    if subdir is not None:
        overrides += [f"hydra.output_subdir={subdir}"]

    trainer = builds(
        Trainer,
        max_epochs=1,
        accelerator="auto",
        devices="${devices}",
        strategy=builds(HydraDDP),
        fast_dev_run=True,
    )
    module = builds(SimpleLightningModule)
    Config = make_config(trainer=trainer, module=module, devices=2)
    job = launch(Config, task_fn, overrides)

    if subdir is None:
        subdir = ".hydra"

    cfg_file = Path(job.working_dir) / subdir / "config.yaml"
    assert cfg_file.exists()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Need at least 2 GPUs to test"
)
@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("testing", [True, False])
def test_ddp_with_hydra_with_datamodule(testing):
    overrides = [f"+run_test={testing}"]
    trainer = builds(
        Trainer,
        max_epochs=1,
        accelerator="auto",
        devices="${devices}",
        strategy=builds(HydraDDP),
        fast_dev_run=True,
    )
    module = builds(SimpleLightningModule)
    datamodule = builds(SimpleDataModule)
    Config = make_config(
        trainer=trainer, module=module, datamodule=datamodule, devices=2
    )
    launch(Config, task_fn_with_datamodule, overrides=overrides)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Need at least 2 GPUs to test"
)
@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("num_jobs", [1, 2])
@pytest.mark.parametrize("testing", [True, False])
def test_ddp_with_hydra_runjob(num_jobs, testing):

    overrides = [f"+run_test={testing}"]

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

    trainer = builds(
        Trainer,
        max_epochs=1,
        accelerator="auto",
        devices="${devices}",
        strategy=builds(HydraDDP),
        fast_dev_run=True,
    )
    module = builds(SimpleLightningModule)
    Config = make_config(trainer=trainer, module=module, devices=2)
    launch(Config, task_fn, overrides, multirun=multirun)

    # Make sure config.yaml was created for each job
    yamls = list(Path.cwd().glob("**/config.yaml"))
    assert len(yamls) == num_jobs

    # Make sure the parameter was set and used
    for yaml in yamls:
        cfg = load_from_yaml(yaml)
        assert cfg.devices == 2
