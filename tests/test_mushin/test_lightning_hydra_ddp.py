# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra_zen import builds, instantiate, launch, load_from_yaml, make_config
from omegaconf.errors import ConfigAttributeError
from pytorch_lightning import Trainer

from rai_toolbox.mushin.hydra import zen
from rai_toolbox.mushin.lightning import HydraDDP
from rai_toolbox.mushin.lightning._pl_main import task as pl_main_task
from rai_toolbox.mushin.testing.lightning import TestLightningModule


def task_fn(cfg: Any):
    trainer: Trainer = instantiate(cfg.trainer)
    module = instantiate(cfg.module)
    if "run_test" in cfg and cfg.run_test:
        trainer.test(module)
    else:
        trainer.fit(module)


@pytest.mark.usefixtures("cleandir")
def test_ddp_with_hydra_raises():
    trainer = builds(
        Trainer,
        max_epochs=1,
        accelerator="auto",
        devices="${devices}",
        strategy=builds(HydraDDP),
        fast_dev_run=True,
    )
    module = builds(TestLightningModule)
    Config = make_config(trainer=trainer, wrong_config_name=module, devices=2)
    with pytest.raises(ConfigAttributeError):
        launch(Config, zen(pl_main_task))


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Need at least 2 GPUs to test"
)
@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("subdir", [None, "dksa"])
@pytest.mark.parametrize("num_jobs", [1, 2])
@pytest.mark.parametrize("testing", [True, False])
def test_ddp_with_hydra_runjob(subdir, num_jobs, testing):

    overrides = [f"+run_test={testing}"]

    if subdir is not None:
        overrides += [f"hydra.output_subdir={subdir}"]

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
    module = builds(TestLightningModule)
    Config = make_config(trainer=trainer, module=module, devices=2)
    launch(Config, task_fn, overrides, multirun=multirun)
    assert "LOCAL_RANK" not in os.environ

    # Make sure config.yaml was created for additional
    # processes.
    yamls = list(Path.cwd().glob("**/config.yaml"))
    assert len(yamls) == num_jobs

    # Make sure the parameter was set and used
    cfg = load_from_yaml(yamls[0])
    assert cfg.devices == 2

    # Make sure PL spawned a job that is logged by Hydra
    logs = list(Path.cwd().glob("**/*.log"))
    assert len(logs) == num_jobs
