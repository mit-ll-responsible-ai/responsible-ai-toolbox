# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import builds, instantiate, launch, make_config
from pytorch_lightning import Trainer

from rai_toolbox.mushin.hydra import MushinPickleJobCallback, zen
from rai_toolbox.mushin.lightning import HydraRerunDDP
from rai_toolbox.mushin.testing.hydra_task_fns import (
    zen_pl_all_task_fn,
    zen_pl_pre_task,
    zen_pl_task_fn_with_datamodule,
)
from rai_toolbox.mushin.testing.lightning import SimpleDataModule, SimpleLightningModule
from rai_toolbox.mushin.workflows import _task_calls

if torch.cuda.device_count() < 2:
    pytest.skip("Need at least 2 GPUs to test", allow_module_level=True)


TrainerConfig = builds(
    Trainer,
    max_epochs=1,
    accelerator="auto",
    devices="${devices}",
    strategy=builds(HydraRerunDDP),
    fast_dev_run=True,
)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Need at least 2 GPUs to test"
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("trainer_state", ["training", "testing", "predicting"])
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_hydra_rerun_ddp(trainer_state, num_jobs):

    task_fn_cfg = builds(
        _task_calls,
        pre_task=builds(zen, zen_pl_pre_task),
        task=builds(zen, zen_pl_task_fn_with_datamodule),
    )

    callback_cfg = dict(
        save_job_info=builds(MushinPickleJobCallback, task_fn=task_fn_cfg)
    )
    cs = ConfigStore.instance()
    cs.store(name="pickle_job", group="hydra/callbacks", node=callback_cfg)

    module = builds(SimpleLightningModule)
    datamodule = builds(SimpleDataModule)
    Config = make_config(
        random_seed=1,
        trainer=TrainerConfig,
        module=module,
        datamodule=datamodule,
        devices=2,
        _trainer_state=trainer_state,
    )

    fake_param = "+foo=" + ",".join(str(i) for i in range(num_jobs))
    multirun = True if num_jobs > 1 else False

    task_fn = instantiate(task_fn_cfg)
    launch(
        Config,
        task_fn,
        overrides=["hydra/callbacks=pickle_job", fake_param],
        multirun=multirun,
    )

    pickles = sorted(Path.cwd().glob("**/.hydra/config.pickle"))
    assert len(pickles) == num_jobs

    pickles = sorted(Path.cwd().glob("**/.hydra/task_fn.pickle"))
    assert len(pickles) == num_jobs


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Need at least 2 GPUs to test"
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_hydra_rerun_ddp_all_states(num_jobs):

    task_fn_cfg = builds(
        _task_calls,
        pre_task=builds(zen, zen_pl_pre_task),
        task=builds(zen, zen_pl_all_task_fn),
    )

    callback_cfg = dict(
        save_job_info=builds(MushinPickleJobCallback, task_fn=task_fn_cfg)
    )
    cs = ConfigStore.instance()
    cs.store(name="pickle_job", group="hydra/callbacks", node=callback_cfg)

    module = builds(SimpleLightningModule)
    datamodule = builds(SimpleDataModule)
    Config = make_config(
        random_seed=1,
        trainer=TrainerConfig,
        module=module,
        datamodule=datamodule,
        devices=2,
    )

    fake_param = "+foo=" + ",".join(str(i) for i in range(num_jobs))
    multirun = True if num_jobs > 1 else False

    task_fn = instantiate(task_fn_cfg)
    launch(
        Config,
        task_fn,
        overrides=["hydra/callbacks=pickle_job", fake_param],
        multirun=multirun,
    )

    pickles = sorted(Path.cwd().glob("**/.hydra/config.pickle"))
    assert len(pickles) == num_jobs

    pickles = sorted(Path.cwd().glob("**/.hydra/task_fn.pickle"))
    assert len(pickles) == num_jobs
