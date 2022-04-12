# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import logging
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning import Callback, LightningModule
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor, nn
from torch.utils.data.dataloader import DataLoader
from torchmetrics import MetricCollection

from .typing import (
    Criterion,
    PartialLightningOptimizer,
    Perturbation,
    PLOptim,
    Predictor,
)

log = logging.getLogger(__name__)


# A Generic Lightning Module
class BaseMushinModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optim: Optional[PartialLightningOptimizer] = None,
        criterion: Optional[Criterion] = None,
        predictor: Optional[Predictor] = None,
        metrics: Optional[MetricCollection] = None,
        perturbation: Optional[Perturbation] = None,
        dataset: Optional[DataLoader] = None,
        val_dataset: Optional[DataLoader] = None,
        test_dataset: Optional[DataLoader] = None,
    ):
        super().__init__()
        self.model = model
        self.optim = dict(optimizer=optim) if callable(optim) else optim
        self.perturbation = perturbation
        self.criterion = criterion
        self.predictor = predictor
        self.metrics = metrics
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train_dataloader(self) -> DataLoader:
        return self.dataset

    def val_dataloader(self) -> DataLoader:
        return self.val_dataset

    def test_dataloader(self) -> DataLoader:
        return self.test_dataset

    def configure_optimizers(self) -> List[PLOptim]:
        optimizers: List[PLOptim] = []

        if not isinstance(self.optim, list):
            self.optim = [self.optim]

        for o in self.optim:
            conf_opt: Dict[str, Any] = dict(frequency=o.get("frequency", 1))

            assert "optimizer" in o and o["optimizer"] is not None
            conf_opt["optimizer"] = o["optimizer"](self.model.parameters())

            if "lr_scheduler" in o and o["lr_scheduler"] is not None:
                conf_opt["lr_scheduler"] = o["lr_scheduler"](conf_opt["optimizer"])

            optimizers.append(conf_opt)
        return optimizers

    def step(
        self, batch: Tuple[Tensor, Tensor], stage: str
    ) -> Dict[str, Union[Tensor, Tuple[Tensor, Tensor]]]:
        data, target = batch

        # Modify data for training
        if self.perturbation is not None:
            data, _ = self.perturbation(model=self.model, data=data, target=target)

        # If criterion is not defined assume the model reports
        # the loss and pred
        if self.criterion is None:
            loss, pred = self.model(data, target)

        elif self.criterion is not None:
            output = self.model(data)
            loss = self.criterion(output, target)

            # No predictor assumes the output is the desired prediction
            if self.predictor is not None:
                pred = self.predictor(output)
            else:
                pred = output
        else:
            assert False  # unreachable

        self.log(f"{stage}/Loss", loss)
        return dict(loss=loss, results=(pred, target))

    def training_step(self, batch, *args, **kwargs):
        loss = self.step(batch, "Train")
        return loss

    def validation_step(self, batch, *args, **kwargs):
        loss = self.step(batch, "Val")
        return loss

    def test_step(self, batch, *args, **kwargs):
        loss = self.step(batch, "Test")
        return loss

    def update_metrics(self, outputs: Dict, stage: str = "Train"):
        if self.metrics is not None and isinstance(self.metrics, MetricCollection):
            results = outputs["results"]
            for key, metric in self.metrics.items():
                val = metric(*results)
                if isinstance(val, Tensor) and val.ndim == 0:
                    self.log(f"{stage}/{key}", val)

    # Handle metrics for both DP and DDP accelerators
    def training_step_end(self, outputs: Union[Tensor, Dict]) -> Tensor:
        if isinstance(outputs, dict):
            loss = outputs["loss"].mean()
            self.update_metrics(outputs, stage="Train")
            return loss
        return outputs

    def validation_step_end(self, outputs: Union[Tensor, Dict]) -> Tensor:
        if isinstance(outputs, dict):
            loss = outputs["loss"].mean()
            self.update_metrics(outputs, stage="Val")
            return loss
        return outputs

    def test_step_end(self, outputs: Union[Tensor, Dict]) -> Tensor:
        if isinstance(outputs, dict):
            loss = outputs["loss"].mean()
            self.update_metrics(outputs, stage="Test")
            return loss
        return outputs


# Use this for saving metrics
class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.test_metrics = defaultdict(list)

    # TODO: Need to save training metrics when the trainer
    # does not execute `on_validation_end` (no validation dataset)

    def on_validation_end(self, trainer, pl_module):
        # Make sure PL is not doing it's sanity check run
        if trainer.sanity_checking:
            return self.val_metrics

        metrics = trainer.callback_metrics
        self.val_metrics["epoch"].append(pl_module.current_epoch)
        for k, v in metrics.items():
            if hasattr(v, "item"):
                v = v.item()
            self.val_metrics[k].append(v)

        torch.save(self.val_metrics, "fit_metrics.pt")
        return self.val_metrics

    def on_test_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        for k, v in metrics.items():
            if isinstance(v, Tensor):
                v = v.item() if v.ndim == 0 else v.detach().cpu().numpy()
            self.test_metrics[k].append(v)

        torch.save(self.test_metrics, "test_metrics.pt")
        return self.test_metrics


class CustomDDP(DDPPlugin):
    def setup_environment(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().setup_environment()

    def _call_children_scripts(self):
        # bookkeeping of spawned processes
        self._check_can_spawn_children()

        # DDP Environment variables
        os.environ["MASTER_ADDR"] = self.cluster_environment.master_address()
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())

        # allow the user to pass the node rank
        os.environ["NODE_RANK"] = str(self.cluster_environment.node_rank())
        os.environ["LOCAL_RANK"] = str(self.cluster_environment.local_rank())

        # the visible devices tell us how many GPUs we want to use.
        # when the trainer script was called the device has already been scoped by the time
        # code reaches this point. so, to call the scripts, we need to leave cuda visible devices alone
        # but forward the GPUs selected via environment variables
        if self.parallel_devices is None:
            raise MisconfigurationException(
                "you selected (distribute_backend = ddp) but did not set Trainer(gpus=?)"
            )

        os.environ["WORLD_SIZE"] = f"{self.num_processes * self.num_nodes}"

        self.interactive_ddp_procs = []

        for local_rank in range(1, self.num_processes):
            env_copy = os.environ.copy()
            env_copy["LOCAL_RANK"] = f"{local_rank}"

            # remove env var if global seed not set
            if (
                os.environ.get("PL_GLOBAL_SEED") is None
                and "PL_GLOBAL_SEED" in env_copy
            ):
                del env_copy["PL_GLOBAL_SEED"]

            # start process
            # if hydra is available and initialized, make sure to set the cwd correctly
            cwd = os.getcwd()
            os_cwd = f'"{cwd}"'  # this is needed to handle characters like `=` in the directory name

            trainer_fn = self.lightning_module.trainer.state.fn
            command = [sys.executable, "-m", "rai_toolbox.mushin._pl_main"]

            v1 = Path(cwd) / "ddp_config.yaml"
            if v1.exists():
                command += ["-cp", cwd, "-cn", "ddp_config.yaml"]
            else:
                hydra_cfg = HydraConfig.get()
                hydra_output = os.path.join(cwd, hydra_cfg.output_subdir)
                command += ["-cp", hydra_output, "-cn", "config.yaml"]

            if trainer_fn == TrainerFn.FITTING:
                command += ["+_ddp_testing=false"]
            else:
                command += ["+_ddp_testing=true"]

            command += [
                f"hydra.output_subdir=.pl_hydra_{local_rank}",
                f"hydra.run.dir={os_cwd}",
                f"hydra.job.name=train_ddp_process_{local_rank}",
            ]

            proc = subprocess.Popen(command, env=env_copy, cwd=cwd)
            self.interactive_ddp_procs.append(proc)

            # starting all processes at once can cause issues
            # with dataloaders delay between 1-10 seconds
            delay = np.random.uniform(1, 5, 1)[0]
            sleep(delay)

        self._rank_0_has_called_call_children_scripts = True

    def teardown(self) -> None:
        """Performs additional teardown steps for PL to allow for Hydra multirun jobs."""
        super().teardown()

        # Remove PL environments so next multirun starts fresh
        envs = (
            "LOCAL_RANK",
            "NODE_RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        )

        for name in envs:
            os.environ.pop(name, None)
