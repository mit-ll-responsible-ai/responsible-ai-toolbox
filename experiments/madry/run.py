# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pathlib import Path

import configs
import hydra
import pytorch_lightning as pl
import torch
from hydra.core.config_store import ConfigStore

from rai_toolbox.mushin.workflows import RobustnessCurve

cs = ConfigStore.instance()
cs.store(name="madry_config", node=configs.Config)


class MadryLabRobustness(RobustnessCurve):
    @staticmethod
    def evaluation_task(
        seed: int, trainer: pl.Trainer, module: pl.LightningModule
    ) -> torch.Tensor:
        pl.seed_everything(seed)
        trainer.test(module)

        assert Path("test_metrics.pt").exists()
        metrics = torch.load("test_metrics.pt")
        return metrics["Test/Accuracy"][-1]


@hydra.main(config_path=None, config_name="madry_config")
def main(cfg: configs.Config):
    robustness_job = MadryLabRobustness(cfg)
    robustness_job.run(job_epsilons=cfg.job_epsilons)
    robustness_job.plot()


if __name__ == "__main__":
    main()
