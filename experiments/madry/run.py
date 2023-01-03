# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pathlib import Path

import configs
import hydra
import pytorch_lightning as pl
import torch as tr
from hydra.core.config_store import ConfigStore

from rai_toolbox.mushin.workflows import RobustnessCurve

cs = ConfigStore.instance()
cs.store(name="madry_config", node=configs.Config)


class MadryLabRobustness(RobustnessCurve):
    @staticmethod
    def pre_task(seed: int):
        pl.seed_everything(seed)

    @staticmethod
    def task(trainer: pl.Trainer, module: pl.LightningModule) -> dict:
        trainer.test(module)

        assert Path("test_metrics.pt").exists()
        return tr.load("test_metrics.pt")


@hydra.main(config_path=None, config_name="madry_config")
def main(cfg: configs.Config):
    robustness_job = MadryLabRobustness(cfg)
    robustness_job.run(epsilon=cfg.epsilon)
    robustness_job.plot("Test/Accuracy", save_filename="robustness_curve.png")


if __name__ == "__main__":
    main()
