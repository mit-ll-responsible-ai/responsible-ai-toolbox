# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import configs
import hydra
from hydra.conf import HydraConf, RunDir, SweepDir
from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate
from pytorch_lightning import seed_everything

Hydra = HydraConf(
    run=RunDir("outputs/epsilon_${epsilon}/${now:%Y-%m-%d}/${now:%H-%M-%S}"),
    sweep=SweepDir(
        dir="multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}",
        subdir="epsilon_${epsilon}",
    ),
)

cs = ConfigStore.instance()
cs.store(name="adversarial_training_config", node=configs.Config)
cs.store(name="config", group="hydra", node=Hydra)


@hydra.main(config_path=None, config_name="adversarial_training_config")
def task_fn(cfg):
    seed_everything(cfg.random_seed)
    module = instantiate(cfg.module)
    trainer = instantiate(cfg.trainer)
    trainer.fit(module)


if __name__ == "__main__":
    task_fn()
