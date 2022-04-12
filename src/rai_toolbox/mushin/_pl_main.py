# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Type

import hydra
from hydra_zen import instantiate
from hydra_zen.typing import Builds
from pytorch_lightning import LightningModule, Trainer


@dataclass
class LightningConfig:
    trainer: Builds[Type[Trainer]]
    module: Builds[Type[LightningModule]]
    _ddp_testing: bool


def task(cfg: LightningConfig) -> None:
    trainer = instantiate(cfg.trainer)
    module = instantiate(cfg.module)

    if cfg._ddp_testing:
        trainer.test(module)
    else:
        trainer.fit(module)


@hydra.main(config_path=None, config_name="config")
def main(cfg):
    task(cfg)


if __name__ == "__main__":
    main()
