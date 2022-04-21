# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""
This script is called from `rai_toolbox.mushin.lightning.launchers.HydraDDP
"""

import logging

import hydra
from pytorch_lightning import LightningModule, Trainer

from ..hydra import zen

log = logging.getLogger(__name__)


def task(trainer: Trainer, module: LightningModule, pl_testing: bool) -> None:
    if pl_testing:
        log.info("Launching child process using Training.test")
        trainer.test(module)
    else:
        log.info("Launching child process using Training.fit")
        trainer.fit(module)


@hydra.main(config_path=None, config_name="config")
def main(cfg):
    zen(task)(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
