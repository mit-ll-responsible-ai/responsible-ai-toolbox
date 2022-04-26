# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""
This script is called from `rai_toolbox.mushin.lightning.launchers.HydraDDP
"""

import logging
from typing import Optional

import hydra
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

from ..hydra import zen

log = logging.getLogger(__name__)


def task(
    trainer: Trainer,
    module: LightningModule,
    datamodule: Optional[LightningDataModule] = None,
    pl_testing: bool = False,
    pl_local_rank: int = 0,
) -> None:
    if pl_testing:
        log.info(f"Rank {pl_local_rank}: Launched subprocess using Training.test")
        trainer.test(module, datamodule=datamodule)
    else:
        log.info(f"Rank {pl_local_rank}: Launched subprocess using Training.fit")
        trainer.fit(module, datamodule=datamodule)


@hydra.main(config_path=None, config_name="config")
def main(cfg):
    zen(task)(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
