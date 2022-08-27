# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""
This script is called from `rai_toolbox.mushin.lightning.launchers.HydraDDP
"""

import logging
from typing import Optional, Union

import hydra
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from ..hydra import zen

log = logging.getLogger(__name__)


def task(
    trainer: Trainer,
    module: LightningModule,
    train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
    val_dataloaders: Optional[EVAL_DATALOADERS] = None,
    dataloaders: Optional[EVAL_DATALOADERS] = None,
    datamodule: Optional[LightningDataModule] = None,
    ckpt_path: Optional[str] = None,
    return_predictions: Optional[bool] = None,
    pl_testing: bool = False,
    pl_predicting: bool = False,
    pl_local_rank: int = 0,
) -> None:
    if pl_testing:
        log.info(f"Rank {pl_local_rank}: Launched subprocess using Training.test")
        trainer.test(
            module, dataloaders=dataloaders, datamodule=datamodule, ckpt_path=ckpt_path
        )
    elif pl_predicting:
        log.info(f"Rank {pl_local_rank}: Launched subprocess using Trainer.predict")
        trainer.predict(
            module,
            dataloaders=dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            return_predictions=return_predictions,
        )
    else:
        log.info(f"Rank {pl_local_rank}: Launched subprocess using Training.fit")
        trainer.fit(
            module,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )


@hydra.main(config_path=None, config_name="config", version_base="1.1")
def main(cfg):
    zen(task)(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
