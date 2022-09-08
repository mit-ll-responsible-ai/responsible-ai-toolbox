from hydra_zen import instantiate
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from typing_extensions import Literal


def pl_pre_task(cfg):
    seed_everything(cfg.random_seed)


def pl_task_fn(cfg):
    trainer: Trainer = instantiate(cfg.trainer)
    module = instantiate(cfg.module)
    assert "_trainer_state" in cfg

    if cfg._trainer_state == "testing":
        trainer.test(module)

    if cfg._trainer_state == "predicting":
        trainer.predict(module)

    if cfg._trainer_state == "training":
        trainer.fit(module)


def pl_task_fn_with_datamodule(cfg):
    trainer: Trainer = instantiate(cfg.trainer)
    module = instantiate(cfg.module)
    datamodule = instantiate(cfg.datamodule)
    assert "_trainer_state" in cfg

    if cfg._trainer_state == "testing":
        trainer.test(module, datamodule=datamodule)

    if cfg._trainer_state == "predicting":
        trainer.predict(module, datamodule=datamodule)

    if cfg._trainer_state == "training":
        trainer.fit(module, datamodule=datamodule)


def zen_pl_pre_task(random_seed: int):
    seed_everything(random_seed)


def zen_pl_task_fn(
    trainer: Trainer,
    module: LightningModule,
    _trainer_state: Literal["training", "testing", "predicting"],
):
    if _trainer_state == "testing":
        trainer.test(module)

    if _trainer_state == "predicting":
        trainer.predict(module)

    if _trainer_state == "training":
        trainer.fit(module)


def zen_pl_task_fn_with_datamodule(
    trainer: Trainer,
    module: LightningModule,
    datamodule: LightningDataModule,
    _trainer_state: Literal["training", "testing", "predicting"],
):
    if _trainer_state == "testing":
        trainer.test(module, datamodule=datamodule)

    if _trainer_state == "predicting":
        trainer.predict(module, datamodule=datamodule)

    if _trainer_state == "training":
        trainer.fit(module, datamodule=datamodule)


def zen_pl_all_task_fn(trainer: Trainer, module: LightningModule):
    trainer.fit(module)
    trainer.test(module)
    trainer.predict(module)
