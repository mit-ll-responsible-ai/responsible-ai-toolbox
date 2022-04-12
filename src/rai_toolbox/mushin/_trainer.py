# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Any, List, Optional, Union

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.collections import MetricCollection

from ._lightning import BaseMushinModule, MetricsCallback
from .typing import Criterion, PartialOptimizer, PartialPLOptim, Perturbation, Predictor


# A generic, simple trainer that uses Lightning without the user needing to use it
def trainer(
    *,
    dataset: Union[Dataset, DataLoader, LightningDataModule],
    val_dataset: Optional[Union[Dataset, DataLoader]] = None,
    model: Union[nn.Module, LightningModule],
    optim: Optional[
        Union[PartialOptimizer, PartialPLOptim, List[PartialPLOptim]]
    ] = None,
    perturbation: Optional[Perturbation] = None,
    criterion: Optional[Criterion] = None,
    predictor: Optional[Predictor] = None,
    metrics: Optional[MetricCollection] = None,
    batch_size: int = 128,
    num_workers: int = 4,
    **lightning_trainer_kwargs: Any,
):
    """Main entry point function for mushin.

    Supports training and testing using PyTorch Lightning backend to automatically
    support distributed processing.


    Parameters
    ----------
    dataset: Dataset | DataLoader | LightningDataModule
        The dataset using for training or testing

    val_dataset: Dataset | DataLoader | None
        If training, this dataset is used for validation

    model: nn.Module | LightningModule
        The model to train or test

    optim: PartialOptimizer | PartialPLOptim | List[PartialPLOptim] | None
        If None, assume we are testing only.

    perturbation: Perturbation | None
        Perturbation function to apply to model, data, or target.

    criterion: Criterion | None
        The criterion to use for the loss calculation.  If None, assume
            `loss, prediction = model(data, target)`

    predictor: Predictor | None
        If criterion is not None, the function to use to map model outputs to predictions
        for metric calculations.

    metrics: MetricCollection | None
        The metrics to use for training, validation, and testing.

    batch_size: int (default: 128)
        The batch size to use for training/validaton or testing

    num_workers: int (default: 4)


    **lightning_trainer_kwargs: Any
        Any additional keywords are passed to initialize `pl.Trainer`.
    """

    # Determine if we are test or training mode
    testing: bool = optim is None
    if isinstance(model, LightningModule):
        testing = model.optim is None

    # Extract LightningDataModule
    data_module = _get_data_module(
        dataset, val_dataset, testing, batch_size, num_workers
    )

    # Create LightningModule
    pl_model = model
    if not isinstance(model, LightningModule):
        if not testing:
            pl_model = BaseMushinModule(
                model,
                optim=optim,
                criterion=criterion,
                predictor=predictor,
                metrics=metrics,
                perturbation=perturbation,
            )
        else:
            pl_model = BaseMushinModule(
                model,
                criterion=criterion,
                predictor=predictor,
                metrics=metrics,
                perturbation=perturbation,
            )

    # Provide default model checkpoint and metrics callbacks
    callbacks = lightning_trainer_kwargs.pop("callbacks", [])
    if callbacks is None:
        callbacks = []

    has_metricscb = False
    has_modelckptcb = False
    for cb in callbacks:
        if isinstance(cb, MetricsCallback):
            has_metricscb = True
        if isinstance(cb, ModelCheckpoint):
            has_modelckptcb = True

    # Add default model checkpoint
    if not has_modelckptcb:
        callbacks.append(
            ModelCheckpoint(
                monitor="Train/Loss",
                save_last=True,
                save_top_k=1,
                filename="epoch_{epoch}",
                auto_insert_metric_name=False,
            )
        )

    # Add default callback to save metrics to local file
    if not has_metricscb:
        callbacks.append(MetricsCallback())

    # Execut Lighting Trainer
    trainer = Trainer(callbacks=callbacks, **lightning_trainer_kwargs)

    if testing:
        trainer.test(pl_model, datamodule=data_module)
    else:
        trainer.fit(pl_model, datamodule=data_module)


def _get_data_module(
    dataset: Union[LightningDataModule, DataLoader, Dataset],
    val_dataset: Optional[Union[Dataset, DataLoader]],
    testing: bool,
    batch_size: int,
    num_workers: int,
) -> LightningDataModule:
    if isinstance(dataset, LightningDataModule):
        return dataset

    elif isinstance(dataset, DataLoader):

        class MushinDataModule(LightningDataModule):
            def __init__(self, dl, val_dl=None):
                super().__init__()
                self.train_dl = None
                self.val_dl = val_dl
                self.test_dl = None

                if testing:
                    self.test_dl = dl
                else:
                    self.train_dl = dl
                    if val_dl is None:
                        self.val_dl = dl

            def train_dataloader(self) -> Optional[DataLoader]:
                return self.train_dl

            def val_dataloader(self) -> Optional[DataLoader]:
                return self.val_dl

            def test_dataloader(self) -> Optional[DataLoader]:
                return self.test_dl

        return MushinDataModule(dataset, val_dataset)

    else:
        dataset_kw = (
            dict(test_dataset=dataset)
            if testing
            else dict(train_dataset=dataset, val_dataset=val_dataset)
        )
        return LightningDataModule.from_datasets(
            batch_size=batch_size,
            num_workers=num_workers,
            **dataset_kw,
        )
