# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Callable, Dict, Iterable

from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MeanMetric

from rai_toolbox import freeze

# Types
Criterion = Callable[[Tensor, Tensor], Tensor]
PartialOptimizer = Callable[[Iterable], Optimizer]


class UniversalPerturbationSolver(LightningModule):
    def __init__(
        self,
        *,
        dataset: DataLoader,
        val_dataset: DataLoader,
        model: nn.Module,
        perturbation: nn.Module,
        optim: PartialOptimizer,
        criterion: Criterion,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.perturbation = perturbation
        self.optim = optim
        self.criterion = criterion

        # Freeze modela set eval mode
        model.eval()
        freeze(model)
        self.model = model

        # Metrics
        self.loss_metric = MeanMetric()
        self.acc_metric = Accuracy()

    def forward(self, data: Tensor) -> Tensor:
        return self.model(self.perturbation(data))

    def train_dataloader(self) -> DataLoader:
        return self.dataset

    def val_dataloader(self) -> DataLoader:
        return self.val_dataset

    def configure_optimizers(self) -> Optimizer:
        return self.optim(self.perturbation.parameters())

    def _step(self, batch, stage: str) -> Dict[str, Tensor]:
        data, target = batch
        pdata = self.perturbation(data)
        output = self.model(pdata)
        loss = self.criterion(output, target)

        acc = self.acc_metric(output, target)
        self.log(f"{stage}/Loss", loss)
        self.log(f"{stage}/Accuracy", acc)
        return loss

    def training_step(self, batch, batch_idx) -> Tensor:
        return self._step(batch, "Train")

    def validation_step(self, batch, batch_idx) -> Tensor:
        return self._step(batch, "Val")
