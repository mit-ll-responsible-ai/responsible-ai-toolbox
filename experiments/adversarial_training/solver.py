# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Callable, Dict, Iterable, Optional

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MeanMetric

from rai_toolbox import evaluating, frozen, negate

# Types
Criterion = Callable[[Tensor, Tensor], Tensor]
PartialOptimizer = Callable[[Iterable], Optimizer]
PartialLRScheduler = Callable[[Optimizer], _LRScheduler]


class AdversarialTrainer(LightningModule):
    def __init__(
        self,
        *,
        model: nn.Module,
        optim: PartialOptimizer,
        lr_scheduler: PartialLRScheduler,
        criterion: Criterion,
        perturbation: nn.Module,
        perturb_optim: PartialOptimizer,
        perturb_steps: int,
        dataset: Optional[DataLoader] = None,
        val_dataset: Optional[DataLoader] = None,
        test_dataset: Optional[DataLoader] = None,
    ) -> None:
        """PyTorch Lightning Module for Adversarial Training

        dataset: DataLoader | None
            The data loader for training
        val_dataset: DataLoader | None
            The data loader for validation
        test_dataset: DataLoader | None
            The data loader for testing
        model: nn.Module
            The model to train
        optim: PartialOptimizer
            The optimizer for training the model
        lr_scheduler: PartialLRScheduler
            The learning rate scheduler for training the model
        criterion: Criterion
            The criterion to minimize for training
            the criterion to maximize for generating perturbations
        perturbation: nn.Module
            The perturbation model
        perturb_optim: PartialOptimizer
            The optimizer to solve for perturbation
        perturb_steps: int
            The number of steps to optimize for perturbations
        """
        super().__init__()
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.model = model

        # Perturbations and Optimizer
        self.perturbation = perturbation
        self.perturb_optim = perturb_optim
        self.perturb_steps = perturb_steps

        # Metrics
        self.loss_metric = MeanMetric()
        self.acc_metric = Accuracy()

    def forward(self, data: Tensor) -> Tensor:
        return self.model(data)

    def train_dataloader(self) -> DataLoader:
        return self.dataset

    def val_dataloader(self) -> DataLoader:
        return self.val_dataset

    def test_dataloader(self) -> DataLoader:
        return self.test_dataset

    def configure_optimizers(self) -> Dict:
        optim = self.optim(self.model.parameters())
        lrsched = self.lr_scheduler(optim)
        return dict(optimizer=optim, lr_scheduler=lrsched)

    @torch.enable_grad()
    def solve_perturbation(self, model, data, target):
        with frozen(model), evaluating(model):
            max_criterion = negate(self.criterion)
            perturber = self.perturbation(data)
            opt = self.perturb_optim(perturber.parameters())

            for _ in range(self.perturb_steps):
                output = model(perturber(data))
                loss = max_criterion(output, target)
                opt.zero_grad()
                loss.backward()
                opt.step()

        with torch.no_grad():
            data_pert = perturber(data)

        return data_pert, target

    def _step(self, batch, stage: str) -> Tensor:
        data, target = batch
        data, target = self.solve_perturbation(self.model, data, target)
        output = self.model(data)
        loss = self.criterion(output, target)
        acc = self.acc_metric(output, target)
        self.log(f"{stage}/Loss", loss)
        self.log(f"{stage}/Accuracy", acc)
        return loss

    def training_step(self, batch, batch_idx) -> Tensor:
        return self._step(batch, "Train")

    def validation_step(self, batch, batch_idx) -> Tensor:
        return self._step(batch, "Val")

    def test_step(self, batch, batch_idx) -> Tensor:
        return self._step(batch, "Test")
