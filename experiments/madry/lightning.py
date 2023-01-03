# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from typing import Callable, Tuple

from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy

# Types
Criterion = Callable[[Tensor, Tensor], Tensor]
Perturbation = Callable[[nn.Module, Tensor, Tensor], Tuple[Tensor, Tensor]]


class EvaluateModule(LightningModule):
    def __init__(
        self,
        *,
        model: nn.Module,
        criterion: Criterion,
        perturbation: Perturbation,
        dataset: Dataset,
        batch_size: int = 1,
        num_workers: int = 0,
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
        criterion: Criterion
            The criterion to minimize for training
            the criterion to maximize for generating perturbations
        perturbation: Callable[[Module, Tensor, Tensor], Tuple[Tensor, Tensor]]
            The perturbation model
        """
        super().__init__()
        self.dataset = dataset
        self.criterion = criterion
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Perturbations and Optimizer
        self.solve_perturbation = perturbation

        # Metrics
        self.acc_metric = Accuracy()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_step(self, batch, *args, **kwargs) -> Tensor:
        data, target = batch
        data, _ = self.solve_perturbation(model=self.model, data=data, target=target)
        output = self.model(data)
        loss = self.criterion(output, target)
        acc = self.acc_metric(output, target)
        self.log("Test/Loss", loss)
        self.log("Test/Accuracy", acc)
        return loss
