# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as fnn


class RandomDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class TestLightningModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return fnn.mse_loss(prediction, torch.ones_like(prediction))

    def step(self, x):
        x = self(x)
        out = fnn.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        self.log("Tensor Metric", torch.tensor(1.0))
        self.log("Tensor 1 Metric", torch.tensor([1.0]))
        self.log("Number Metric", 1.0)
        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outputs) -> None:
        torch.stack([x["loss"] if isinstance(x, dict) else x for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        self.log("Val Tensor Metric", torch.tensor(1.0))
        self.log("Val Tensor 1 Metric", torch.tensor([1.0]))
        self.log("Val Number Metric", 1.0)
        return {"x": loss}

    def validation_epoch_end(self, outputs) -> None:
        torch.stack([x["loss"] if isinstance(x, dict) else x for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        self.log("Tensor Metric", torch.tensor(1.0))
        self.log("Tensor 1 Metric", torch.tensor([1.0]))
        self.log("Number Metric", 1.0)
        return {"y": loss}

    def test_epoch_end(self, outputs) -> None:
        torch.stack([x["y"] if isinstance(x, dict) else x for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))
