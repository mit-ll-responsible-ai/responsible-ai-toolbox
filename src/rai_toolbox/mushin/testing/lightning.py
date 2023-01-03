# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torch.nn import functional as fnn
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SimpleLightningModuleNoData(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return fnn.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        self.log("fit_item_tensor_metric", torch.tensor(1.0))
        self.log("fit_tensor_metric", torch.tensor([1.0]))
        self.log("fit_number_metric", 1.0)
        return loss

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        self.log("val_item_tensor_metric", torch.tensor(1.0))
        self.log("val_tensor_metric", torch.tensor([1.0]))
        self.log("val_number_metric", 1.0)
        return loss

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        self.log("test_item_tensor_metric", torch.tensor(1.0))
        self.log("test_tensor_metric", torch.tensor([1.0]))
        self.log("test_number_metric", 1.0)
        return loss

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


class SimpleLightningModule(SimpleLightningModuleNoData):
    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))


class SimpleDataModule(LightningDataModule):
    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))
