# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from functools import partial
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from rai_toolbox.mushin import trainer

pytorch_userwarning_filters = pytest.mark.filterwarnings(
    "ignore:The dataloader",
    "ignore: The number of training samples",
    "ignore: GPU available but not used",
)


@pytorch_userwarning_filters
@pytest.mark.usefixtures("cleandir")
def test_trainer_training():
    data = torch.rand(100, 10)
    target = torch.rand(100, 2)
    model = nn.Sequential(nn.Linear(10, 10), nn.Tanh(), nn.Linear(10, 2))
    optim = partial(torch.optim.SGD, lr=0.1)
    criterion = nn.MSELoss()
    predictor = nn.Identity()

    dataset = TensorDataset(data, target)
    trainer(
        dataset=dataset,
        val_dataset=dataset,
        model=model,
        optim=optim,
        criterion=criterion,
        predictor=predictor,
        num_workers=0,
        batch_size=50,
        gpus=0,
    )
    assert Path("fit_metrics.pt").exists()


@pytorch_userwarning_filters
@pytest.mark.usefixtures("cleandir")
def test_trainer_testing():
    data = torch.rand(100, 10)
    target = torch.rand(100, 2)
    model = nn.Sequential(nn.Linear(10, 10), nn.Tanh(), nn.Linear(10, 2))
    criterion = nn.MSELoss()
    predictor = nn.Identity()

    dataset = TensorDataset(data, target)
    trainer(
        dataset=dataset,
        model=model,
        criterion=criterion,
        predictor=predictor,
        num_workers=0,
        batch_size=50,
        gpus=0,
    )
    assert Path("test_metrics.pt").exists()
