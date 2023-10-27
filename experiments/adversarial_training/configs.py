# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import pytorch_lightning as pl
from hydra_zen import builds, make_config, make_custom_builds_fn
from pytorch_lightning.callbacks import ModelCheckpoint
from solver import AdversarialTrainer
from torch import nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from rai_experiments.models.small_resnet import resnet50 as cifar_resnet50
from rai_toolbox.mushin._utils import load_from_checkpoint
from rai_toolbox.mushin.lightning import HydraDDP, MetricsCallback
from rai_toolbox.optim import L2ProjectedOptim
from rai_toolbox.perturbations.models import AdditivePerturbation

##################
# Custom Functions
##################
pbuilds = make_custom_builds_fn(zen_partial=True)


def get_stepsize(factor, steps, epsilon):
    return factor * epsilon / steps


#########
# Dataset
#########
CIFAR10Normalizer = builds(
    transforms.Normalize,
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)

CIFAR10Transforms = builds(
    transforms.Compose,
    [
        builds(transforms.RandomCrop, size=32, padding=4),
        builds(transforms.RandomHorizontalFlip),
        builds(transforms.ColorJitter, brightness=0.25, contrast=0.25, saturation=0.25),
        builds(transforms.RandomRotation, degrees=2),
        builds(transforms.ToTensor),
    ],
)

CIFAR10 = builds(
    datasets.CIFAR10,
    root="${oc.env:HOME}/.torch/data",
    train=True,
    download=True,
    transform=CIFAR10Transforms,
)

TestCIFAR10 = builds(
    datasets.CIFAR10,
    root="${oc.env:HOME}/.torch/data",
    train=False,
    download=True,
    transform=builds(transforms.ToTensor),
)

#######
# Model
#######
ResNet50 = builds(
    load_from_checkpoint,
    model=builds(cifar_resnet50),
    ckpt="${ckpt}",
    weights_key="state_dict",
    weights_key_strip="model.1",
    populate_full_signature=True,
)

CIFARModel = builds(nn.Sequential, CIFAR10Normalizer, ResNet50)

###########
# Optimizer
###########
# Model Optimizers
SGDOpt = pbuilds(
    SGD, lr="${lr}", momentum="${momentum}", weight_decay="${weight_decay}"
)
StepLR = pbuilds(lr_scheduler.StepLR, step_size="${step_size}", gamma="${gamma}")

# PGD Optimizer and Model
L2PGOpt = pbuilds(
    L2ProjectedOptim,
    InnerOpt=SGD,
    lr=builds(get_stepsize, factor=2.5, steps="${perturb_steps}", epsilon="${epsilon}"),
    epsilon="${epsilon}",
)

Perturbation = pbuilds(AdditivePerturbation, populate_full_signature=True)

########
# Solver
########
Solver = builds(
    AdversarialTrainer,
    dataset="${dataset}",
    val_dataset="${val_dataset}",
    test_dataset="${test_dataset}",
    model="${model}",
    optim=SGDOpt,
    lr_scheduler=StepLR,
    criterion=builds(nn.CrossEntropyLoss),
    perturbation=Perturbation,
    perturb_optim=L2PGOpt,
    perturb_steps="${perturb_steps}",
)

Trainer = builds(
    pl.Trainer,
    devices="${gpus}",
    max_epochs="${max_epochs}",
    strategy=builds(HydraDDP),
    callbacks=[
        builds(MetricsCallback),
        builds(
            ModelCheckpoint,
            monitor="Val/Loss",
            save_last=True,
            save_top_k=1,
            filename="epoch_{epoch}",
            auto_insert_metric_name=False,
        ),
    ],
    populate_full_signature=True,
)


#########
# Configs
#########
DatasetCfg = make_config(
    batch_size=128,
    num_workers=8,
    dataset=builds(
        DataLoader,
        CIFAR10,
        batch_size="${batch_size}",
        num_workers="${num_workers}",
        shuffle=True,
        pin_memory=True,
    ),
    val_dataset=builds(
        DataLoader,
        TestCIFAR10,
        batch_size="${batch_size}",
        num_workers="${num_workers}",
        pin_memory=True,
    ),
    test_dataset=None,
)

ModelCfg = make_config(ckpt=None, model=CIFARModel)

ModuleCfg = make_config(
    # SGD Parameters
    momentum=0.9,
    lr=0.1,
    weight_decay=5e-4,
    # StepLR Parameters
    step_size=50,
    gamma=0.1,
    # PGD Parameters
    perturb_steps=7,
    epsilon=1.0,
    # The LightningModule
    module=Solver,
)

TrainerCfg = make_config(max_epochs=100, devices=2, trainer=Trainer)

Config = make_config(
    random_seed=234,
    bases=(DatasetCfg, ModelCfg, ModuleCfg, TrainerCfg),
)
