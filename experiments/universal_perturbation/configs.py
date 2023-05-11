# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from hydra_zen import MISSING, builds, make_config, make_custom_builds_fn
from pytorch_lightning.callbacks import ModelCheckpoint
from solver import UniversalPerturbationSolver
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from rai_experiments.models.small_resnet import resnet50 as cifar_resnet50
from rai_toolbox import negate
from rai_toolbox.mushin import load_from_checkpoint
from rai_toolbox.mushin.lightning import HydraDDP, MetricsCallback
from rai_toolbox.optim import L1qFrankWolfe, L2ProjectedOptim
from rai_toolbox.perturbations import AdditivePerturbation
from rai_toolbox.perturbations.init import (
    uniform_like_l1_n_ball_,
    uniform_like_l2_n_ball_,
)

cs = ConfigStore.instance()
pbuilds = make_custom_builds_fn(zen_partial=True)

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
    [builds(transforms.RandomCrop, size=32, padding=4), builds(transforms.ToTensor)],
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
LoadFromCheckpoint = builds(
    load_from_checkpoint,
    model=MISSING,
    ckpt=None,
    weights_key="state_dict",
    populate_full_signature=True,
)

CIFARResNet50 = LoadFromCheckpoint(model=builds(cifar_resnet50), ckpt="${...ckpt}")
CIFARModel = builds(
    nn.Sequential,
    CIFAR10Normalizer,
    CIFARResNet50,
    zen_meta=dict(ckpt="${oc.env:HOME}/.torch/models/${ckpt}.pt"),
)

###########
# Optimizer
###########

L2PGOpt = pbuilds(
    L2ProjectedOptim,
    InnerOpt=SGD,
    lr="${lr}",
    epsilon="${epsilon}",
    param_ndim=None,  # ensure projection doesn't broadcast
)

L1qFWOpt = pbuilds(
    L1qFrankWolfe,
    q=0.975,
    lr=1.0,
    epsilon="${epsilon}",
    param_ndim=None,  # ensure grad transformation doesn't broadcast
)


L2Init = pbuilds(uniform_like_l2_n_ball_)
L1Init = pbuilds(uniform_like_l1_n_ball_)
Perturbation = builds(AdditivePerturbation, data_or_shape=(3, 32, 32), init_fn=None)

########
# Solver
########
Criterion = builds(negate, F.cross_entropy)

Solver = builds(
    UniversalPerturbationSolver,
    dataset="${dataset}",
    val_dataset="${val_dataset}",
    model="${model}",
    optim=MISSING,
    criterion=Criterion,
    perturbation=Perturbation,
)

Trainer = builds(
    pl.Trainer,
    devices="${gpus}",
    max_epochs="${max_epochs}",
    accelerator="gpu",
    strategy=builds(HydraDDP),
    callbacks=[
        builds(MetricsCallback),
        builds(
            ModelCheckpoint,
            monitor="Train/Loss",
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
)

ModelCfg = make_config(ckpt="mitll_cifar_l2_1_0", model=CIFARModel)
SolverCfg = make_config(lr=0.1, epsilon=0.0, init_fn=None, module=Solver)
TrainerCfg = make_config(max_epochs=40, devices=2, trainer=Trainer)
Config = make_config(
    defaults=[
        "_self_",
        {"module/optim": "l2pgd"},
        {"module/perturbation/init_fn": "${module/optim}"},
    ],
    random_seed=1223,
    bases=(DatasetCfg, ModelCfg, SolverCfg, TrainerCfg),
)

##########################
# Swappable Configurations
##########################
cs.store(name="l2pg", group="module/optim", node=L2PGOpt)
cs.store(name="l1qfw", group="module/optim", node=L1qFWOpt)
cs.store(name="l2pg", group="module/perturbation/init_fn", node=L2Init)
cs.store(name="l1qfw", group="module/perturbation/init_fn", node=L1Init)
