# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import MISSING, make_config, make_custom_builds_fn
from lightning import EvaluateModule
from torch import nn
from torch.optim import SGD
from torchvision import datasets, models, transforms

from rai_experiments.models.small_resnet import resnet50 as cifar_resnet50
from rai_toolbox import datasets as rai_datasets
from rai_toolbox.mushin import load_from_checkpoint
from rai_toolbox.mushin.lightning import HydraDDP, MetricsCallback
from rai_toolbox.optim import L2ProjectedOptim
from rai_toolbox.perturbations import AdditivePerturbation, gradient_ascent
from rai_toolbox.perturbations.init import uniform_like_l2_n_ball_

###############
# Custom Builds
###############
builds = make_custom_builds_fn()
pbuilds = make_custom_builds_fn(zen_partial=True)

#########
# Dataset
#########
CIFAR10Normalizer = builds(
    transforms.Normalize,
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)

CIFAR10 = builds(
    datasets.CIFAR10,
    root="${data_path}",
    train=False,
    download=True,
    transform=builds(transforms.ToTensor),
)

ImageNetNormalizer = builds(
    transforms.Normalize,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

TestTransformImageNet = builds(
    transforms.Compose,
    [
        builds(transforms.Resize, 256),
        builds(transforms.CenterCrop, 224),
        builds(transforms.ToTensor),
    ],
)

RestrictedImageNet = builds(
    rai_datasets.RestrictedImageNet,
    root="${data_path}",
    transform=TestTransformImageNet,
)


#######
# Model
#######
LoadFromCheckpoint = builds(
    load_from_checkpoint,
    model=MISSING,
    ckpt="${ckpt}",
    weights_key="state_dict",
    populate_full_signature=True,
)

CIFARResNet50 = builds(
    load_from_checkpoint,
    model=builds(cifar_resnet50),
    builds_bases=(LoadFromCheckpoint,),
)

CIFARModel = builds(nn.Sequential, CIFAR10Normalizer, CIFARResNet50)

RestictedImageNetResNet50 = builds(
    load_from_checkpoint,
    model=builds(models.resnet50, num_classes=9),
    builds_bases=(LoadFromCheckpoint,),
)

RestictedImageNetModel = builds(
    nn.Sequential, ImageNetNormalizer, RestictedImageNetResNet50
)


#####
# PGD
#####
def get_stepsize(factor, steps, epsilon):
    return factor * epsilon / steps


L2PGOpt = pbuilds(
    L2ProjectedOptim,
    InnerOpt=SGD,
    lr=builds(get_stepsize, factor=2.5, steps="${steps}", epsilon="${epsilon}"),
    epsilon="${epsilon}",
)

PGDModel = pbuilds(
    AdditivePerturbation, init_fn=pbuilds(uniform_like_l2_n_ball_, epsilon="${epsilon}")
)

L2PGD = pbuilds(
    gradient_ascent,
    # Type of Perturbation
    perturbation_model=PGDModel,
    # Optimizer (e.g., L2 PGD)
    optimizer=L2PGOpt,
    # solver parameters
    steps="${steps}",
    use_best=True,
    targeted=False,
    criterion=builds(torch.nn.CrossEntropyLoss, reduction="none"),
)


############
# PL Trainer
############
Trainer = builds(
    pl.Trainer,
    max_epochs=1,
    num_nodes=1,
    accelerator="gpu",
    devices="${gpus}",
    strategy=builds(HydraDDP),
    callbacks=[builds(MetricsCallback)],
    populate_full_signature=True,
)

Evaluator = builds(
    EvaluateModule,
    dataset="${dataset}",
    model="${model}",
    perturbation="${perturbation}",
    criterion=builds(nn.CrossEntropyLoss),
    batch_size="${batch_size}",
    num_workers="${num_workers}",
    hydra_convert="all",
)


######################################
# Experiment Configs and Task Function
# - Replaces config.yaml
######################################
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

DatasetCfg = make_config(
    data_path=str(Path.home() / ".torch" / "data"),
    batch_size=256,
    num_workers=8,
    dataset=MISSING,
)

ModelCfg = make_config(
    ckpt=None,
    model=MISSING,
)

PerturbationCfg = make_config(steps=7, perturbation=MISSING)
TrainerCfg = make_config(gpus=NUM_GPUS, trainer=Trainer, module=Evaluator)

Config = make_config(
    defaults=[
        "_self_",
        {"perturbation": "l2pgd"},
        {"dataset": "cifar10"},
        {"model": "cifar10_resnet50"},
    ],
    seed=12219,
    epsilon=0.0,
    bases=(DatasetCfg, ModelCfg, PerturbationCfg, TrainerCfg),
)


###################
# Swappable Configs
###################
cs = ConfigStore.instance()
cs.store(group="dataset", name="cifar10", node=CIFAR10)
cs.store(group="dataset", name="restricted_imagenet", node=RestrictedImageNet)
cs.store(group="model", name="cifar10_resnet50", node=CIFARModel)
cs.store(group="model", name="resnet50", node=RestictedImageNetModel)
cs.store(group="perturbation", name="l2pgd", node=L2PGD)
