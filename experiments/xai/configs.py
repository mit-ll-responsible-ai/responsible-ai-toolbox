# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from hydra_zen import MISSING, make_custom_builds_fn
from torch import nn
from torchvision import transforms

from rai_experiments.models import resnet
from rai_toolbox import datasets
from rai_toolbox.mushin import load_from_checkpoint
from rai_toolbox.optim import L1qFrankWolfe
from rai_toolbox.perturbations import gradient_ascent
from rai_toolbox.perturbations.models import AdditivePerturbation

###############
# Custom Builds
###############
builds = make_custom_builds_fn()
pbuilds = make_custom_builds_fn(zen_partial=True)

#########
# Dataset
#########
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


ImageNet = builds(
    datasets.ImageNet,
    root="${data_path}",
    transform=TestTransformImageNet,
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

ResNet50 = LoadFromCheckpoint(
    model=builds(resnet.resnet50, num_classes=1000), ckpt="${...ckpt}"
)
ImageNetModel = builds(
    nn.Sequential, ImageNetNormalizer, ResNet50, zen_meta=dict(ckpt="${ckpt}")
)

#####
# PGD
#####
L1q = pbuilds(L1qFrankWolfe, lr=1.0, epsilon=1.0, q=0.975, dq=0.05)
L1FW = pbuilds(
    gradient_ascent,
    perturbation_model=pbuilds(AdditivePerturbation),
    optimizer=L1q,
    steps=45,
    populate_full_signature=True,
)
