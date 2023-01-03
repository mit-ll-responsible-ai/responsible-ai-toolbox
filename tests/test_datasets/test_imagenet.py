# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Type

import numpy as np
import pytest
from PIL import Image
from torchvision.datasets import folder

from rai_toolbox.datasets import ImageNet, ImageNetM10, RestrictedImageNet
from rai_toolbox.datasets._imagenet_base import remap_classes


def create_random_jpeg(fn):
    im = np.random.rand(30, 30, 3) * 255
    im_out = Image.fromarray(im.astype("uint8")).convert("RGB")
    im_out.save(fn)


class SimpleDataFolder(folder.DatasetFolder):
    def __init__(self, root):
        super().__init__(
            root,
            folder.default_loader,
            folder.IMG_EXTENSIONS,
        )


@pytest.mark.parametrize("Dataset", [SimpleDataFolder, ImageNet])
@pytest.mark.parametrize("num_classes", range(1, 4))
@pytest.mark.usefixtures("cleandir")
def test_datafolder(Dataset: Type[ImageNet], num_classes: int):
    for cls_num in range(1, num_classes + 1):
        Path(f"class{cls_num}").mkdir()
        create_random_jpeg(Path(f"class{cls_num}") / "img.jpeg")

    ds = Dataset(".")
    assert len(ds) == num_classes
    assert len(ds.class_to_idx.values()) == num_classes


@pytest.mark.parametrize(
    "Dataset, num_new_classes, num_idxs",
    [(ImageNetM10, 60, 10), (RestrictedImageNet, 203, 9)],
)
def test_remap_classess(Dataset, num_new_classes, num_idxs):
    class_to_idx = {}
    for i in range(1000):
        class_to_idx[f"{i}"] = i

    new_classes, new_class_to_idx = remap_classes(Dataset._range_sets, class_to_idx)
    assert len(set(new_class_to_idx.values())) == num_idxs
    assert len(set(new_class_to_idx.keys())) == len(new_classes)
    assert len(new_classes) == num_new_classes


# def test_imagenet():
#     root = Path.home() / ".torch" / "data" / "ImageNet" / "train"
#     if not root.exists():
#         pytest.skip("Cannot find imagenet director")
#     ds = ImageNet(root=root)
#     assert len(set(ds.class_to_idx.values())) == 1000


# def test_restricted_imagenet():
#     root = Path.home() / ".torch" / "data" / "ImageNet" / "train"
#     if not root.exists():
#         pytest.skip("Cannot find imagenet director")
#     ds = RestrictedImageNet(root=root)
#     assert len(set(ds.class_to_idx.values())) == 9


# def test_imagenetm10():
#     root = Path.home() / ".torch" / "data" / "ImageNet" / "train"
#     if not root.exists():
#         pytest.skip("Cannot find imagenet director")
#     ds = ImageNetM10(root=root)
#     assert len(set(ds.class_to_idx.values())) == 10
