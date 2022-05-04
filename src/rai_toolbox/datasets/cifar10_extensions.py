# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
from PIL import Image
from torchvision.datasets.utils import download_url

from ._cifar10_base import _CIFAR10Base
from ._utils import md5_check

PathLike = Union[str, Path]

__all__ = ["CIFAR10P1"]


class CIFAR10P1(_CIFAR10Base):
    """CIFAR-10.1 dataset is a new test set for CIFAR-10.

    "CIFAR-10.1 contains 2,000 new test images that were sampled after multiple years of
    research on the original CIFAR-10 dataset. The data collection for CIFAR-10.1 was
    designed to minimize distribution shift relative to the original dataset."

    See https://github.com/modestyachts/CIFAR-10.1 for more details.
    """

    _data_md5 = "4fcae82cb1326aec9ed1dc1fc62345b8"
    _labels_md5 = "09a97fb7c430502fcbd69d95093a3f85"
    url = "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/"
    _data_filename = "cifar10.1_v6_data.npy"
    _labels_filename = "cifar10.1_v6_labels.npy"

    def __init__(
        self,
        root: PathLike,
        transform: Optional[Callable[[Image.Image], Any]] = None,
        target_transform: Optional[Callable[[int], Any]] = None,
        download: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        root : PathLike
            Root directory of dataset where
            CIFAR-10.1 files exist or will be saved to if download is set to True.

        transform :  Optional[Callable[[Image], Any]]
            A function/transform that takes in a PIL image
            and returns a transformed version. E.g., ``transforms.RandomCrop``

        target_transform : Optional[Callable]
            A function/transform that takes in a target
            and returns a transformed version.

        download : bool, optional (default=False)
            If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again."""

        super().__init__(
            root=str(root), transform=transform, target_transform=target_transform
        )

        self._root = Path(self.root).resolve()
        self._img_path = self._root / self._data_filename
        self._labels_path = self._root / self._labels_filename

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        # shape-(N, H, W, C)
        self.data = np.load(self._img_path)
        self.targets = np.load(self._labels_path)

    def _check_integrity(self) -> bool:
        img_path = self._root / self._data_filename
        labels_path = self._root / self._labels_filename

        if not img_path.is_file() and not labels_path.is_file():
            return False

        img_md5 = md5_check(img_path)
        label_md5 = md5_check(labels_path)

        return img_md5 == self._data_md5 and label_md5 == self._labels_md5

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_url(
            self.url + self._data_filename,
            self.root,
            filename=self._data_filename,
            md5=self._data_md5,
        )
        download_url(
            self.url + self._labels_filename,
            self.root,
            filename=self._labels_filename,
            md5=self._labels_md5,
        )

    def extra_repr(self) -> str:
        return "Split: Test"
