# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import hashlib
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from typing_extensions import Literal

from ._utils import md5_check

__all__ = ["MNISTC"]

# typing hints
PathLike = Union[str, Path]
Corruptions = Literal[
    "brightness",
    "canny_edges",
    "dotted_line",
    "fog",
    "glass_blur",
    "identity",
    "impulse_noise",
    "motion_blur",
    "rotate",
    "scale",
    "shear",
    "shot_noise",
    "spatter",
    "stripe",
    "translate",
    "zigzag",
]
Groupings = Literal["train", "test", "combined"]


class MNISTC(VisionDataset):

    url = "https://zenodo.org/record/3239543/files/mnist_c.zip"
    filename = "mnist_c.zip"
    zip_md5: str = "4b34b33045869ee6d424616cd3a65da3"
    files_md5: str = "d2bff22ed798fda041f7a54c096a1a2b"
    base_folder: str = "MNISTC"

    all_corruptions = (
        "brightness",
        "canny_edges",
        "dotted_line",
        "fog",
        "glass_blur",
        "identity",
        "impulse_noise",
        "motion_blur",
        "rotate",
        "scale",
        "shear",
        "shot_noise",
        "spatter",
        "stripe",
        "translate",
        "zigzag",
    )

    all_groupings = ("train", "test", "combined")

    def __init__(
        self,
        root: PathLike,
        corruption: Corruptions,
        grouping: Groupings,
        transform: Optional[Callable[[Image.Image], Any]] = None,
        target_transform: Optional[Callable[[int], Any]] = None,
        download: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        root : PathLike
            Root directory of dataset where directory
            ``MNISTC`` exists or will be saved to if download is set to True.

        corruption : str
            The type of corruption, e.g., "fog". See `MNISTC.all_corruptions()` for a full list of corruptions.

        grouping : str
            The type of grouping for the dataset: "train", "test", or "combined".

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

        # TODO: check to see how using _root below in downloader affects torchvision root in base class
        self._root = (Path(self.root) / self.base_folder).resolve()
        assert (
            corruption in self.all_corruptions
        ), f"The corruption '{corruption}' is invalid"
        self.corruption = corruption

        assert grouping in self.all_groupings, (
            f"The grouping '{grouping}' is invalid: valid choices"
            " are 'train', 'test', or 'combined'"
        )
        self.grouping = grouping

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        if self.grouping == "test" or self.grouping == "combined":
            test_images_path = Path.joinpath(
                self._root, "mnist_c", corruption, "test_images.npy"
            )
            test_labels_path = Path.joinpath(
                self._root, "mnist_c", corruption, "test_labels.npy"
            )
            mmap_test = np.load(test_images_path, mmap_mode="r")
            test_target = np.load(test_labels_path)

        if self.grouping == "train" or self.grouping == "combined":
            train_images_path = Path.joinpath(
                self._root, "mnist_c", corruption, "train_images.npy"
            )
            train_labels_path = Path.joinpath(
                self._root, "mnist_c", corruption, "train_labels.npy"
            )
            mmap_train = np.load(train_images_path, mmap_mode="r")
            train_target = np.load(train_labels_path)

        # shape-(N, H, W, C)
        if self.grouping == "train":
            self.data = mmap_train
            self.targets = train_target
        elif self.grouping == "test":
            self.data = mmap_test
            self.targets = test_target
        elif self.grouping == "combined":
            self.data = np.concatenate((mmap_test, mmap_train))
            self.targets = np.concatenate((test_target, train_target))
        else:
            raise RuntimeError(f"Unknown grouping: {self.grouping}")

    def _check_integrity(self) -> bool:
        root_path = Path(self._root)
        zip_path = Path.joinpath(root_path, self.filename)

        if Path.is_file(zip_path):
            zip_md5_check = md5_check(zip_path) == self.zip_md5
            if not zip_md5_check:
                return False

        files_to_checksum = []

        for file in Path.glob(root_path, "mnist_c/**/*"):
            if not file.is_file():
                continue
            files_to_checksum.append(file)

        # sort the file paths to make MD5 deterministic
        files_to_checksum.sort()
        hash_md5 = hashlib.md5()

        for file in files_to_checksum:
            # not using package .util hasher since this is
            # hashing over several files and needs to keep state
            # between files
            chunksize: int = 1024**2
            with open(file, "rb") as f:
                for chunk in iter(lambda: f.read(chunksize), b""):
                    hash_md5.update(chunk)

        if hash_md5.hexdigest() != self.files_md5:
            return False

        return True

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Parameters
        ----------
        index: int

        Returns
        -------
        Tuple
            (transform(image), target_transform(target)) where
            target is the index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image -- need to drop last dimension to
        # properly import using PIL
        img = Image.fromarray(img[:, :, 0])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url, self._root, filename=self.filename, md5=self.zip_md5
        )
