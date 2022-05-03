# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive
from typing_extensions import Literal

from ._cifar10_base import _CIFAR10Base
from ._utils import md5_check

__all__ = ["CIFAR10C", "CIFAR100C"]

PathLike = Union[str, Path]
Corruptions = Literal[
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
]

Severity = Literal[1, 2, 3, 4, 5]


class CIFAR10C(_CIFAR10Base):
    """
    A CIFAR-10-based test dataset for benchmarking neural network robustness to common
    corruptions [1]_. For each corruption and for each severity (`1-5`), the
    dataset consists of the 10,000 test images from CIFAR-10 with that corruption
    applied to those images.

    References
    ----------
    .. [1] https://zenodo.org/record/2535967#.YnAe_9rMKbg
    """

    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
    filename = "CIFAR-10-C.tar"
    tar_md5: str = "56bf5dcef84df0e2308c6dcbcbbd8499"
    base_folder: str = "CIFAR-10-C"

    _file_to_md5: Dict[str, str] = {
        "brightness": "0a81ef75e0b523c3383219c330a85d48",
        "contrast": "3c8262171c51307f916c30a3308235a8",
        "defocus_blur": "7d1322666342a0702b1957e92f6254bc",
        "elastic_transform": "9421657c6cd452429cf6ce96cc412b5f",
        "fog": "7b397314b5670f825465fbcd1f6e9ccd",
        "frost": "31f6ab3bce1d9934abfb0cc13656f141",
        "gaussian_blur": "c33370155bc9b055fb4a89113d3c559d",
        "gaussian_noise": "ecaf8b9a2399ffeda7680934c33405fd",
        "glass_blur": "7361fb4019269e02dbf6925f083e8629",
        "impulse_noise": "2090e01c83519ec51427e65116af6b1a",
        "jpeg_compression": "2b9cc4c864e0193bb64db8d7728f8187",
        "labels": "c439b113295ed5254878798ffe28fd54",
        "motion_blur": "fffa5f852ff7ad299cfe8a7643f090f4",
        "pixelate": "0f14f7e2db14288304e1de10df16832f",
        "saturate": "1cfae0964219c5102abbb883e538cc56",
        "shot_noise": "3a7239bb118894f013d9bf1984be7f11",
        "snow": "bb238de8555123da9c282dea23bd6e55",
        "spatter": "8a5a3903a7f8f65b59501a6093b4311e",
        "speckle_noise": "ef00b87611792b00df09c0b0237a1e30",
        "zoom_blur": "6ea8e63f1c5cdee1517533840641641b",
    }

    all_corruptions = (
        "brightness",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "fog",
        "frost",
        "gaussian_blur",
        "gaussian_noise",
        "glass_blur",
        "impulse_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        "saturate",
        "shot_noise",
        "snow",
        "spatter",
        "speckle_noise",
        "zoom_blur",
    )
    noise_corruptions = (
        "gaussian_noise",
        "impulse_noise",
        "shot_noise",
    )
    blur_corruptions = (
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
    )
    weather_corruptions = (
        "brightness",
        "fog",
        "frost",
        "snow",
    )
    digital_corruptions = (
        "contrast",
        "elastic_transform",
        "jpeg_compression",
        "pixelate",
    )

    # not included in augmix results
    extra_corruptions = (
        "gaussian_blur",
        "saturate",
        "spatter",
        "speckle_noise",
    )

    def __init__(
        self,
        root: PathLike,
        corruption: Corruptions,
        severity: Severity,
        transform: Optional[Callable[[Image.Image], Any]] = None,
        target_transform: Optional[Callable[[int], Any]] = None,
        download: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        root : PathLike
            Root directory of dataset where directory
            ``CIFAR-10-C`` exists or will be saved to if download is set to True.

        corruption : str
            The type of corruption, e.g., "fog". See `CIFAR10C.all_corruptions()` for a full list of corruptions.

        severity : Literal[1, 2, 3, 4, 5]
            The severity-level of the corruption, ranging from 1 to 5.

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

        self._root = (Path(self.root) / self.base_folder).resolve()
        assert corruption in self._file_to_md5
        self.corruption = corruption

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        if not isinstance(severity, (int, Integral)) and 1 <= severity <= 5:
            raise ValueError(f"`severity` must be an integer in [1, 5]. Got {severity}")

        severity -= 1  # type: ignore

        mmap = np.load(self._root / (corruption + ".npy"), mmap_mode="r")
        size = len(mmap) // 5

        # shape-(N, H, W, C)
        self.data = np.array(mmap[severity * size : (severity + 1) * size])
        self.targets = np.load(self._root / "labels.npy")

    def _check_integrity(self) -> bool:
        img_path = self._root / (self.corruption + ".npy")
        labels_path = self._root / "labels.npy"

        if not img_path.is_file() and not labels_path.is_file():
            return False

        img_md5 = md5_check(img_path)
        label_md5 = md5_check(labels_path)

        return (
            img_md5 == self._file_to_md5[self.corruption]
            and label_md5 == self._file_to_md5["labels"]
        )

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.tar_md5
        )

    def extra_repr(self) -> str:
        return "Split: Test"


class CIFAR100C(CIFAR10C):
    """
    A CIFAR-100-based test dataset for benchmarking neural network robustness to common
    corruptions [1]_. For each corruption and for each severity (`1-5`), the
    dataset consists of the 10,000 test images from CIFAR-100 with that corruption
    applied to those images.

    Parameters
    ----------
    root : PathLike
        Root directory of dataset where directory
        ``CIFAR-100-C`` exists or will be saved to if download is set to True.

    corruption : str
        The type of corruption, e.g., "fog". See `CIFAR100C.all_corruptions()` for a full
        list of corruptions.

    severity : Literal[1, 2, 3, 4, 5]
        The severity-level of the corruption, ranging from 1 to 5.

    transform :  Optional[Callable[[Image], Any]]
        A function/transform that takes in an PIL image
        and returns a transformed version. E.g., ``transforms.RandomCrop``

    target_transform : Optional[Callable]
        A function/transform that takes in a target
        and returns a transformed version.

    download: bool, optional (default=False)
        If true, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again.

    References
    ----------
    .. [1] https://zenodo.org/record/2535967#.YnAe_9rMKbg
    """

    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
    filename = "CIFAR-100-C.tar"
    tar_md5: str = "11f0ed0f1191edbf9fa23466ae6021d3"
    base_folder: str = "CIFAR-100-C"

    _file_to_md5: Dict[str, str] = {
        "brightness": "f22d7195aecd6abb541e27fca230c171",
        "contrast": "322bb385f1d05154ee197ca16535f71e",
        "defocus_blur": "d923e3d9c585a27f0956e2f2ad832564",
        "elastic_transform": "a0792bd6581f6810878be71acedfc65a",
        "fog": "4efc7ebd5e82b028bdbe13048e3ea564",
        "frost": "3a39c6823bdfaa0bf8b12fe7004b8117",
        "gaussian_blur": "5204ba0d557839772ef5a4196a052c3e",
        "gaussian_noise": "ecc4d366eac432bdf25c024086f5e97d",
        "glass_blur": "0bf384f38e5ccbf8dd479d9059b913e1",
        "impulse_noise": "3b3c210ddfa0b5cb918ff4537a429fef",
        "jpeg_compression": "c851b7f1324e1d2ffddeb76920576d11",
        "labels": "bb4026e9ce52996b95f439544568cdb2",
        "motion_blur": "732a7e2e54152ff97c742d4c388c5516",
        "pixelate": "96c00c60f144539e14cffb02ddbd0640",
        "saturate": "c0697e9fdd646916a61e9c312c77bf6b",
        "shot_noise": "b0a1fa6e1e465a747c1b204b1914048a",
        "snow": "0237be164583af146b7b144e73b43465",
        "spatter": "12ccf41d62564d36e1f6a6ada5022728",
        "speckle_noise": "e3f215b1a0f9fd9fd6f0d1cf94a7ce99",
        "zoom_blur": "0204613400c034a81c4830d5df81cb82",
    }

    classes = (
        "apple",
        "aquarium_fish",
        "baby",
        "bear",
        "beaver",
        "bed",
        "bee",
        "beetle",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "cattle",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
        "cockroach",
        "couch",
        "crab",
        "crocodile",
        "cup",
        "dinosaur",
        "dolphin",
        "elephant",
        "flatfish",
        "forest",
        "fox",
        "girl",
        "hamster",
        "house",
        "kangaroo",
        "keyboard",
        "lamp",
        "lawn_mower",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "man",
        "maple_tree",
        "motorcycle",
        "mountain",
        "mouse",
        "mushroom",
        "oak_tree",
        "orange",
        "orchid",
        "otter",
        "palm_tree",
        "pear",
        "pickup_truck",
        "pine_tree",
        "plain",
        "plate",
        "poppy",
        "porcupine",
        "possum",
        "rabbit",
        "raccoon",
        "ray",
        "road",
        "rocket",
        "rose",
        "sea",
        "seal",
        "shark",
        "shrew",
        "skunk",
        "skyscraper",
        "snail",
        "snake",
        "spider",
        "squirrel",
        "streetcar",
        "sunflower",
        "sweet_pepper",
        "table",
        "tank",
        "telephone",
        "television",
        "tiger",
        "tractor",
        "train",
        "trout",
        "tulip",
        "turtle",
        "wardrobe",
        "whale",
        "willow_tree",
        "wolf",
        "woman",
        "worm",
    )
