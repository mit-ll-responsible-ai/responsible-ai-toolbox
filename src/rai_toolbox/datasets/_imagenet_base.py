# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from torchvision.datasets import folder


class ImageNet(folder.DatasetFolder):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        """
        Parameters
        ----------
        root : Union[str, Path]
            Root directory of dataset where
            ImageNet folders exist.

        transform :  Optional[Callable[[Image], Any]]
            A function/transform that takes in a PIL image
            and returns a transformed version. E.g., ``transforms.RandomCrop``

        target_transform : Optional[Callable]
            A function/transform that takes in a target
            and returns a transformed version.

        loader : Callable[[str], Any], optional (default=`folder.default_loader`)
            A function to load a sample given its path.

        is_valid_file : Optional[Callable[[str], bool]], optional (default=None)
            A function that takes path of a file
            and checks if the file is a valid file.
        """
        super().__init__(
            root,
            loader,
            folder.IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )


def remap_classes(
    range_sets: Tuple[Set[int], ...], class_to_idx: Dict[str, int]
) -> Tuple[List[str], Dict[str, int]]:
    """Filters existing classes and maps them to new index based on a pre-defined mapping.

    Parameters
    ----------
    range_sets: Tuple[Set[int], ...]
        Each element of the tuple is a list of indices that map the original class to the new class.

    class_to_idx: Dict[str, int]
        Dictionary mapping class name to class index

    Returns
    -------
    Tuple[List[str], Dict[str, int]]
        List of classes and dictionary mapping each class to an index.
    """
    mapping = {}
    for class_name, idx in class_to_idx.items():
        for new_idx, range_set in enumerate(range_sets):
            if idx in range_set:
                mapping[class_name] = new_idx
                break

    filtered_classes = list(mapping.keys())
    filtered_classes.sort()
    return filtered_classes, mapping


class _ReducedImageNet(folder.DatasetFolder):
    _range_sets: Tuple[Set[int], ...]

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root=root,
            loader=loader,
            extensions=folder.IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes, class_to_idx = super().find_classes(directory)
        classes, class_to_idx = remap_classes(self._range_sets, class_to_idx)
        return classes, class_to_idx


class RestrictedImageNet(_ReducedImageNet):
    """Restricted ImageNet, see https://arxiv.org/abs/1906.00945


    **WARNING**: This dataset is imbalanced


    Dog:
        ImageNet Class Indices: 151-268
    Cat:
        ImageNet Class Indices: 281-285
    Frog:
        ImageNet Class Indices: 30-32
    Turtle:
        ImageNet Class Indices: 33-37
    Bird:
        ImageNet Class Indices: 80-100
    Monkey:
        ImageNet Class Indices: 365-382
    Fish:
        ImageNet Class Indices: 389-397
    Crab:
        ImageNet Class Indices: 118-121
    Insect:
        ImageNet Class Indices: 300-319

    """

    _range_sets = tuple(
        set(range(s, e + 1))
        for s, e in [
            (151, 268),
            (281, 285),
            (30, 32),
            (33, 37),
            (80, 100),
            (365, 382),
            (389, 397),
            (118, 121),
            (300, 319),
        ]
    )


class ImageNetM10(_ReducedImageNet):
    """ImageNet-M10, see https://arxiv.org/pdf/2112.15329.pdf

    ImageNet-M10 consists of ten super-classes, each corresponding a WordNet ID in the hierarchy.
    Different super-classes contain varying number of ImageNet classes, so the dataset is balanced by
    choosing six classes within each super-class.

    Dog:
        WordNet ID: n02084071
        ImageNet Class Indices: 151 to 156
    Bird:
        WordNet ID: n01503061
        ImageNet Class Indices: 7 to 12
    Insect:
        WordNet ID: n02159955
        ImageNet Class Indices: 300 to 305
    Monkey:
        WordNet ID: n02484322
        ImageNet Class Indices: 370 to 375
    Car:
        WordNet ID: n02958343
        ImageNet Class Indices: 407, 436, 468, 511, 609, 627
    Feline:
        WordNet ID: n02120997
        ImageNet Class Indices: 286 to 291
    Truck:
        WordNet ID: n04490091
        ImageNet Class Indices: 555, 569, 675, 717, 734, 864
    Fruit:
        WordNet ID: n13134947
        ImageNet Class Indices: 948, 984, 987 to 990
    Fungus:
        WordNet ID: n12992868
        ImageNet Class Indices: 991 to 996
    Boat:
        WordNet ID: n02858304
        ImageNet Class Indices: 472, 554, 576, 625, 814, 914

    """

    _range_sets = (
        set(range(151, 156 + 1)),
        set(range(7, 12 + 1)),
        set(range(300, 305 + 1)),
        set(range(370, 375 + 1)),
        {407, 436, 468, 511, 609, 627},
        set(range(286, 291 + 1)),
        {555, 569, 675, 717, 734, 864},
        {948, 984, 987, 988, 999, 990},
        set(range(991, 996 + 1)),
        {472, 554, 576, 625, 814, 914},
    )
