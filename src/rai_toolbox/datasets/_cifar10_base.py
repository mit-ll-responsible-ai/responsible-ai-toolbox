# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset

__all__ = ["_CIFAR10Base"]


class _CIFAR10Base(VisionDataset):
    classes = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    data: np.ndarray  # shape-(N, H, W, C)
    targets: np.ndarray  # shape-(N,)
    _class_to_idx = None

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
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    @property
    def class_to_idx(self) -> Dict[str, int]:
        if self._class_to_idx is None:
            self._class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        return self._class_to_idx

    def __len__(self) -> int:
        return len(self.data)
