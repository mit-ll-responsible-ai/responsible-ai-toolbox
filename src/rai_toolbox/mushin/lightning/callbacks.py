# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from collections import defaultdict
from pathlib import Path
from typing import Union

import torch
from pytorch_lightning import Callback

from rai_toolbox._utils import value_check


class MetricsCallback(Callback):
    """Saves validation and test metrics stored in `trainer.callback_metrics`.

    Parameters
    ----------
    save_dir : str, optional (default=".")

    filename : str, optional (default="metrics.pt")
        The base filename used to store metrics.  For `FITTING` the file is prepended
        with "fit_" and and for `TESTING` the file is prepended with `test_`.

    Notes
    -----
    No metrics will be saved during `FITTING` if no validation metrics are calculated.
    This is a limitation of PyTorch Lightning. Future versions will save the training
    step metrics when no validation metrics are calculated.

    Examples
    --------

    >>> from pytorch_lightning import Trainer
    >>> from rai_toolbox.mushin import MetricsCallback

    >>> metrics_callback = MetricsCallback()
    >>> trainer = Trainer(callbacks=[metrics_callback])
    """

    def __init__(
        self,
        save_dir: Union[Path, str] = ".",
        filename: Union[Path, str] = "metrics.pt",
    ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.filename = value_check("filename", filename, type_=(str, Path))
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.test_metrics = defaultdict(list)

    def _get_filename(self, stage: str):
        return self.save_dir / f"{stage}_{self.filename}"

    def _process_metrics(self, stored_metrics, metrics):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                if v.ndim == 0:
                    v = v.item()
                else:
                    v = v.cpu().numpy()
            stored_metrics[k].append(v)

    def on_validation_end(self, trainer, pl_module):
        # Make sure PL is not doing it's sanity check run
        if trainer.sanity_checking:
            return self.val_metrics

        metrics = trainer.callback_metrics
        self.val_metrics["epoch"].append(pl_module.current_epoch)
        self._process_metrics(self.val_metrics, metrics)
        torch.save(self.val_metrics, self._get_filename("fit"))
        return self.val_metrics

    def on_test_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        self._process_metrics(self.test_metrics, metrics)
        torch.save(self.test_metrics, self._get_filename("test"))
        return self.test_metrics
