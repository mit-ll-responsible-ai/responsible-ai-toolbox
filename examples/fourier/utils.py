# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch as tr
from hydra_zen import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from torch.nn import Module

root = Path("/home/ry26099/uq_by_design/fourier_pert_exps/cifar10_results/")

__all__ = ["load_model_from_yaml", "renorm", "imshow", "plot_heatmaps"]


def load_model_from_yaml(
    model_name: str,
    root: Path = root,
) -> Module:
    """Loads the lightning module without an optimizer"""
    assert (root / model_name).is_dir()
    root = root / model_name
    path_to_yaml = root / ".hydra/config.yaml"

    path_to_yaml = Path(path_to_yaml)
    path_to_ckpts = root / "checkpoints"

    path_to_ckpt, *_ = list(
        p for p in path_to_ckpts.glob("*.ckpt") if "last" not in p.name
    )

    ExpConfig = OmegaConf.load(path_to_yaml.resolve())
    ExpConfig.lightning_module.lr_scheduler = None
    ExpConfig.lightning_module.lr_sched_config = None
    lit_module: LightningModule = instantiate(ExpConfig.lightning_module)

    module = type(lit_module).load_from_checkpoint(
        str((path_to_ckpt).resolve()),
        model=lit_module.model,
        optim=None,
    )
    module.eval()
    return module


def renorm(x: Union[np.ndarray, tr.Tensor]) -> Union[np.ndarray, tr.Tensor]:
    x = x - x.min()
    x /= x.max()
    return x


def imshow(
    ax: plt.Axes, img: Union[np.ndarray, tr.Tensor], title: Optional[str] = None
):
    ax.imshow(renorm(np.transpose(img, (1, 2, 0))))
    ax.set_axis_off()
    if title:
        ax.set_title(title)


def plot_heatmaps(results: Dict[str, Any]):
    heatmaps = {}
    for metric_name, metrics in results.items():
        heatmap = np.zeros((32, 32))

        for p, ps, datum in metrics:
            heatmap[p] = datum.item()
            heatmap[ps] = datum.item()
        heatmaps[metric_name] = heatmap

    fig, axes = plt.subplots(ncols=len(heatmaps), figsize=(10, 10))

    im = None

    for (name, heat), ax in zip(heatmaps.items(), axes):  # type: ignore
        if name == "accuracy":
            name = "classification_error"
            heat = 1 - heat
        im = ax.imshow(heat, vmin=0, vmax=1, cmap="plasma")
        ax.set_title(name.replace("_", " ").title())
        ax.set_axis_off()

    if im is not None:
        cbar_ax = fig.add_axes([0.95, 0.30, 0.05, 0.4])
        cbar_ax.set_title("Error-Rate")
        fig.colorbar(im, cax=cbar_ax)

    return fig, axes
