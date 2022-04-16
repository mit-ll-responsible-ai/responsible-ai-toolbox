# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from hydra_zen import load_from_yaml
from omegaconf import DictConfig, ListConfig
from torch import nn

log = logging.getLogger(__name__)


def load_from_checkpoint(
    model: nn.Module,
    *,
    ckpt: Optional[Union[str, Path]] = None,
    weights_key: Optional[str] = None,
    weights_key_strip: Optional[str] = None,
    model_attr: Optional[str] = None,
) -> nn.Module:
    """Load model weights.

    Parameters
    ----------
    model : Module
        The PyTorch Module

    ckpt : Optional[Union[str, Path]]
        The path to the file containing the model weights. If no path is provided
        the model will not be updated.

    weights_key : Optional[str] (default: "state_dict")
        (load_module=False) The key from the checkpoint file containing the model
        weights.

    weights_key_strip : Optional[str] (default: "model")
        (load_module=False) The prefix to remove from each weight's key prior
        to loading the module.

    model_attr : Optional[str] (default: "model")
        (load_module=False) The attribute of the module containing the `torch.nn.Module`

    Returns
    -------
    module : LightningModule
    """
    if ckpt is None:
        return model

    ckpt = Path(str(ckpt))
    if not ckpt.exists():
        ckpt = Path.home() / ".torch" / "models" / ckpt
    log.info(f"Loading model checkpoint from {ckpt}")

    ckpt_data: Dict[str, Any] = torch.load(ckpt, map_location="cpu")

    if weights_key is not None:
        assert weights_key in ckpt_data
        ckpt_data = ckpt_data[weights_key]

    if weights_key_strip:
        if not weights_key_strip.endswith("."):
            weights_key_strip = weights_key_strip + "."

        ckpt_data = {
            k[len(weights_key_strip) :]: v
            for k, v in ckpt_data.items()
            if k.startswith(weights_key_strip)
        }

    if model_attr is None:
        # The weights can be loaded in directly
        model.load_state_dict(ckpt_data)  # type: ignore

    else:
        assert hasattr(model, model_attr)
        getattr(model, model_attr).load_state_dict(ckpt_data)

    return model


@dataclass
class Experiment:
    working_dir: str
    cfg: Optional[Union[Dict, ListConfig, DictConfig]]
    ckpts: List[str]
    metrics: Dict


def load_experiment(
    exp_path: Union[str, Path], search_path: Optional[Union[str, Path]] = None
) -> Union[Experiment, List[Experiment]]:
    """Loads all configuration and metrics outputs in an experiment directory.

    Parameters
    ----------
    exp_path: Union[str, Path]
        The directory to search for data. Directory must include the
        ".hydra/config.yaml" file.

    Returns
    ----------
    exps: Union[Experiment, List[Experiment]]

    """
    assert Path(exp_path).exists(), f"{exp_path} not found"

    # first find all .hydra files
    if search_path is None:
        search_path = ".hydra"
    cfg_files = sorted(Path(exp_path).absolute().glob(f"**/{str(search_path)}"))

    # For each file load metrics data
    exps = []
    for path in cfg_files:
        # Save experiment configuration
        cfg_files = list(path.parent.glob("**/config.yaml"))
        cfg = None
        if len(cfg_files) == 1:
            cfg = load_from_yaml(cfg_files[0])

        # Load metrics files
        files = path.parent.glob("*.pt")
        metrics = dict()
        for f in files:
            name = f.name
            metrics[name[:-3]] = torch.load(f)

        # Load path to checkpoints
        ckpts = [str(ckpt.resolve()) for ckpt in path.parent.glob("**/*.ckpt")]

        # Append experiment to list
        exps.append(Experiment(str(path.parent.parent), cfg, ckpts, metrics))

    if len(exps) == 1:
        return exps[0]

    return exps
