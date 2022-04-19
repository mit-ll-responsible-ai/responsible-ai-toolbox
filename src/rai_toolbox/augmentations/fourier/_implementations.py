# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import random
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Collection,
    DefaultDict,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import torch as tr
from torch.nn.functional import softmax
from torchmetrics import Metric

from rai_toolbox import evaluating
from rai_toolbox._utils import get_device

from ._fourier_basis import generate_fourier_bases

T = TypeVar("T", np.ndarray, tr.Tensor)


def perturb_batch(
    *,
    batch: T,
    basis: np.ndarray,
    basis_norm: float,
    rand_flip_per_channel: bool,
) -> T:
    """
    Given a single Fourier basis array, perturbs a batch of images by applying
        color_channel += rand_sign * norm * normed_basis

    to each color channel of each image.

    Parameters
    ----------
    batch: np.ndarray, shape-(N, C, H, W)
        N: batch size, C: number of color channels.

    basis: np.ndarray, shape-(H, W)
        Assumed to already be normalized

    basis_norm: float

    rand_flip_per_channel: bool
        If True, the std-lib `random` module is used to draw a random sign associated
        with each of the to-be-perturbed color channels.

    Returns
    -------
    perturbed_batch: np.ndarray, shape-(N, C, H, W)
    """
    basis = basis * basis_norm

    if not rand_flip_per_channel:
        if isinstance(batch, tr.Tensor):
            basis = tr.tensor(basis, dtype=batch.dtype, device=batch.device)
        return batch + basis

    _channel_flips = np.array(
        [random.randrange(-1, 2, 2) for i in range(batch.shape[0] * batch.shape[1])],
        dtype="float32",
    )

    out = np.multiply.outer(_channel_flips, basis).reshape(batch.shape)

    if isinstance(batch, tr.Tensor):
        out = tr.tensor(out, dtype=batch.dtype, device=batch.device)

    out += batch
    return out


def normalize(
    imgs: np.ndarray,
    *,
    mean: Union[float, Sequence[float]],
    std: Union[float, Sequence[float]],
    inplace: bool = False,
):
    """
    Returns (imgs - mean) / std

    Parameters
    ----------
    imgs: np.ndarray, shape-(N, C, H, W)
    mean : float | shape-(C,)
    std : float | shape-(C,)
    inplace: bool, optional (default=False)

    Returns
    -------
    normalized_imgs: np.ndarray, shape-(N, C, H, W)
    """
    mean_arr = np.atleast_1d(mean)[None, :, None, None].astype(imgs.dtype)
    std_arr = np.atleast_1d(std)[None, :, None, None].astype(imgs.dtype)

    assert imgs.ndim == 4
    if not inplace:
        return (imgs - mean_arr) / std_arr
    else:
        imgs -= mean_arr
        imgs /= std_arr
        return imgs


class HeatMapEntry(NamedTuple):
    pos: Tuple[int, int]
    sym_pos: Tuple[int, int]
    result: Any


def create_heatmaps(
    dataloader: Collection[Tuple[tr.Tensor, tr.Tensor]],
    image_height_width: Tuple[int, int],
    *,
    model: tr.nn.Module,
    metrics: Dict[str, Type[Metric]],
    basis_norm: float,
    rand_flip_per_channel: bool,
    post_pert_batch_transform: Optional[Callable[[tr.Tensor], tr.Tensor]] = None,
    device: Optional[Union[tr.device, str, int]] = None,
    row_col_coords: Optional[Iterable[Tuple[int, int]]] = None,
    factor_2pi_phase_shift: float = 0,
) -> Dict[str, List[HeatMapEntry]]:
    from rai_toolbox._utils.tqdm import tqdm
    
    _outer_total = (
        None if row_col_coords is not None else int(np.prod(image_height_width)) // 2
    )

    with evaluating(model), tr.no_grad():
        if device is not None:
            device = tr.device(device)
            model.to(device=device)
        else:
            device = get_device(model)

        if post_pert_batch_transform is None:

            def post_pert_batch_transform(x):
                return x

        results: DefaultDict[str, Dict[Tuple[int, int], HeatMapEntry]] = defaultdict(
            dict
        )

        for batch, targets in tqdm(dataloader, desc="batch"):
            batch = batch.pin_memory()
            targets = targets.to(device=device)

            for basis in tqdm(
                generate_fourier_bases(
                    *image_height_width,
                    dtype="float32",
                    row_col_coords=row_col_coords,
                    factor_2pi_phase_shift=factor_2pi_phase_shift,
                ),
                total=_outer_total,
                desc="fourier-grid",
                leave=False,
            ):
                p_batch = batch.to(device=device)

                p_batch = perturb_batch(
                    batch=p_batch,
                    basis=basis.basis,
                    basis_norm=basis_norm,
                    rand_flip_per_channel=rand_flip_per_channel,
                )

                p_batch = post_pert_batch_transform(p_batch)

                probs = softmax(model(p_batch), dim=1).cpu()
                targets = targets.cpu()

                for name, M in metrics.items():
                    if basis.position not in results[name]:
                        results[name][basis.position] = HeatMapEntry(
                            basis.position,
                            sym_pos=basis.sym_position,
                            result=M(),
                        )
                    results[name][basis.position].result.update(
                        preds=probs, target=targets
                    )

        out: Dict[str, List[HeatMapEntry]] = {}
        for metric_name, r in results.items():
            out[metric_name] = [
                HeatMapEntry(p, ps, metric.compute())
                for _, (p, ps, metric) in r.items()
            ]
        return out
