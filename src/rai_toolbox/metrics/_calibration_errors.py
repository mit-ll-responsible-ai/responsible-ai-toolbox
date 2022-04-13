# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Any, List, NamedTuple, Optional

import torch
import torch as tr
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from typing_extensions import Literal


class CalibrationMetrics(NamedTuple):
    error: tr.Tensor
    binned_accuracies: tr.Tensor
    binned_confidences: tr.Tensor


ValidNorms = Literal["l1", "l2", "max"]


def calibration_error_contig_bin(
    *,
    confidences: tr.Tensor,
    matches: tr.Tensor,
    bin_size: int,
    norm: ValidNorms,
):
    """Test set examples are partitioned into contiguous bins that each contain
    `bin_size` number of examples, ordered by prediction confidence.

    Computes (Sum[prop-in-bin * (acc - mean-conf) ** 2]) ^ 1/2

    Parameters
    ----------
    confidences : Tensor, shape-(N,)
        The highest confidence score for each of N data.

    matches : Tensor, shape-(N,)
        A boolean-valued tensor indicating whether a given top prediction
        matched the ground-truth label.

    bin_size: int
        The bin-size used in the calculation.

    Returns
    -------
    CalibrationMetrics
        - error: tr.Tensor, shape-()
        - binned_accuracies: tr.Tensor, shape-(n_bin,)
        - binned_confidences: tr.Tensor, shape-(n_bin,)
    """
    assert bin_size > 0

    indices_sort_conf = tr.argsort(confidences)
    confidences = confidences[indices_sort_conf]
    matches = matches[indices_sort_conf]

    n_bins = len(matches) // bin_size
    conf_bin = tr.zeros(n_bins, dtype=confidences.dtype, device=confidences.device)
    acc_bin = tr.zeros_like(conf_bin)
    prop_bin = tr.zeros_like(conf_bin)

    confs = []
    accs = []
    for i in range(n_bins):
        matches_in_bin = matches[i * bin_size : (i + 1) * bin_size]
        acc_bin[i] = matches_in_bin.float().mean()
        conf_bin[i] = confidences[i * bin_size : (i + 1) * bin_size].float().mean()
        prop_bin[i] = len(matches_in_bin) / len(matches)

    if norm == "l1":
        ce = torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
    elif norm == "max":
        ce = torch.max(torch.abs(acc_bin - conf_bin))
    elif norm == "l2":
        ce = torch.sum(torch.pow(acc_bin - conf_bin, 2) * prop_bin)
        ce = torch.sqrt(ce) if ce > 0 else torch.tensor(0)
    else:
        raise ValueError(f"`norm` must be l1, l2, or max, got {norm}")

    return CalibrationMetrics(
        error=ce,
        binned_accuracies=tr.tensor(accs).type_as(confidences),
        binned_confidences=tr.tensor(confs).type_as(confidences),
    )


class CalibrationError(Metric):
    r"""
    Computes the top-label calibration error from
    https://arxiv.org/pdf/1909.10155.pdf using contiguous binning.


    Parameters
    ----------
    bin_size: int
        The size of each contiguous bin.

    norm: Literal["l1", "l2", "max"]
        Norm used to compare empirical and expected probability bins.
        Defaults to "l2".

    compute_on_step: bool, optional (default=True)
        Forward only calls ``update()`` and return None if this is set to False.

    dist_sync_on_step: bool, optional (default=False)
        Synchronize metric state across processes at each ``forward()``
        before returning the value at the step

    process_group: Optional[Any]
        Specify the process group on which synchronization is called.
        default: None (which selects the entire world)
    """
    DISTANCES = {"l1", "l2", "max"}
    confidences: List[Tensor]
    matches: List[Tensor]

    def __init__(
        self,
        bin_size: int,
        norm: ValidNorms = "l2",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        if norm not in self.DISTANCES:
            raise ValueError(
                f"Norm {norm} is not supported. Please select from l1, l2, or max. "
            )

        if not isinstance(bin_size, int) or bin_size <= 0:
            raise ValueError(
                f"Expected argument `bin_size` to be a int larger than 0 but got {bin_size}"
            )
        self.bin_size = bin_size
        self.norm: ValidNorms = norm

        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("matches", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Computes top-level confidences and matches, adding them to the
        internal state.

        Parameters
        ----------
        preds: Tensor, shape-(N, C)
            Model output probabilities among C classes for a size-N batch of data.

        target: Tensor, shape-(N, )
            Ground-truth target class labels for a size-N batch of data.
        """
        confidences, predictions = tr.max(preds, dim=1)
        matches = predictions.eq(target)

        self.confidences.append(confidences)
        self.matches.append(matches)

    def compute(self) -> Tensor:
        """
        Computes the calibration error.

        Returns
        -------
        expected_calibration_error: Tensor, shape-()
            Calibration error across previously collected examples.
        """
        confidences = dim_zero_cat(self.confidences)
        matches = dim_zero_cat(self.matches)
        return calibration_error_contig_bin(
            confidences=confidences,
            matches=matches,
            norm=self.norm,
            bin_size=self.bin_size,
        ).error
