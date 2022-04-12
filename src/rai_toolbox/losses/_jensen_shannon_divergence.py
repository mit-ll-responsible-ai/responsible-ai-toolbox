# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Optional

import torch as tr
import torch.nn.functional as F


def jensen_shannon_divergence(
    *probs: tr.Tensor, weight: Optional[float] = None
) -> tr.Tensor:
    """Computes the Jensen-Shannon divergence between n distributions:
                      JSD(P1, P2, ..., Pn)

    This loss is symmetric and is bounded by 0 <= JSD(P1, P2, ..., Pn) <= ln(n)

    Parameters
    ----------
    probs : tr.Tensor, shape-(N, D)
        A collection of n probability distributions. Each conveys of batch of N
        distributions over D categories.

    weight : Optional[float]
        A scaling factor that will be applied to the consistency loss.

    Returns
    -------
    loss : tr.Tensor, shape-(,)
        The scalar loss computed via the batch-mean.

    Notes
    -----
    The JSD loss is computed for each corresponding n-tuple of distributions and
    the batch-mean is ultimately returned.

    References
    ----------
    https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence"""

    if len(probs) < 2 or any(
        not isinstance(p, tr.Tensor) or p.dim() != 2 for p in probs
    ):
        raise ValueError(
            f"*probs must consist of at least two Tensors, and each tensor must have a shape of (N, D). Got {probs}"
        )

    zero = tr.tensor(0.0).type_as(probs[0])
    # Clamp mixture distribution to avoid exploding KL divergence
    log_p_mixture = tr.clamp(sum(probs, start=zero) / len(probs), 1e-7, 1).log()
    loss = sum(
        (F.kl_div(log_p_mixture, p, reduction="batchmean") for p in probs), start=zero
    ) / len(probs)

    if weight is not None:
        loss = loss * weight

    return loss
