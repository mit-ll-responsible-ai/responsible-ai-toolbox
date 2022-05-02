# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Optional

import torch as tr
import torch.nn.functional as F

from rai_toolbox._typing import ArrayLike


def jensen_shannon_divergence(
    *probs: ArrayLike, weight: Optional[float] = None
) -> tr.Tensor:
    """Computes the Jensen-Shannon divergence [1]_ between n distributions:
                      JSD(P1, P2, ..., Pn)

    This loss is symmetric and is bounded by 0 <= JSD(P1, P2, ..., Pn) <= ln(n)

    Parameters
    ----------
    probs : ArrayLike, shape-(N, D)
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
    [1] .. https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

    Examples
    --------
    Let's measure the divergence between three length-two discrete distributiones

    >>> from rai_toolbox.losses import jensen_shannon_divergence
    >>> P1 = [[0.0, 1.0]]
    >>> P2 = [[1.0, 0.0]]
    >>> P3 = [[0.5, 0.5]]
    >>> jensen_shannon_divergence(P1, P2, P3)
    tensor(0.4621)

    The divergence is symmetric.

    >>> jensen_shannon_divergence(P1, P3, P2)
    tensor(0.4621)
    """

    probs = tuple(tr.as_tensor(p) for p in probs)

    zero = tr.tensor(0.0).type_as(probs[0])
    # Clamp mixture distribution to avoid exploding KL divergence
    log_p_mixture = tr.clamp(sum(probs, zero) / len(probs), 1e-7, 1).log()
    loss = sum(
        (F.kl_div(log_p_mixture, p, reduction="batchmean") for p in probs), zero
    ) / len(probs)

    if weight is not None:
        loss = loss * weight

    return loss
