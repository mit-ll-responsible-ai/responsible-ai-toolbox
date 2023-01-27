# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from torch import Tensor


def get_axis(axarr, H, W, i, j):
    H, W = H - 1, W - 1
    if not (H or W):
        ax = axarr
    elif not (H and W):
        ax = axarr[max(i, j)]
    else:
        ax = axarr[i][j]
    return ax


def show_image_row(
    xlist: Tensor,
    ylist: Optional[List[str]] = None,
    fontsize: int = 12,
    size: Tuple[float, float] = (2.5, 2.5),
    tlist: Optional[List[str]] = None,
    filename: Optional[str] = None,
):
    H, W = len(xlist), len(xlist[0])
    fig, axarr = plt.subplots(
        H, W, figsize=(size[0] * W, size[1] * H), subplot_kw=dict(xticks=[], yticks=[])
    )
    for w in range(W):
        for h in range(H):
            ax = get_axis(axarr, H, W, h, w)
            im = xlist[h][w]
            if len(im) == 1:
                im = im.repeat(3, 1, 1)

            ax.imshow(im.permute(1, 2, 0))
            if ylist and w == 0:
                ax.set_ylabel(ylist[h], fontsize=fontsize)
            if tlist:
                ax.set_title(tlist[h][w], fontsize=fontsize)
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    return fig
