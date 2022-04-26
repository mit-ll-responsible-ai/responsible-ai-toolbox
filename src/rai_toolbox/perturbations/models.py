# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from abc import abstractmethod
from typing import Callable, Iterator, Optional, Sequence, Union

import torch as tr
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class PerturbationModel(Protocol):
    """Protocol for Perturbation Models."""

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover
        ...

    @abstractmethod
    def __call__(self, data: Tensor) -> Tensor:  # pragma: no cover
        """A perturbation model should take data as the input and output the
        peturbed data.

        Parameters
        ----------
        data: Tensor
            The data to perturb.

        Returns
        -------
        Tensor
            The perturbed data with the same shape as `data`.
        """

    def parameters(
        self, recurse: bool = True
    ) -> Iterator[Parameter]:  # pragma: no cover
        ...


class AdditivePerturbation(nn.Module, PerturbationModel):
    r"""Modifies a piece or batch of data by adding a perturbation: :math:`x \rightarrow x+\delta`.

    :math:`\delta` is a trainable parameter of `AdditivePerturbation`.
    """

    def __init__(
        self,
        data_or_shape: Union[Tensor, Sequence[int]],
        init_fn: Optional[Callable[[Tensor], None]] = None,
        *,
        device: Optional[tr.device] = None,
        dtype: Optional[tr.dtype] = None,
        delta_ndim: Optional[int] = None,
    ) -> None:
        """The init function should support data is the argument to initialize
        perturbations.

        Parameters
        ----------
        data_or_shape: Union[Tensor, Tuple[int, ...]]
            Determines the shape of the perturbation. If a tensor is supplied, its dtype
            and device are mirrored by the initialized perturbation.

            This parameter can be modified to control whether the perturbation adds
            elementwise or broadcast-adds over `x`.

        init_fn: Optional[Callable[[Tensor], None]]
            Operates in-place on a zero'd perturbation tensor to determine the
            final initialization of the perturbation.

        device: Optional[tr.device]
            If specified, takes precedent over the device associated with `data_or_shape`.

        dtype: Optional[tr.dtype] = None
            If specified, takes precedent over the dtype associated with `data_or_shape`.

        delta_ndim: Optional[int] = None
            If a positive number, determines the dimensionality of the perturbation.
            If a negative number, indicates the 'offset' from the dimensionality of
            `data_or_shape`. E.g. if `data_or_shape` has a shape (N, C, H, W), and
            if `delta_ndim=-1`, then the perturbation will have shape (C, H, W),
            and will be applied in a broadcast fashion.
        """
        super().__init__()

        init_kwargs = {}

        if isinstance(data_or_shape, tr.Tensor):
            shape = data_or_shape.shape
            init_kwargs.update(
                {
                    "dtype": data_or_shape.dtype,
                    "device": data_or_shape.device,
                    "layout": data_or_shape.layout,
                }
            )
        else:
            shape = tuple(data_or_shape)

        if device is not None:
            init_kwargs["device"] = device

        if dtype is not None:
            init_kwargs["dtype"] = dtype

        if delta_ndim is not None:
            offset = len(shape) - delta_ndim if delta_ndim >= 0 else abs(delta_ndim)
            shape = shape[offset:]

        self.delta = nn.Parameter(tr.zeros(shape, **init_kwargs))  # type: ignore

        if init_fn is not None:
            init_fn(self.delta)

    def forward(self, data: Tensor) -> Tensor:
        """Add perturbation to data.

        Parameters
        ----------
        data: Tensor
            The data to perturb.

        Returns
        -------
        Tensor
            The perturbed data
        """
        return data + self.delta
