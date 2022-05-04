# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from abc import abstractmethod
from typing import Any, Callable, Iterator, Optional, Sequence, Union

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



    Attributes
    ----------
    delta : torch.Tensor
        :math:`\delta` is the sole trainable parameter of `AdditivePerturbation`.
    """

    def __init__(
        self,
        data_or_shape: Union[Tensor, Sequence[int]],
        init_fn: Optional[Callable[[Tensor], None]] = None,
        *,
        device: Optional[tr.device] = None,
        dtype: Optional[tr.dtype] = None,
        delta_ndim: Optional[int] = None,
        **init_fn_kwargs: Any,
    ) -> None:
        """The init function should support data as the argument to initialize
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
            `data_or_shape`. E.g., if `data_or_shape` has a shape (N, C, H, W), and
            if `delta_ndim=-1`, then the perturbation will have shape (C, H, W),
            and will be applied in a broadcast fashion.

        **init_fn_kwargs: Any
            Keyword arguments passed to `init_fn`.

        Examples
        --------
        **Basic Additive Perturbations**

        Let's imagine we have a batch of three shape-`(2, 2)` images (our toy data will
        be all ones) that we want to perturb. We'll randomly initialize a shape-`(3, 2,
        2)` tensor of perturbations to apply additively to the shape-`(3, 2, 2)` batch.

        >>> import torch as tr
        >>> from rai_toolbox.perturbations import AdditivePerturbation, uniform_like_l1_n_ball_
        >>> data = tr.ones(3, 2, 2)

        We provide a `generator` argument to control the RNG in `~rai_toolbox.
        perturbations.uniform_like_l1_n_ball_`.

        >>> pert_model = AdditivePerturbation(
        ...     data_or_shape=data,
        ...     init_fn=uniform_like_l1_n_ball_,
        ...     generator=tr.Generator().manual_seed(0),  # controls RNG of init
        ... )

        Accessing the initialized perturbations.

        >>> pert_model.delta
        Parameter containing:
        tensor([[[0.0885, 0.0436],
                [0.3642, 0.2720]],
                .
                [[0.3074, 0.1827],
                [0.1440, 0.2624]],
                .
                [[0.3489, 0.0528],
                [0.0539, 0.1767]]], requires_grad=True)

        Applying the perturbations to a batch of data.

        >>> pert_data = pert_model(data)
        >>> pert_data
        tensor([[[1.0885, 1.0436],
                [1.3642, 1.2720]],
                .
                [[1.3074, 1.1827],
                [1.1440, 1.2624]],
                .
                [[1.3489, 1.0528],
                [1.0539, 1.1767]]], grad_fn=<AddBackward0>)

        Involving the perturbed data in a computational graph where auto-diff is
        performed, then gradients are computed for the perturbations.

        >>> (pert_data ** 2).sum().backward()
        >>> pert_model.delta.grad
        tensor([[[2.1770, 2.0871],
                [2.7285, 2.5439]],
                .
                [[2.6148, 2.3653],
                [2.2880, 2.5247]],
                .
                [[2.6978, 2.1056],
                [2.1078, 2.3534]]])

        **Broadcasted ("Universal") Perturbations**

        Suppose that we want to use a single shape-`(2, 2)` tensor to perturb each datum
        in a batch. We can create a perturbation model in a similar manner, but
        specifying `delta_ndim=-1` indicates that our perturbation should have one
        fewer dimension than our data; whereas our batch has shape-`(N, 2, 2)`, our
        perturbation model's parameter will have shape-`(2, 2)`

        >>> pert_model = AdditivePerturbation(
        ...     data_or_shape=data,
        ...     delta_ndim=-1,
        ...     init_fn=uniform_like_l1_n_ball_,
        ...     generator=tr.Generator().manual_seed(1),  # controls RNG of init
        ... )
        >>> pert_model.delta
        Parameter containing:
        tensor([[0.2793, 0.4783],
                [0.4031, 0.3316]], requires_grad=True)

        Perturbing a batch of data now performs broadcast-addition of this tensor over
        the batch.

        >>> pert_data = pert_model(data)
        >>> pert_data
        tensor([[[1.2793, 1.4783],
                [1.4031, 1.3316]],
                .
                [[1.2793, 1.4783],
                [1.4031, 1.3316]],
                .
                [[1.2793, 1.4783],
                [1.4031, 1.3316]]], grad_fn=<AddBackward0>)

        .. important::

           Downstream reductions of this broadcast-pertubed data should involve
           a mean – not a sum – over the batch dimension so that the resulting gradient
           computed for the perturbation is not scaled by batch-size.

           >>> (pert_data ** 2).mean().backward()
           >>> pert_model.delta.grad
           tensor([[0.6397, 0.7392],
                   [0.7015, 0.6658]])

           Similarly, when using a `~rai_toolbox.optim.ParamTransformingOptimizer` to
           optimize this broadcasted perturbation, we should specify `param_ndim=None`
           to ensure that the parameter transformations are not broadcasted over our
           perturbation tensor and/or its gradient, as it has no batch dimension.
        """
        super().__init__()

        _init_kwargs = {}

        if isinstance(data_or_shape, tr.Tensor):
            shape = data_or_shape.shape
            _init_kwargs.update(
                {
                    "dtype": data_or_shape.dtype,
                    "device": data_or_shape.device,
                    "layout": data_or_shape.layout,
                }
            )
        else:
            shape = tuple(data_or_shape)

        if device is not None:
            _init_kwargs["device"] = device

        if dtype is not None:
            _init_kwargs["dtype"] = dtype

        if delta_ndim is not None:
            offset = len(shape) - delta_ndim if delta_ndim >= 0 else abs(delta_ndim)
            shape = shape[offset:]

        self.delta = nn.Parameter(tr.zeros(shape, **_init_kwargs))  # type: ignore
        del _init_kwargs

        if init_fn is not None:
            init_fn(self.delta, **init_fn_kwargs)
        elif init_fn_kwargs:
            raise TypeError(
                f"No `init_fn` was specified, but the keyword arguments "
                f"{init_fn_kwargs} were provided."
            )

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
