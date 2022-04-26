# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import functools
from typing import Any, Callable, List, Optional, Tuple, Union

import torch as tr
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module

from rai_toolbox import negate
from rai_toolbox._typing import (
    ArrayLike,
    InstantiatesTo,
    Optimizer,
    OptimizerType,
    Partial,
    instantiates_to,
)
from rai_toolbox._utils.stateful import evaluating, frozen
from rai_toolbox.perturbations import AdditivePerturbation, PerturbationModel


def gradient_ascent(
    *,
    model: Callable[[Tensor], Tensor],
    data: ArrayLike,
    target: ArrayLike,
    optimizer: Union[OptimizerType, Partial[Optimizer]],
    steps: int,
    perturbation_model: Union[
        PerturbationModel, InstantiatesTo[PerturbationModel]
    ] = AdditivePerturbation,
    targeted: bool = False,
    use_best: bool = False,
    criterion: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    reduction_fn: Callable[[Tensor], Tensor] = tr.sum,
    **optim_kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    """Solve for a set of perturbations for a given set of data and a model.

    This performs, for `steps` iterations, the following optimization::

       optim = optimizer(perturbation_model.parameters)
       pert_data = perturbation_model(data)
       loss = criterion(model(pert_data), target)
       loss = (1 if targeted else -1) * loss  # default: targeted=False
       optim.step()

    Note that, by default, this perturbs the data away from `target` (i.e. this performs
    gradient *ascent*), given a standard loss function that seeks to minimize the
    diffence between the model's output and the target. See `targeted` to toggle this
    behavior.

    Parameters
    ----------
    model : Callable[[Tensor], Tensor]
        Differentiable function that processes the (perturbed) data prior to computing
        the loss.

        If `model` is a `torch.nn.Module` then its weights will be frozen and it will
        be set to eval mode during the perturbation-solve phase.

    data : Tensor, shape-(N, ...)
        The input data to perturb.

    target : Tensor, shape-(N, ...)
        If `targeted==False` (default), then this is the target to perturb away from.
        If `targeted==True`, then this is the target to perturb toward.

    optimizer : Type[Optimizer] | Partial[Optimizer]
        The optimizer to use for updating the perturbation model

    steps : int
        Number of projected gradient steps

    perturbation_model : PerturbationModel | Type[PerturbationModel], optional (default=AdditivePerturbation)
        A `torch.nn.Module` whose parameters are updated by the solver. Its forward-pass applies the perturbation to the data. Default is
        `AdditivePerturbation`, which simply adds the perturbation to the data.

        The perturbation model should not modify the data in-place.

        If `perturbation_model` is a type, then it will be instantiated as
        `perturbation_model(data)`.

    targeted : bool (default: False)
        If `True`, then perturb towards the defined `target` otherwise move away from
        `target`.

    use_best : bool (default: True)
        Whether to only report the best perturbation over all steps.
        Note: Requires criterion to output a loss per sample, e.g., set
        `reduction="none"`

    criterion : Optional[Callable[[Tensor, Tensor], Tensor]]
        The criterion to use for calculating the loss.  If `None` then
        `CrossEntropyLoss` is used.

    reduction_fn : Callable[[Tensor], Tensor], optional (default=tr.sum)
        Used to reduce the shape-(N,) per-datum loss to a scalar.

    **optim_kwargs : Any
       Keyword arguments passed to `optimizer` when it is instatiated.

    Returns
    -------
    xadv, losses : tuple[Tensor, Tensor], shape-(N, ...), shape-(N, ...)
        The perturbed data, if `use_best==True` then this is the best perturbation
        based on the loss across all steps.

        The loss for each perturbed data point, if `use_best==True` then this is the
        best loss across all steps.

    Examples
    --------
    Let's perturb two data points, x=-1.0 and x=2.0, to maximize `L(x) = |x|`. We will
    use five standard gradient steps, using a learning rate of 0.1. The default
    perturbation model is simply additive: `x -> x + δ`.

    This solver is refining `δ`, whose initial value is 0 by default, to maximize
    `L(x) = |x|`. Thus we should find that our solved perturbations modify our data as:
    `-1.0 -> -1.5` and `2.0 -> 2.5`, respectively.

    >>> from rai_toolbox.perturbations import gradient_ascent
    >>> from torch.optim import SGD

    >>> identity_model = lambda data: data
    >>> abs_diff = lambda model_out, target: (model_out - target).abs()

    >>> perturbed_data, losses = gradient_ascent(
    ...    data=[-1.0, 2.0],
    ...    target=0.0,
    ...    model=identity_model,
    ...    criterion=abs_diff,
    ...    optimizer=SGD,
    ...    lr=0.1,
    ...    steps=5,
    ... )
    >>> perturbed_data
    tensor([-1.5000,  2.5000])

    We can instead specify `targeted=True` and perform gradient *descent*.
    Here, the perturbations we solve for should modify our data as:
    -1.0 -> -0.5 and 2.0 -> 1.5, respectively.

    >>> perturbed_data, losses = gradient_ascent(
    ...    data=[-1.0, 2.0],
    ...    target=0.0,
    ...    model=identity_model,
    ...    criterion=abs_diff,
    ...    optimizer=SGD,
    ...    lr=0.1,
    ...    steps=5,
    ...    targeted=True,
    ... )
    >>> perturbed_data
    tensor([-0.5000,  1.5000])
    """
    data = tr.as_tensor(data)
    target = tr.as_tensor(target)

    if not data.is_leaf:
        data = data.detach()

    if not target.is_leaf:
        target = target.detach()

    # Initialize
    best_loss = None
    best_x = None

    if criterion is None:
        criterion = CrossEntropyLoss(reduction="none")

    if not targeted:
        # maximize the objective function
        criterion = negate(criterion)

    if instantiates_to(perturbation_model, PerturbationModel):
        pmodel = perturbation_model(data)
    else:
        if not isinstance(perturbation_model, PerturbationModel):
            raise TypeError(
                f"`perturbation_model` must be satisfy the `PerturbationModel`"
                f" protocol, got: {perturbation_model}"
            )

        pmodel = perturbation_model

    optim = optimizer(pmodel.parameters(), **optim_kwargs)

    to_freeze: List[Any] = [data, target]

    if isinstance(model, Module):
        to_freeze.append(model)

    # don't pass non nn.Module to frozen/eval
    packed_model = (model,) if isinstance(model, Module) else ()

    # Projected Gradient Descent
    with frozen(*to_freeze), evaluating(*packed_model), tr.enable_grad():
        for _ in range(steps):
            # Calculate the gradient of loss
            xadv = pmodel(data)
            logits = model(xadv)
            losses = criterion(logits, target)
            loss = reduction_fn(losses)

            # Update the perturbation
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            if use_best:
                if (
                    (losses.ndim == 0 and data.ndim > 0)
                    or losses.ndim > data.ndim
                    or losses.shape != data.shape[: losses.ndim]
                ):
                    raise ValueError(
                        f"`use_best=True` but `criterion` does not ouput a per-datum-loss. "
                        f"I.e. `criterion` returned a tensor of shape-{tuple(losses.shape)} for a "
                        f"batch of shape-{tuple(data.shape)}. Expected a tensor of "
                        f"shape-{(len(data),)} or greater."
                    )

                best_loss, best_x = _replace_best(losses, best_loss, xadv, best_x)

        # free up memory
        optim.zero_grad(set_to_none=True)

        # Final evalulation
        with tr.no_grad():
            xadv = pmodel(data)
            logits = model(xadv)
            losses = criterion(logits, target)

            if use_best:
                losses, xadv = _replace_best(losses, best_loss, xadv, best_x)

    if not targeted:
        losses = -1 * losses
    return xadv.detach(), losses.detach()


def random_restart(
    perturber: Callable[..., Tuple[Tensor, Tensor]],
    repeats: int,
) -> Callable[..., Tuple[Tensor, Tensor]]:
    """Executes a perturbation function multiple times saving out the best perturbation.

    Parameters
    ----------
    perturber : Callable[..., Tuple[Tensor, Tensor]]
        The perturbation function, e.g., projected_gradient_perturbation.

    repeats : int
        The number of times to run perturber

    Returns
    -------
    random_restart_fn : Callable[..., Tuple[Tensor, Tensor]]
        Wrapped function that will execute `perturber` `repeats` times.

    """
    if repeats < 1:
        raise ValueError(f"expected times >= 1, got {repeats}")

    @functools.wraps(perturber)
    def random_restart_fn(*args, **kwargs) -> Tuple[Tensor, Tensor]:
        targeted = kwargs.get("targeted", False)
        use_best = kwargs.get("use_best", False)
        best_x = None
        best_loss = None

        for _ in range(repeats):
            # run the attack
            xadv, losses = perturber(*args, **kwargs)

            # Save best loss for each data point
            if use_best:
                best_loss, best_x = _replace_best(
                    loss=losses,
                    best_loss=best_loss,
                    data=xadv,
                    best_data=best_x,
                    min=targeted,
                )
            else:
                best_loss = losses
                best_x = xadv

        assert isinstance(best_x, Tensor)
        assert isinstance(best_loss, Tensor)
        return best_x, best_loss

    return random_restart_fn


# A function that updates the best loss and best input
def _replace_best(
    loss: Tensor,
    best_loss: Optional[Tensor],
    data: Tensor,
    best_data: Optional[Tensor],
    min: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Returns the data with the smallest (or largest) loss

    Parameters
    ----------
    loss : Tensor, shape-(N, ...)
        N: batch size

    best_loss : Optional[Tensor], shape-(N, ...)
        N: batch size

    data : Tensor, shape-(N, ...)
        N: batch size

    best_data : Optional[Tensor], shape-(N, ...)
        N: batch size

    min : bool (default: True)
        Whether best is minimum (True) or maximum (False)

    Returns
    -------
    best_loss, best_data : Tuple[Tensor, Tensor]
    """
    if best_loss is None:
        best_data = data
        best_loss = loss
    else:
        assert best_data is not None
        if min:
            replace = loss < best_loss
        else:
            replace = loss > best_loss

        best_data[replace] = data[replace]
        best_loss[replace] = loss[replace]

    return best_loss, best_data
