# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import functools
from typing import Any, Callable, List, Optional, Tuple, Union

import torch as tr
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer as _TorchOptim
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

from rai_toolbox import negate
from rai_toolbox._typing import (
    ArrayLike,
    InstantiatesTo,
    Optimizer,
    OptimizerType,
    Partial,
    instantiates_to,
)
from rai_toolbox._utils import value_check
from rai_toolbox._utils.stateful import evaluating, frozen
from rai_toolbox.optim.misc import ClampedShrinkingThresholdOptim
from rai_toolbox.perturbations import AdditivePerturbation, PerturbationModel


def gradient_ascent(
    *,
    model: Callable[[Tensor], Tensor],
    data: ArrayLike,
    target: ArrayLike,
    optimizer: Union[Optimizer, OptimizerType, Partial[Optimizer]],
    steps: int,
    perturbation_model: Union[
        PerturbationModel, InstantiatesTo[PerturbationModel]
    ] = AdditivePerturbation,
    targeted: bool = False,
    use_best: bool = False,
    criterion: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    reduction_fn: Callable[[Tensor], Tensor] = tr.sum,
    lr_scheduler: Optional[
        Union[Callable[[Optimizer], _LRScheduler], _LRScheduler]
    ] = None,
    **optim_kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    """Solve for a set of perturbations for a given set of data and a model,
    and then apply those perturbations to the data.

    This performs, for `steps` iterations, the following optimization::

       optim = optimizer(perturbation_model.parameters)
       pert_data = perturbation_model(data)
       loss = criterion(model(pert_data), target)
       loss = (1 if targeted else -1) * loss  # default: targeted=False
       optim.step()

    Note that, by default, this perturbs the data away from `target` (i.e., this performs
    gradient *ascent*), given a standard loss function that seeks to minimize the
    diffence between the model's output and the target. See `targeted` to toggle this
    behavior.

    Parameters
    ----------
    model : Callable[[Tensor], Tensor]
        Differentiable function that processes the (perturbed) data prior to computing
        the loss.

        If `model` is a `torch.nn.Module`, then its weights will be frozen and it will
        be set to eval mode during the perturbation-solve phase.

    data : ArrayLike, shape-(N, ...)
        The input data to perturb.

    target : ArrayLike, shape-(N, ...)
        If `targeted==False` (default), then this is the target to perturb away from.
        If `targeted==True`, then this is the target to perturb toward.

    optimizer : Optimizer | Type[Optimizer] | Partial[Optimizer]
        The optimizer to use for updating the perturbation model.

        If `optimizer` is uninstantiated, it will be instantiated as
        `optimizer(perturbation_model.parameters(), **optim_kwargs)`

    steps : int
        Number of projected gradient steps.

    perturbation_model : PerturbationModel | Type[PerturbationModel], optional (default=AdditivePerturbation)
        A `torch.nn.Module` whose parameters are updated by the solver. Its forward-pass
        applies the perturbation to the data. Default is
        `AdditivePerturbation`, which simply adds the perturbation to the data.

        The perturbation model should not modify the data in-place.

        If `perturbation_model` is a type, then it will be instantiated as
        `perturbation_model(data)`.

    criterion : Optional[Callable[[Tensor, Tensor], Tensor]]
        The criterion to use for calculating the loss **per-datum**.  I.e., for a
        shape-(N, ...) batch of data, `criterion` should return a shape-(N,) tensor of
        loss values – one for each datum in the batch.

        If `None`, then `CrossEntropyLoss(reduction=None)` is used.

    targeted : bool (default: False)
        If `True`, then perturb towards the defined `target`, otherwise move away from
        `target`.

        Note: Default (`targeted=False`) implements gradient *ascent*.
        To perform gradient *descent*, set `targeted=True`.

    use_best : bool (default: True)
        Whether to only report the best perturbation over all steps.
        Note: Requires criterion to output a loss per sample, e.g., set
        `reduction="none"`.

    reduction_fn : Callable[[Tensor], Tensor], optional (default=torch.sum)
        Used to reduce the shape-(N,) per-datum loss to a scalar. This should be
        set to `torch.mean` when solving for a "universal" perturbation.

    lr_scheduler: Optional[Type[LRScheduler] | LRScheduler]
        A learning rate scheduler, whose step is applied after each gradient-based update.

        If `lr_scheduler` is uninstantiated, it will be instantiated as
        `lr_scheduler(optimizer)`

    **optim_kwargs : Any
       Keyword arguments passed to `optimizer` when it is instatiated.

    Returns
    -------
    xadv, losses : tuple[Tensor, Tensor], shape-(N, ...), shape-(N, ...)
        The perturbed data, if `use_best==True` then this is the best perturbation
        based on the loss across all steps.

        The loss for each perturbed data point, if `use_best==True` then this is the
        best loss across all steps.

    Notes
    -----
    `model` is automatically set to eval-mode and its parameters are set to
    `requires_grad=False` within the context of this function.

    Examples
    --------
    Let's perturb two data points, `x1=-1.0` and `x2=2.0`, to maximize
    `L(δ; x) = |x + δ|` w.r.t `δ`. We will use five standard gradient steps, using a
    learning  rate of 0.1. The default perturbation model is simply additive:
    `x -> x + δ`.

    This solver is refining `δ1` and `δ2`, whose initial values are 0 by default, to
    maximize `L(x) = |x|` for `x1` and `x2`, respectively. Thus we should find that our
    solved perturbations modify our data as: `x-1.0 -> -1.5` and `2.0 -> 2.5`,
    respectively.

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

    **Accessing the perturbations**

    To gain direct access to the solved perturbations, we can provide our own
    perturbation model to the solver. Let's solve the same optimization problem, but
    provide our own instance of `AdditivePerturbation`

    >>> from rai_toolbox.perturbations import AdditivePerturbation
    >>> pert_model = AdditivePerturbation(data_or_shape=(2,))
    >>> perturbed_data, losses = gradient_ascent(
    ...    perturbation_model=pert_model,
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

    Now we can access the values that were solved for `δ1` and `δ2`.

    >>> pert_model.delta
    Parameter containing:
    tensor([-0.5000,  0.5000], requires_grad=True)
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
        # TODO: open issue with pyright for false positive
        pmodel = perturbation_model(data)
    else:
        if not isinstance(perturbation_model, PerturbationModel):
            raise TypeError(
                f"`perturbation_model` must be satisfy the `PerturbationModel`"
                f" protocol, got: {perturbation_model}"
            )

        pmodel = perturbation_model

    if instantiates_to(optimizer, _TorchOptim):
        optim = optimizer(pmodel.parameters(), **optim_kwargs)
    else:
        if not isinstance(optimizer, _TorchOptim):
            raise TypeError(
                f"`optimizer` must be an instance of Optimizer or must instantiate to "
                f"Optimizer; got: {optimizer}"
            )
        if instantiates_to(perturbation_model, PerturbationModel):
            raise TypeError(
                "An initialized optimizer can only be passed to the solver in "
                "combination with an intialized perturbation model."
            )
        if optim_kwargs:
            raise TypeError(
                "**optim_kwargs were specified, but the provided `optimizer` has "
                "already been instaniated"
            )

        optim = optimizer

    if lr_scheduler is not None and callable(lr_scheduler):
        lr_scheduler = lr_scheduler(optim)

    if lr_scheduler is not None and not isinstance(lr_scheduler, _LRScheduler):
        value_check("lr_scheduler", lr_scheduler, type_=_LRScheduler, optional=True)

    to_freeze: List[Any] = [data, target]

    if isinstance(model, Module):
        to_freeze.append(model)

    # don't pass non nn.Module to frozen/eval
    packed_model = (model,) if isinstance(model, Module) else ()

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
            if lr_scheduler is not None:
                lr_scheduler.step()

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
                # we negate the loss when `targeted=True` so min-loss is always best
                losses, xadv = _replace_best(losses, best_loss, xadv, best_x, min=True)

    if not targeted:
        # The returned loss is always relative to the original criterion.
        # E.g. an adversarial perturbation with a "low" negated loss will yield
        # a high loss.
        losses = -1 * losses
    return xadv.detach(), losses.detach()


def _attack_loss(
    scores: Tensor,
    labels: Tensor,
    margin: Union[float, tr.Tensor],
    targeted: bool = True,
):
    # scores: shape-(N, C)
    # labels: shape-(N,)  values in [0, C)
    margin = tr.as_tensor(margin)
    mask = F.one_hot(labels, num_classes=scores.shape[1])
    non_target_scores = tr.max(scores - (mask * 1e25), dim=1).values
    del mask

    target_scores = scores[(range(len(labels)), labels)]
    if targeted:
        return tr.maximum(non_target_scores - target_scores, -margin)
    else:
        return tr.maximum(target_scores - non_target_scores, -margin)


def elastic_net_attack(
    *,
    model: Callable[[Tensor], Tensor],
    data: ArrayLike,
    target: ArrayLike,
    steps: int,
    beta: float,
    c: float,
    confidence: float,
    lr: float,
    targeted: bool = True,
    pert_clamp_min: Optional[Union[float, Tensor]] = None,
    pert_clamp_max: Optional[Union[float, Tensor]] = None,
    lr_schedule: Optional[Partial[_LRScheduler]] = None,
    **optim_kwargs,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Parameters
    ----------
    model : Callable[[Tensor], Tensor]
        Differentiable function that processes the (perturbed) data prior to computing
        the loss.

        If `model` is a `torch.nn.Module`, then its weights will be frozen and it will
        be set to eval mode during the perturbation-solve phase.

    data : ArrayLike, shape-(N, ...)
        A batch of N pieces of input data to perturb.

    target : ArrayLike, shape-(N,)
        A batch of N of target/truth labels.

        If `targeted==True` (default), then this is the target to perturb toward.
        If `targeted==False`, then this is the target to perturb away from.

    beta : float
        The shrinkage threshold and L1 penalty weight

    c : float
        The weight of the adversarial attack loss: c * adv_attack_loss + elastic_net_loss

    confidence : float
        Controls the separation between the target and the next most likely
        prediction among all classes other than the target class

    lr : float
        The learning rate

    targeted : bool, optional (default=True)
        If `targeted==True` (default), then the target to perturbed toward.
        If `targeted==False`, then this is the target to perturbed away from.

    pert_clamp_min : Optional[Union[float, Tensor]]
        If specified, sets a lower-bound on the domain of the additive perturbation.
        Set to `a - data` to enforce `a <= data + δ`.

    pert_clamp_max : Optional[Union[float, Tensor]]
        If specified, sets an upper-bound on the domain of the additive perturbation.
        Set to `b - data` to enforce `data + δ <= b`.

    lr_schedule : Optional[Partial[_LRScheduler]] = None
        Specifies the learning rate schedule for the optimizer. Will be instantiated
        as `lr_schedule(optimizer)`. By default a square-root decay-to-0 lr schedule is
        used. To disable the lr schedule, specify `partial(LambdaLR, lr_lambda=lambda k: 1)`

    Returns
    -------
    successes, best_xadv, best_loss : Tuple[Tensor, Tensor, Tensor]
        - successes: shape-(N,) boolean-valued array indicating whether a solution
            was found for each datum. I.e. "success" for datum-j means that
            the model thinks that `x_j + δ_j` does (doesn't) resemble the
            specified target (label).
        - best_xadv: shape-(N, ...) tensor of the best perturbed examples.
            If `successes[j]` is `False` then `best_xadv[j]` will be a tensor of nans.
        - best_loss: shape-(N,) tensor of smallest elastic net losses.
            If `successes[j]` is `False` then `best_loss[j]` will be `inf`.

        Here "best" indicates that the elastic net loss, `b |δ| + |δ|^2_2`, is minimized

    Examples
    --------
    Here is a trivial scenario where we are merely perturbing the "logits" themselves so
    that the specified targets will be optimized for. Let's see that the longer
    we run the optimizer, the more the learned perturbation shrinks while still
    amounting to a successful attack.

    >>> from rai_toolbox.perturbations.solvers import elastic_net_attack
    >>> logits = [[0.497, 0.503]]
    >>> target = [0]

    >>> for num_steps in [1, 10, 100]:
    ...     _, x_adv, _ = elastic_net_attack(
    ...         model=lambda x: x,
    ...         data=logits,
    ...         target=target,
    ...         beta=1e-3,
    ...         c=2,
    ...         steps=num_steps,
    ...         confidence=.01,
    ...         lr=0.5,
    ...     )
    ...     print(f"num-steps: {num_steps}\n{x_adv}")
    num-steps: 1
    tensor([[ 1.4960, -0.4960]])
    num-steps: 10
    tensor([[0.5062, 0.4938]])
    num-steps: 100
    tensor([[0.5018, 0.4982]])
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

    pmodel = AdditivePerturbation(data)

    # clamps perturbations so that 0 <= x + δ <= 1
    optim = ClampedShrinkingThresholdOptim(
        pmodel.parameters(),
        shrink_size=beta,
        clamp_min=pert_clamp_min,
        clamp_max=pert_clamp_max,
        lr=lr,
        **optim_kwargs,
    )

    if lr_schedule is None:
        # sqrt lr-decay to 0
        lr_sched = LambdaLR(optim, lambda k: (1 - min(k, steps) / steps) ** (0.5))
    else:
        lr_sched = lr_schedule(optim)

    to_freeze: List[Any] = [data, target]

    if isinstance(model, Module):
        to_freeze.append(model)

    # don't pass non nn.Module to frozen/eval
    packed_model = (model,) if isinstance(model, Module) else ()

    delta_old = pmodel.delta

    def elastic_net_loss():
        l1 = tr.linalg.norm(pmodel.delta.view(len(pmodel.delta), -1), dim=1, ord=1)
        l2_sqr = tr.einsum("n...,n...->n", pmodel.delta, pmodel.delta)
        return beta * l1 + l2_sqr

    def get_best_loss_and_adv_example(tracking_success, best_loss, best_x):
        xadv = pmodel(data)
        preds = tr.argmax(model(xadv), dim=1)
        success = target == preds if targeted else target != preds
        if tracking_success is None:
            tracking_success = success

        tracking_success |= success
        best_loss, best_x = _replace_best(
            elastic_net_loss(),
            best_loss,
            xadv,
            best_x,
            success=success,
            min=True,
        )

        return tracking_success, best_loss, best_x

    tracking_success = None

    with frozen(*to_freeze), evaluating(*packed_model), tr.enable_grad():
        for k in range(steps):
            # Calculate the gradient of loss
            with tr.no_grad():
                delta_stored = pmodel.delta.clone().detach()  # δx_new
                if k > 0:
                    # evaluates model(x + δx)
                    tracking_success, best_loss, best_x = get_best_loss_and_adv_example(
                        tracking_success, best_loss, best_x
                    )

                    # computes δy = δx_new - (k/(k+3))(δx_new - δx_old)
                    pmodel.delta *= 1 + k / (k + 3)  # type: ignore
                    pmodel.delta -= (k / (k + 3)) * delta_old  # type: ignore
                delta_old = delta_stored  # δx_old <- δx_new

            # evaluates model(x + δy)
            logits = model(pmodel(data))

            losses = (
                c * _attack_loss(logits, target, margin=confidence, targeted=True)
                + elastic_net_loss()
            )

            loss = losses.sum()

            # Update the perturbation
            optim.zero_grad(set_to_none=True)
            loss.backward()

            # computes δx_new = Sb(δy - lr * grad(δy))  (stored in pmodel.delta)
            optim.step()
            lr_sched.step()

        # free up memory
        optim.zero_grad(set_to_none=True)

        # Final evalulation
        with tr.no_grad():
            tracking_success, best_loss, best_x = get_best_loss_and_adv_example(
                tracking_success, best_loss, best_x
            )

    return tracking_success.detach(), best_x.detach(), best_loss.detach()


def random_restart(
    solver: Callable[..., Tuple[Tensor, Tensor]],
    repeats: int,
) -> Callable[..., Tuple[Tensor, Tensor]]:
    """Executes a solver function multiple times, saving out the best perturbations.

    Parameters
    ----------
    solver : Callable[..., Tuple[Tensor, Tensor]]
        The solver whose execution will be repeated.

    repeats : int
        The number of times to run `solver`

    Returns
    -------
    random_restart_fn : Callable[..., Tuple[Tensor, Tensor]]
        Wrapped function that will execute `solver` `repeats` times.

    Examples
    --------
    Let's perturb two data points, `x1=-1.0` and `x2=2.0`, to maximize
    `L(δ; x) = |x + δ|` w.r.t `δ`. Our perturbation will randomly initialize `δ1` and
    `δ2` and we will re-run the solver three times – retaining the best perturbation of
    `x1` and `x2` respectively.

    >>> from functools import partial
    >>> import torch as tr
    >>> from torch.optim import SGD
    >>> from rai_toolbox.perturbations.init import uniform_like_l1_n_ball_
    >>> from rai_toolbox.perturbations import AdditivePerturbation, gradient_ascent, random_restart
    >>>
    >>> def verbose_abs_diff(model_out, target):
    ...     # used to print out loss at each solver step (for purpose of example)
    ...     out =  (model_out - target).abs()
    ...     print(out)
    ...     return out

    Configuring a peturbation model to randomly initialize the perturbations.

    >>> RNG = tr.Generator().manual_seed(0)
    >>> PertModel = partial(AdditivePerturbation, init_fn=uniform_like_l1_n_ball_, generator=RNG)

    Creating and running repeating solver.

    >>> gradient_ascent_with_restart = random_restart(gradient_ascent, 3)

    >>> perturbed_data, losses = gradient_ascent_with_restart(
    ...    perturbation_model=PertModel,
    ...    data=[-1.0, 2.0],
    ...    target=0.0,
    ...    model=lambda data: data,
    ...    criterion=verbose_abs_diff,
    ...    optimizer=SGD,
    ...    lr=0.1,
    ...    steps=1,
    ... )
    tensor([0.5037, 2.7682], grad_fn=<AbsBackward0>)
    tensor([0.6037, 2.8682])
    tensor([0.9115, 2.1320], grad_fn=<AbsBackward0>)
    tensor([1.0115, 2.2320])
    tensor([0.6926, 2.6341], grad_fn=<AbsBackward0>)
    tensor([0.7926, 2.7341])

    See that for `x1` the highest loss is `1.0115`, and for `x2` it is `2.8682`. This
    should be reflected `losses` and `perturbed_data` that were retained across the
    restarts.

    >>> losses
    tensor([1.0115, 2.8682])
    >>> perturbed_data
    tensor([-1.0115,  2.8682])
    """
    value_check("repeats", repeats, min_=1, incl_min=True)

    @functools.wraps(solver)
    def random_restart_fn(*args, **kwargs) -> Tuple[Tensor, Tensor]:
        targeted = kwargs.get("targeted", False)

        best_x = None
        best_loss = None

        for _ in range(repeats):
            # run the attack
            xadv, losses = solver(*args, **kwargs)

            # Save best loss for each data point
            best_loss, best_x = _replace_best(
                loss=losses,
                best_loss=best_loss,
                data=xadv,
                best_data=best_x,
                min=targeted,
            )

        assert best_x is not None
        assert best_loss is not None
        return best_x, best_loss

    return random_restart_fn


# A function that updates the best loss and best input
def _replace_best(
    loss: Tensor,
    best_loss: Optional[Tensor],
    data: Tensor,
    best_data: Optional[Tensor],
    min: bool = True,
    success: Optional[Tensor] = None,
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

    success: Optional[Tensor], shape-(N,)
        Indicates whether or not an attack was successful, and thus if
        it can be used to replace the best data and loss.

    Returns
    -------
    best_loss, best_data : Tuple[Tensor, Tensor]
    """
    if best_loss is None:
        best_data = data
        best_loss = loss
        if success is not None:
            best_data[~success] = float("nan")
            best_loss[~success] = float("inf") if min else -float("inf")
    else:
        assert best_data is not None
        if min:
            replace = loss < best_loss
        else:
            replace = loss > best_loss
        if success is not None:
            tr.logical_and(replace, success, out=replace)

        best_data[replace] = data[replace]
        best_loss[replace] = loss[replace]

    return best_loss, best_data
