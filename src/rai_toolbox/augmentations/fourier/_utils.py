# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from functools import wraps
from inspect import Parameter, signature
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import torch as tr
from torch.nn import Module
from torch.nn.functional import softmax

_F = TypeVar("_F", bound=Callable[..., Any])


def get_device(module: Module) -> tr.device:
    for p in module.parameters():
        return p.device

    # no parameters
    return tr.device("cpu")


class ModuleInfo(NamedTuple):
    module: Module
    device: tr.device
    was_training: bool
    name: str


class evaluating:
    """A decorator that puts modules in eval mode, but preserves their training-states
    and devices after the decorated function is complete.

    >>> @evaluating('model1', 'model2')
    ... def func(*args, model1: nn.Module, model2: nn.Module, **kwargs):
    ...     # `model1` and `model2` are placed in eval-mode.
    ...     # After the function is executed, their training
    ...     # states are reverted, and they are placed back on
    ...     # their original devices
    """

    def __init__(self, *var_names: str, no_grad: bool = True):
        """
        Parameters
        ----------
        *var_names : str
            The names of the modules to be placed in eval mode.
            These must be keyword-only arguments in the function
            being decorated.

        no_grad : bool, optional (default=True)
            If True, the decorated function is run within a `torch.no_grad()`
            context."""
        # keeps track of what MemGuard was at a given depth
        self._var_names = var_names
        self._no_grad = no_grad

    def __call__(self, func: _F) -> _F:
        """Decorates a function within the context"""
        # TODO: Make this work for non-named arguments

        sig_mapping = signature(func).parameters

        for name in self._var_names:
            if (
                name not in sig_mapping
                or sig_mapping[name].kind != Parameter.KEYWORD_ONLY
            ):
                raise TypeError(
                    f"Internal Error: `@evaluating({name})` was specified, but {name}"
                    f" is either not in the signature or is not a keyword-only argument."
                )

        modules: List[ModuleInfo] = []

        if self._no_grad:
            func = tr.no_grad()(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            for name in self._var_names:
                module: Module = kwargs[name]
                modules.append(
                    ModuleInfo(module, get_device(module), module.training, name)
                )
            try:
                for item in modules:
                    item.module.eval()
                return func(*args, **kwargs)
            finally:
                for item in modules:
                    if item.was_training:
                        item.module.train()
                    item.module.to(item.device)

        return cast(_F, wrapper)


@evaluating("model")
def process_metrics(
    *,
    dataloader: Collection[Tuple[tr.Tensor, tr.Tensor]],
    model: tr.nn.Module,
    metrics: Dict[str, Callable[[tr.Tensor, tr.Tensor], Any]],
    device: Optional[Union[tr.device, str, int]] = None,
):
    all_logits = []
    all_targets = []

    if device is not None:
        device = tr.device(device)
        model.to(device=device)
    else:
        device = get_device(model)

    for batch, targets in dataloader:
        batch = batch.to(device=device)
        targets = targets.to(device=device)

        all_logits.append(model(batch).cpu())
        all_targets.append(targets.cpu())

    all_pred = softmax(tr.cat(all_logits, dim=0), dim=1)
    all_targets = tr.cat(all_targets, dim=0)
    return {name: metric(all_pred, all_targets) for name, metric in metrics.items()}
