# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from abc import ABCMeta, abstractmethod
from functools import wraps
from typing import Callable, Dict, Iterable, TypeVar, Union, cast

import torch as tr

from .itertools import flatten_params

T = TypeVar("T", bound=Callable)
NoneType = type(None)


def freeze(
    *items: Union[
        tr.Tensor,
        tr.nn.Module,
        tr.optim.Optimizer,
        Iterable[tr.Tensor],
        Iterable[Dict[str, Iterable[tr.Tensor]]],
    ]
):
    """'Freezes' collections of tensors by setting `requires_grad=False`.
    Returns a callable that, when called, restores the state of the tensors.

    Parameters
    ----------
    *items: tr.Tensor | tr.nn.Module | tr.optim.Optimizer | Iterable[tr.Tensor] | Iterable[Dict[str, Iterable[tr.Tensor]]]
        Tensors, modules, optimizers, or param-groups.

    Returns
    -------
    unfreeze : Callable[[], None]
        Can be called without any input to restore the states of the frozen tensors.

    Notes
    -----
    'Unfreezing' the tensors restores their original states faithfully.

    Examples
    --------
    >>> import torch as tr
    >>> from rai_toolbox.utils import freeze

    Basic behavior

    >>> x = tr.tensor(1.0, requires_grad=True)
    >>> unfreeze = freeze(x)
    >>> x.requires_grad
    False
    >>> unfreeze()
    >>> x.requires_grad
    True

    Freezing a module

    >>> from torch.nn import Linear
    >>> m = Linear(2, 3)
    >>> m.weight.requires_grad, m.bias.requires_grad
    (True, True)
    >>> unfreeze = freeze(m)
    >>> m.weight.requires_grad, m.bias.requires_grad
    (False, False)
    >>> unfreeze()
    >>> m.weight.requires_grad, m.bias.requires_grad
    (True, True)
    """
    param_states: Dict[tr.Tensor, bool] = {}
    for item in items:
        if isinstance(item, tr.nn.Module):
            item = item.parameters()
        elif isinstance(item, tr.optim.Optimizer):
            item = item.param_groups

        for param in flatten_params(item):
            if param not in param_states:
                # we need to check to see if we have already encountered a parameter
                # so that we avoid overwriting its original state during a second
                # encounter
                param_states[param] = param.requires_grad
                param.requires_grad_(False)

    def restore_state():
        for p, requires_grad in param_states.items():
            p.requires_grad_(requires_grad)

    return restore_state


class ContextDecorator(metaclass=ABCMeta):
    @abstractmethod
    def __enter__(self):
        raise NotImplementedError()

    @abstractmethod
    def __exit__(self, type, value, traceback):
        raise NotImplementedError()

    def __call__(self, func: T) -> T:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return cast(T, wrapper)


class frozen(ContextDecorator):
    """A context manager/decorator for 'freezing' collections of tensors; i.e.
    `requires_grad` is set to `False` for the tensors during the context."""

    def __init__(
        self,
        *items: Union[
            tr.Tensor,
            tr.nn.Module,
            tr.optim.Optimizer,
            Iterable[tr.Tensor],
            Iterable[Dict[str, Iterable[tr.Tensor]]],
        ],
    ) -> None:
        """
        Parameters
        ----------
        *items: tr.Tensor | tr.nn.Module | tr.optim.Optimizer | Iterable[tr.Tensor] | Iterable[Dict[str, Iterable[tr.Tensor]]]
            Tensors, modules, optimizers, or param-groups to be frozen.

        Examples
        --------
        >>> import torch as tr
        >>> from rai_toolbox.utils._implementations import frozen

        Demonstrating `frozen` as a context manager.

        >>> x = tr.tensor(1.0, requires_grad=True)
        >>> with frozen(x):
        ...    print(x.requires_grad)
        False
        >>> x.requires_grad
        True

        Demonstrating `frozen` as a decorator.

        >>> x = tr.tensor(1.0, requires_grad=True)
        >>> @frozen(x)
        ... def f():
        ...     print("hello world")
        ...     return x.requires_grad
        >>> x.requires_grad # x isn't frozen until f is called
        True
        >>> f()
        hello world
        False
        >>> x.requires_grad
        True
        """
        self._items = items

    def __enter__(self):
        self._unfreeze = freeze(*self._items)

    def __exit__(self, type, value, traceback):
        self._unfreeze()


class evaluating(ContextDecorator):
    """A context manager / decorator that temporarily places one
    or more modules in eval mode during the context."""

    def __init__(self, *modules: tr.nn.Module) -> None:
        """
        Parameters
        ----------
        *modules: tr.nn.Module

        Notes
        -----
        A module's state is restored faithfully; e.g., a module that
        was already in eval mode will not be placed in train mode upon
        leaving the `evaluating` context.

        Examples
        --------
        >>> from torch.nn import Linear
        >>> from rai_toolbox import evaluating

        Using `evaluating` as a context manager.

        >>> module = Linear(1, 1)
        >>> module.training
        True
        >>> with evaluating(module):
        ...     print(module.training)
        False
        >>> module.training
        True

        Using `evaluating` as a decorator.

        >>> def f():
        ...     print("hello world")
        ...     return module.training
        >>> f = evaluating(module)(f)
        >>> module.training
        True
        >>> f()
        hello world
        False
        >>> module.training
        True
        """
        self._modules = {m: m.training for m in modules}

    def __enter__(self):
        for m in self._modules:
            m.eval()

    def __exit__(self, type, value, traceback):
        for module, train_status in self._modules.items():
            module.train(train_status)
