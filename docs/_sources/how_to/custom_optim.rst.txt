.. meta::
   :description: A guide for using the rAI-toolbox to create a parameter-transforming PyTorch Optimizer

.. _how-to-optim:

=========================================
Create a Parameter-Transforming Optimizer
=========================================

Popular techniques for creating adversarial or targeted perturbations often entail solving for the perturbations using gradients that have been normalized in a particular manner, as well as ensuring that the resulting perturbations are constrained within a particular domain. In the rAI-toolbox, PyTorch optimizers are used to solve for perturbations in this way.

In this How-To, we will use the `~rai_toolbox.optim.ParamTransformingOptimizer` to design an optimizer, which is particularly well-suited for optimizing perturbations, that has the following properties:

1. It encapsulates another optimizer, which is responsible for performing the gradient-based updates to the optimized parameters.
2. Prior to updating the parameter, it normalizes each parameter's gradient by their max-value, .
3. After each parameter is updated, its elements are to clamped fall within some configurable domain bounds: `[clamp_min, clamp_max]`

We will then see how we can control how this optimizer applies (or "broadcasts") these parameter transformations via the optimizer's `param_ndim` value.

Designing the optimizer
=======================
By inheriting from `~rai_toolbox.optim.ParamTransformingOptimizer` our main task is to implement two methods: `~rai_toolbox.optim.ParamTransformingOptimizer._pre_step_transform_` and `~rai_toolbox.optim.ParamTransformingOptimizer._post_step_transform_`.

The `~rai_toolbox.optim.ParamTransformingOptimizer.__init__` method for our optimizer will accept `clamp_min` and `clamp_max` as configurable options;
these bounds need to be added to the `defaults` that are passed to the to the base class so that all of the optimizer's parameter groups have access to them.
Additionally, `InnerOpt`, which is the optimizer whose `.step()` method actually performs the gradient-based updates to the parameters, will have `torch.optim.SGD` as its default; `**inner_opt_kwargs` will ultimately be passed to `InnerOpt(...)`.
Lastly, `param_ndim` is an important parameter, which will control how our max-normalization is applied to each gradient.

.. code-block:: python
   :caption: Defining the interface to our optimizer.

   from typing import Optional

   from rai_toolbox.optim import ParamTransformingOptimizer

   import torch as tr
   from torch.optim import SGD

   class CustomOptim(ParamTransformingOptimizer):
       def __init__(
           self,
           params,
           InnerOpt = SGD,
           *,
           clamp_min: Optional[float] = None,
           clamp_max: Optional[float] = None,
           defaults: Optional[dict] = None,
           param_ndim: Optional[int] = -1,
           **inner_opt_kwargs,
       ) -> None:
           if defaults is None:
               defaults = {}
           
           defaults.setdefault("clamp_min", clamp_min)
           defaults.setdefault("clamp_max", clamp_max)

           super().__init__(
               params,
               InnerOpt,
               defaults=defaults,
               param_ndim=param_ndim,
               **inner_opt_kwargs,
           )

Prior to updating its parameters during the "step" process, our optimizer will normalize each parameter's gradient by the gradient's max-value; this will be performed by `~rai_toolbox.optim.ParamTransformingOptimizer._pre_step_transform_`.
Note that we will design this method with the assumption that each parameter has a shape `(N, d1, ..., dm)`, where `N` is a batch dimension and where we want to compute `N` max values — over each of the shape-`(d1, ..., dm)` subtensors.
By adhering to this standard, `~rai_toolbox.optim.ParamTransformingOptimizer` will be able to subsequently control how `~rai_toolbox.optim.ParamTransformingOptimizer._pre_step_transform_` is applied to each parameter via `param_ndim`; this is an inherited capability that we are leveraging. Note that we don't need to use `param_ndim` directly at all.


.. code-block:: python
   :caption: Defining `_pre_step_transform_` to max-norm a parameter's gradient

   def _pre_step_transform_(self, param: tr.Tensor, optim_group: dict) -> None:
       # assume param has shape-(N, d1, ..., dm)
       if param.grad is None:
           return   
       
       # (N, d1, ..., dm) -> (N, d1 * ... * dm)
       g = param.grad.flatten(1) 
       max_norms = tr.max(g, dim=1).values  # computes N max values
       max_norms = max_norms.view(-1, *([1] * (param.ndim - 1)))  # make broadcastable
       param.grad /= tr.clamp(max_norms, 1e-20, None)  # gradient is modified *in-place*

Once a parameter has been updated, we want to ensure that its elements are clamped to fall within user-specified bounds via `~rai_toolbox.optim.ParamTransformingOptimizer._post_step_transform_`. These bounds are provided via the `optim_group` dictionary that is passed to the method.


.. code-block:: python
   :caption: Defining `_post_step_transform_` to constrain the updated parameter

   def _post_step_transform_(self, param: tr.Tensor, optim_group: dict) -> None:
       # note that the clamp is applied *in-place* to the parameter
       param.clamp_(min=optim_group["clamp_min"], max=optim_group["clamp_max"])

An advantage of accessing the clamp-bounds via `optim_group` rather than via instance-attributes is that they can then be configured on a per parameter group basis.
Note that we do not need to worry about doing any parameter reshaping for this method, since clamp occurs elementwise, and not over particular axes/dimensions of the tensor.


Putting it all together, our custom optimizer is given by

.. code-block:: python
   :caption: The full definition of `CustomOptim`

   from typing import Optional

   from rai_toolbox.optim import ParamTransformingOptimizer

   import torch as tr
   from torch.optim import SGD

   class CustomOptim(ParamTransformingOptimizer):
       def __init__(
           self,
           params,
           InnerOpt = SGD,
           *,
           clamp_min: Optional[float] = None,
           clamp_max: Optional[float] = None,
           defaults: Optional[dict] = None,
           param_ndim: Optional[int] = -1,
           **inner_opt_kwargs,
       ) -> None:
           if defaults is None:
               defaults = {}
           
           defaults.setdefault("clamp_min", clamp_min)
           defaults.setdefault("clamp_max", clamp_max)

           super().__init__(
               params,
               InnerOpt,
               defaults=defaults,
               param_ndim=param_ndim,
               **inner_opt_kwargs,
           )

       def _pre_step_transform_(self, param: tr.Tensor, optim_group: dict) -> None:
           # assume param has shape-(N, d1, ..., dm)
           if param.grad is None:
               return   
           
           # (N, d1, ..., dm) -> (N, d1 * ... * dm)
           g = param.grad.flatten(1) 
           max_norms = tr.max(g, dim=1).values  # computes N max values
           max_norms = max_norms.view(-1, *([1] * (param.ndim - 1)))  # make broadcastable
           param.grad /= tr.clamp(max_norms, 1e-20, None)  # gradient is modified *in-place*


       def _post_step_transform_(self, param: tr.Tensor, optim_group: dict) -> None:
           # note that the clamp is applied *in-place* to the parameter
           param.clamp_(min=optim_group["clamp_min"], max=optim_group["clamp_max"])


Using the optimizer
===================

First, we'll study the effect of `param_ndim` on our optimizer's behavior.
Let's create a simple shape-`(2, 2)` tensor, which will be the sole parameter that our optimizer will update.
We will clamp the parameter's elements to fall within :math:`(-\infty, 18.75]`.
The actual gradient-based parameter update will be performed by `torch.optim.SGD` with `lr=1.0`.

We'll perform a single update to our parameter, but with using `param_ndim` values of 0, 1, and 2 respectively.

.. code-block:: pycon
   :caption: Exercising our optimizer using varying `param_ndim` values.

   >>> for param_ndim in [0, 1, 2]:
   ...     x = tr.tensor([[1.0, 2.0],
   ...                    [20.0, 10.0]], requires_grad=True)
   ...
   ...     optim = CustomOptim([x], param_ndim=param_ndim, clamp_min=None, clamp_max=18.75,  lr=1.0)
   ...
   ...     loss = (x**2).sum()
   ...     loss.backward()
   ...     optim.step()  # max-norm grad -> update param -> clamp param
   ...     print(f"param_ndim={param_ndim}\nNormed grad:\n{x.grad}\nUpdated x:\n{x}\n..")
   ...     optim.zero_grad()
   param_ndim=0
   Normed grad:
   tensor([[1., 1.],
           [1., 1.]])
   Updated x:
   tensor([[ 0.0000,  1.0000],
           [18.7500,  9.0000]], requires_grad=True)
   ..
   param_ndim=1
   Normed grad:
   tensor([[0.5000, 1.0000],
           [1.0000, 0.5000]])
   Updated x:
   tensor([[ 0.5000,  1.0000],
           [18.7500,  9.5000]], requires_grad=True)
   ..
   param_ndim=2
   Normed grad:
   tensor([[0.0500, 0.1000],
           [1.0000, 0.5000]])
   Updated x:
   tensor([[ 0.9500,  1.9000],
           [18.7500,  9.5000]], requires_grad=True)
   ..

In each of these cases the parameter is then updated via `SGD([x], lr=1.0).step()` using the max-normed gradient, and the resulting parameter is clamped to the desired domain.
See that for `param_ndim=0`, the max-norm is applied elementwise to the gradient; for `param_ndim=1` the max-norm is applied independently to each 1D row; lastly, `param_ndim=2` the max-norm is applied to over the entire 2D parameter.
Controlling this behavior is important when our parameter represents a single datum (e.g. a "universal perturbation") vs a batch-style tensor. See :ref:`these docs <param-ndim-add>` for more details.

See that we can easily swap out the `SGD`-based inner optimizer for any other optimizer; let's using Adam as the inner-optimizer:


.. code-block:: pycon
   :caption: Using `Adam` as the inner-optimizer
   
   >>> from torch.optim import Adam
   >>> 
   >>> optim = CustomOptim(
   ...     [x],
   ...     InnerOpt=Adam,
   ...     clamp_min=None,
   ...     clamp_max=18.75,
   ... )
   >>> optim
   CustomOptim [Adam](
   Parameter Group 0
       amsgrad: False
       betas: (0.9, 0.999)
       clamp_max: 18.75
       clamp_min: None
       eps: 1e-08
       grad_bias: 0.0
       grad_scale: 1.0
       lr: 0.001
       maximize: False
       param_ndim: -1
       weight_decay: 0
   )

Great! We've designed our own parameter-transforming optimizer, which we could use to solve for novel data perturbations!

.. admonition:: References

   - :ref:`Off-the-shelf optimizers provided by the rAI-toolbox <built-in-optim>`
   - `~rai_toolbox.optim.ParamTransformingOptimizer`
