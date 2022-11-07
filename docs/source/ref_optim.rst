.. meta::
   :description: Reference documentation responsible AI toolbox PyTorch optimizers

.. _optim-reference:

##########
Optimizers
##########

Our optimizers are designed to compose with off-the-shelf `torch.optim.Optimizer` implementations by adding the ability to modify parameters – and their gradients – before and after the optimizer's step process.
This is facilitated by `~rai_toolbox.optim.ParamTransformingOptimizer`, which is able to compose with any PyTorch optimizer (referred to as `InnerOpt` throughout the reference docs) and add parameter/gradient-transforming capabilities to it.


The capabilities and implementations provided here are particularly useful for :ref:`solving for data perturbations <pert-reference>`.
Popular adversarial training and evaluation methods often involve normalizing parameter gradients prior to applying updates, as well as constraining (or projecting) the updated parameters.
Thus, our optimizers are particularly well-suited for such applications.
E.g., `~rai_toolbox.optim.SignedGradientOptim` implements the fast gradient sign method, and can encapsulate any other optimizer (e.g., `torch.optim.Adam`) which performs the actual gradient-based step.

.. _param-ndim:

Because these optimizers are frequently used to update perturbations of data, and not model weights, it is often necessary to control how the parameter-transformations performed by `~rai_toolbox.optim.ParamTransformingOptimizer` are broadcast over each tensor.
For example, we may be solving for a single perturbation (e.g., a "universal" perturbation), or for a *batch* of perturbations. In the latter case our parameter transformations ought to broadcast over the leading batch dimension. `param_ndim` is exposed throughout our optimizer APIs to control this behavior.
Refer to `~rai_toolbox.optim.ParamTransformingOptimizer` for more details.

All of our reference documentation features detailed Examples sections; scroll to the bottom of any given reference page to see them. 
For additional instructions for creating your own parameter-transforming optimizer please refer to :ref:`our How-To guide <how-to-optim>`.

.. _built-in-optim:

************************************** 
Base Parameter-Transforming Optimizers
**************************************

.. currentmodule:: rai_toolbox.optim

.. autosummary::
   :toctree: generated/

   ParamTransformingOptimizer
   ChainedParamTransformingOptimizer


********************************
Optimizers with Normed Gradients
********************************
.. currentmodule:: rai_toolbox.optim

.. autosummary::
   :toctree: generated/

   L1NormedGradientOptim
   L2NormedGradientOptim
   SignedGradientOptim
   L1qNormedGradientOptim


**********************************************
Miscellaneous Gradient-Transforming Optimizers
**********************************************
.. currentmodule:: rai_toolbox.optim

.. autosummary::
   :toctree: generated/

   TopQGradientOptimizer
   ClampedGradientOptimizer
   ClampedParameterOptimizer


************************************************
Optimizers with Projections Onto Constraint Sets
************************************************
.. currentmodule:: rai_toolbox.optim

.. autosummary::
   :toctree: generated/

   L2ProjectedOptim
   LinfProjectedOptim


**********************
Frank-Wolfe Optimizers
**********************
.. currentmodule:: rai_toolbox.optim

.. autosummary::
   :toctree: generated/

   FrankWolfe
   L1FrankWolfe
   L2FrankWolfe
   LinfFrankWolfe
   L1qFrankWolfe
   L1qNormedGradientOptim
