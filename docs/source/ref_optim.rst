.. meta::
   :description: Reference documentation responsible AI toolbox PyTorch optimizers

.. _optim-reference:

##########
Optimizers
##########

WHAT OUR OPTIMIZERS ARE ALL ABOUT (PyTorch-centric)

***************
Base Optimizers
***************
.. currentmodule:: rai_toolbox.optim

.. autosummary::
   :toctree: generated/

   ParamTransformingOptimizer
   ChainedGradTransformerOptimizer


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


************************************************
Optimizers with Projections Onto Constraint Sets
************************************************
.. currentmodule:: rai_toolbox.optim

.. autosummary::
   :toctree: generated/

   L2ProjectedOptim
   LinfProjectedOptim


**********************
Frank Wolfe Optimizers
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
