.. meta::
   :description: Reference documentation responsible AI toolbox PyTorch perturbations

.. _pert-reference:

#############
Perturbations
#############

The rAI-toolbox provides utilities for applying perturbations to data and solving for optimal perturbations. A perturbation model is a `torch.nn.Module` whose parameters are used by its forward pass to perturb a datum or a batch of data.
E.g., `~rai_toolbox.perturbations.AdditivePerturbation` applies the perturbation :math:`x \rightarrow x + \delta` via its forward pass, where :math:`\delta` is the sole learnable parameter of this perturbation model.

Thus solving for perturbations is cast as a standard PyTorch optimization problem, where :ref:`optimizers <optim-reference>` are used to make gradient-based updates to the perturbations.
In this way, the rAI-toolbox enables adversarial perturbation workflows to be performed by standard training and testing frameworks (e.g. via `PyTorch-Lightning <https://www.pytorchlightning.ai/>`_).
We also provide some custom solvers, e.g., `~rai_toolbox.perturbations.gradient_ascent`, to facilitate this line of work.

.. _pert-models:

******
Models
******


.. currentmodule:: rai_toolbox.perturbations

.. autosummary::
   :toctree: generated/

   PerturbationModel
   AdditivePerturbation


.. _pert-solvers:

*******
Solvers
*******


.. currentmodule:: rai_toolbox.perturbations

.. autosummary::
   :toctree: generated/

   gradient_ascent
   random_restart


************
Initializers
************


.. currentmodule:: rai_toolbox.perturbations

.. autosummary::
   :toctree: generated/

   uniform_like_l1_n_ball_
   uniform_like_l2_n_ball_
   uniform_like_linf_n_ball_