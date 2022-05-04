.. meta::
   :description: Reference documentation for the mushin submodule of the responsible AI toolbox (rAI-toolbox).

.. _mushin-reference:

######
Mushin
######

.. important:: The features provided by `rai_toolbox.mushin` are in early-beta phase and are subject to compatibility-breaking changes in the future.

.. note:: `rai_toolbox.mushin` requires additional installation dependencies. These can be installed via
   .. console:: 
      
      $ pip install rai-toolbox[mushin]

`Mushin <https://en.wikipedia.org/wiki/Mushin_(mental_state)>`_ means, roughly, 
*"no-mind"*;
`rai_toolbox.mushin` provides utilities that greatly reduce the "boilerplate" code and overall complexity of conducting machine learning experiments, tests, and analyses.
Unlike the rest of the toolbox, which adheres solely to essential PyTorch APIs, `mushin` is intentionally opinionated and specialized in its design;
it reflects our (the toolbox dev team's) shared workflows, best practices, and favorite tools for doing research and development.

As such, `mushin` is designed around `PyTorch-Lightning <https://www.pytorchlightning.ai/>`_, which facilitates boilerplate-free and performant machine learning work,
and around `hydra-zen <https://github.com/mit-ll-responsible-ai/hydra-zen>`_, which makes it easy to design configurable and reproducible workflows that leverage `the Hydra framework <https://github.com/facebookresearch/hydra>`_.



*********
Workflows
*********

Workflows are designed to simplify and automate the process of configuring, running, and reproducing various data science and machine learning workflows.
In part, these serve to greatly simplify the process of organizing and running jobs using the Hydra framework, and aggregating the results of those jobs for analysis.

.. currentmodule:: rai_toolbox.mushin

.. autosummary::
   :toctree: generated/

   BaseWorkflow
   MultiRunMetricsWorkflow
   RobustnessCurve


***************************
PyTorch-Lightning Utilities
***************************

Tools and utilities that make PyTorch-Lightning easy to use for our work. Some of these utilities also enable much-needed compatibility between PyTorch-Lightning and Hydra.

.. currentmodule:: rai_toolbox.mushin

.. autosummary::
   :toctree: generated/

   MetricsCallback
   HydraDDP
