.. meta::
   :description: Reference documentation for the responsible AI toolbox (rAI-toolbox).

.. _toolbox-reference:

#########
Reference
#########

Encyclopedia Responsibilia.

All reference documentation includes detailed Examples sections. Please scroll to the 
bottom of any given reference page to see the examples.

Wherever possible, the features provided by the rAI-toolbox are designed to: 1) adhere to common PyTorch APIs – for natural compatibility with popular machine learning frameworks and libraries – and 2) to be agnostic to the domain of application.
E.g., our APIs for :ref:`optimizing data perturbations <pert-reference>` are compatible with any tensor-based data; we do not assume that users are interested only in computer vision and natural language processing applications.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ref_optim
   ref_perturbation
   ref_datasets
   ref_augmentations
   ref_losses
   ref_mushin
   ref_utils