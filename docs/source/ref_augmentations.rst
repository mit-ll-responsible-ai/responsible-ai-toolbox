.. meta::
   :description: Reference documentation responsible AI toolbox PyTorch augmentations

.. _aug-reference:

#############
Augmentations
#############

The rAI-toolbox provides methods and utilities for generating diverse data
augmentations that are not readily available via other existing libraries
(e.g., `torchvision.transforms`, `kornia`).

*****************************************
Fourier-Basis Augmentations and Utilities
*****************************************
Utilities derived from `Fourier-Based Augmentations for Improved Robustness and Uncertainty Calibration <https://arxiv.org/abs/2202.12412>`_.

.. currentmodule:: rai_toolbox.augmentations.fourier

.. autosummary::
   :toctree: generated/

   FourierBasis
   generate_fourier_bases
   create_heatmaps


**********************************
AugMix Augmentations and Utilities
**********************************

Utilities derived from `AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty <https://arxiv.org/abs/1912.02781>`_.

.. currentmodule:: rai_toolbox.augmentations.augmix

.. autosummary::
   :toctree: generated/

   AugMix
   Fork
   augment_and_mix
