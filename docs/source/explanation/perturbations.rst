.. meta::
   :description: An explanation of our approach to data optimization problems.


==============================
Approach to Data Perturbations
==============================

Th rAI-toolbox facilitates solving for optimizations over data by strictly adhering to
PyTorch APIs. A perturbation model is defined as a `torch.nn.Module`, and its optimizer
as `torch.optim.Optimizer`.

In this explanation, we will discuss our motivation for
taking such a design approach, and some of the benefits it affords.

Standard Training and Adversarial Examples
==========================================

Whereas standard machine learning model-training frameworks are designed to refine
the parameters of the ML model (i.e., architecture and weights), methods for studying
the robustness and explainability of the model naturally involve analyses of and
optimizations over the data (i.e., inputs to the model and representations extracted
by the model). This optimization over the data space increases the complexity of
responsible AI worksflows over that of the standard setting.

For example, consider the standard optimization objective for training a model,
:math:`f_\theta`, parameterized by :math:`\theta`:

.. math::

    \min\limits_{\theta \in \Theta} \mathbb{E}_{(x,y)\sim D} [\mathcal{L}(f_\theta(x),y)],

where :math:`x` and :math:`y` represent the data input and corresponding output,
respectively, sampled from a data distribution, :math:`D`, and :math:`\mathcal{L}`
is the loss function to be minimized. Note that here, the data samples are fixed,
and the search is done over the model's weight space.

In contrast, consider the optimization objective for solving for an adversarial
example to fool the model into producing an incorrect output, which is a common
practice for assessing the robustness of the model:

.. math::

    \max\limits_{\delta \in \Delta} \mathcal{L}(f_\theta(x + \delta),y),

where a perturbation, :math:`\delta`, is optimized to maximize loss against the true
output, subject to a constraint set, :math:`\Delta`. Here, the model parameters are
held fixed, and the search is conducted over the data space.

A plethora of approaches for solving this objective under different loss
configurations and constraint sets have been proposed by the Robust AI research
community. One popular approach is to use iterative project gradient descent
(PGD) on the negative cross-entropy loss, with an :math:`L^p`-ball of radius
:math:`\epsilon` and :math:`p=1,2,` or :math:`\infty` as the constraint set.

Common Responsible AI Workflows
===============================

A wide range of responsible AI techniques involve optimizing parameters of data
transformations, often in addition to optimizations over model parameters as well:

- Data augmentations / corruptions: :math:`g_\delta(x)`
    - Model-independent
- Adversarial examples: :math:`\max\limits_{\delta \in \Delta} \mathcal{L}(f_\theta(g_{\delta}(x)),y)`
    - Optimize transformation over single data point
- Universal adversarial perturbations: :math:`\max\limits_{\delta \in \Delta} \mathbb{E}_{(x,y)\sim D} [\mathcal{L}(f_\theta(g_\delta(x)),y)]` 
    - Optimize transformation over data distribution
- Adversarial (robust) training: :math:`\min\limits_{\theta \in \Theta} \mathbb{E}_{(x,y)\sim D} [ \max\limits_{\delta \in \Delta} \mathcal{L}(f_\theta(g_\delta(x)),y) ]`
    - Optimize model on transformed data
- "Universal" adversarial training: :math:`\min\limits_{\theta \in \Theta} \max\limits_{\delta \in \Delta} \mathbb{E}_{(x,y)\sim D} [\mathcal{L}(f_\theta(g_\delta(x)),y) ]`
    - Optimize model on transformed data distribution

where :math:`g_\delta` represents a model for transforming data, parameterized by
:math:`\delta`.

The rai-toolbox is designed to support all of the flavors of analysis represented by
the above workflows. Users can immediately leverage our in-house perturbation
models, optimizers, and perturbation solvers, or build their own in a manner that
can be easily composed with other existing tools from the PyTorch ecosystem for creating
distributed and scalable Responsible AI workflows.
