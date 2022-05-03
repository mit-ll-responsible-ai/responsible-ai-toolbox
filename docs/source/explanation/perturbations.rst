.. meta::
   :description: An explanation of our approach to data optimization problems.


==============================
Approach to Data Perturbations
==============================

The rai-toolbox facilitates solving for optimizations over data perturbations by strictly
adhering to `PyTorch <https://pytorch.org/>`_ APIs. A perturbation model is defined as a 
`torch.nn.Module`, and its optimizer as `torch.optim.Optimizer`. This allows us to
leverage PyTorch's `automatic differentiation <https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>`_
engine to solve for optimal perturbations to data.


Model optimization vs. data optimization
========================================

Standard machine learning model-training frameworks are designed to refine
the parameters of the ML model (i.e., architecture and weights), while methods for studying
the robustness and explainability of the model naturally involve analyses of and
optimizations over the data (i.e., inputs to the model and representations extracted
by the model). This optimization over the data space increases the complexity of
responsible AI workflows over that of the standard setting.

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
output, subject to a constraint set, :math:`\Delta`. Note that now, the model parameters
are held fixed, and the search is conducted over the data space.

A plethora of approaches for solving this objective under different loss
configurations and constraint sets have been proposed by the Robust AI research
community. One popular approach is to use iterative project gradient descent
(PGD) on the negative cross-entropy loss, with an :math:`L^p`-ball of radius
:math:`\epsilon` and :math:`p=1,2,` or :math:`\infty` as the constraint set.

Solvers for data perturbations
==============================

A variety of other tools exist that implement large libraries of techniques
proposed by the research community (such as PGD) in a framework-agnostic API.
Their perturbation solvers are often written from scratch and look something like this:

.. code-block:: python
   :caption: Notional perturbation solver implementing PGD

   def perturbation_solver(model, data, target, epsilon, lr, steps):
      # initialize perturbation parameter
      delta = initialize(data)

      for i in range(steps):
         # perturbation model
         perturbed_data = data + delta
         
         # calculate loss
         loss = criterion(model(perturbed_data), target)
         
         # optimize
         grad = autograd(loss, delta)
         with no_grad():
            delta = delta + lr * grad
            delta = project(delta, epsilon)

Note that the code for applying the perturbation and taking an optimization
step is embedded within the for loop of the solver. If a user wanted to swap
out the optimizer or use a slightly different perturbation model, they would
need to re-write an entirely new solver.

By adhering to PyTorch APIs, the rai-toolbox provides a generic perturbation
solver that looks similar to the standard workflow for training ML models,
and enables users to easily swap out optimizers and perturbation modules:

.. code-block:: python
   :caption: rai-toolbox approach to implementing PGD
   
   from torch.nn import Module
   from torch.optim import Optimizer
   
   # Implements PyTorch Module API
   class AdditivePerturbation(Module):
      def __init__(self, data):
         super().__init__()
         self.delta = initialize(data)
      
      def forward(self, x):
         return x + self.delta

   # Implements PyTorch Optimizer API
   class ProjectedOptimizer(Optimizer):
      def __init__(self, params, lr, epsilon):
         super().__init__(params)
         self.lr = lr
         self.epsilon = epsilon

      def step(self):
         update(params, self.lr)
         project(params, self.epsilon)

   def perturbation_solver(model, data, target, perturbation_model, optimizer, steps):
      # initialize perturbation and optimizer
      perturb = perturbation_model(data)
      optim = optimizer(perturb.parameters())

      for i in range(steps):
         # perturbation model
         perturbed_data = perturb(data)

         # calculate loss
         loss = criterion(model(perturbed_data), target)

         # optimize
         opt.zero_grad()
         loss.backward()
         opt.step()


Common data-related workflows supported by `rai-toolbox`
========================================================

A wide range of responsible AI techniques involve optimizing parameters of data
transformations, often in addition to optimizations over model parameters:

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
`models <https://mit-ll-responsible-ai.github.io/responsible-ai-toolbox/ref_perturbation.html#models>`_,
`optimizers <https://mit-ll-responsible-ai.github.io/responsible-ai-toolbox/ref_optim.html>`_,
and `solvers <https://mit-ll-responsible-ai.github.io/responsible-ai-toolbox/ref_perturbation.html#solvers>`_,
or build their own in a manner that can be easily composed with other existing tools
from the PyTorch ecosystem for creating distributed and scalable Responsible AI workflows.
