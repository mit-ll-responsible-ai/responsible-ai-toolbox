.. meta::
   :description: An explanation of the responsible AI approach to data optimization problems.


=====================
On Data Perturbations
=====================

This article provides an overview of common data optimization problems, using model-dependent criteria, that are relevant for assessing and enhancing robustness in machine learning models.
We present a broad picture of how we have designed the rAI-toolbox to solve these problems, and provide pseudo-code that illustrates how :ref:`optimizations <optim-reference>` over :ref:`data perturbations <pert-reference>` adhere strictly
to core `PyTorch <https://pytorch.org/>`_ APIs; i.e.,
a perturbation model is defined as a `torch.nn.Module`, and its optimizer as `torch.optim.Optimizer`. 


Model optimization vs. data optimization
========================================

Standard machine learning model-training frameworks are designed to refine
the parameters of the machine learning model (i.e., its architecture and weights), while methods for studying
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
output, subject to a constraint set, :math:`\Delta`. Here, the model parameters
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
   :caption: Notional perturbation solver

   def perturbation_solver(
       model: callable,
       data: Tensor,
       target: Tensor,
       lr: float,
       steps: int,
       criterion: callable,
       initialize_fn: callable,
       project_fn: callable,
   ) -> Tensor:
       # initialize perturbation parameter
       delta = initialize_fn(data)
   
       for _ in range(steps):
           # perturbation applied manually / in-line
           perturbed_data = data + delta
   
           # calculate loss
           loss = criterion(model(perturbed_data), target)
   
           # optimize
           grad = autograd(loss, delta)
           with no_grad():
               delta = delta + lr * grad  # perturbation updated manually / in-line
               delta = project_fn(delta)
       return delta


Note that the code for applying the perturbation and taking an optimization
step is embedded within the for loop of the solver. If a user wanted to swap
out the optimizer methodology or use a different perturbation model, one would
need to write an entirely new solver.

By adhering to PyTorch APIs, the rAI-toolbox frames the process of solving for a perturbation in the standard workflow for training ML models. I.e., we specify perturbation models, which are responsible for initializing, storing, and applying perturbations, and perturbation optimizers, which update the perturbations based on their gradients while also applying normalizations and constraints to the perturbations and their gradients.


.. code-block:: python
   :caption: rAI-toolbox approach to solving for perturbations
   
   from torch.nn import Module
   from torch.optim import Optimizer
   
   # Implements PyTorch Module API
   class PerturbationModel(Module):
      def __init__(self, *args, **kwargs):
         super().__init__()
         # initialize parameters of perturbation model
      
      def forward(self, x):
         perturbed_data = # use model's parameters to perturb data 
         return perturbed_data

   # Implements PyTorch Optimizer API
   class PerturbationOptimizer(Optimizer):
      def _pre_step_(self, param, **kwds): # e.g., perform gradient-normalization
      def _step_(self, param, **kwds): # perform gradient-based update on parameter
      def _post_step_(self, param, grad): # e.g., project updated parameter into constraint set

      def step(self):
         for param in self.all_params:
            self._pre_step_(param)
            self._step_(param, param.grad)
            self._post_step_(param)


Having framed the perturbation process as a `torch.nn.Module`, whose parameters (e.g., the perturbation itself) are optimized and constrained via the `torch.optim.Optimizer` API, we can take any standard trainer, e.g.:

.. code-block:: python
   :caption: A standard PyTorch trainer 

   def standard_trainer(model, data, target, optimizer, steps, criterion):
      for _ in range(steps):
         # calculate loss
         loss = criterion(model(data), target)

         # optimize
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()


and solve for the optimal perturbation via:

.. code-block:: python
   :caption: Solving for perturbations using a standard PyTorch trainer

   from torch.nn import Sequential
   from rai_toolbox import freeze

   pert_model = PerturbationModel(...)
   optim = PerturbationOptimizer(pert_model.parameters(), ...)

   ml_model = MyNeuralNetwork(...)

   # model(data) -> ml_model(pert_model(data))
   model = Sequential(pert_model, freeze(ml_model.eval()))

   # solve for perturbations
   standard_trainer(model, optimizer=optim, data=..., target=..., steps=..., criterion=...)

   # solved perturbations are stored in `pert_model`

We can then use `pert_model` to apply these optimized perturbations to new data

.. code-block:: python
   :caption: Peturbing data

   data = # some tensor of data
   pert_data = pert_model(data)  # applies optimized peturbation to `data`

The abstractions provided by a perturbation model and a perturbation optimizer yields a natural delegation of functionality, which makes it easy for us to modify the critical implementation details of this problem. E.g., One can modify the optimizer to adjust how the perturbation is constrained, or how its gradient is normalized; the perturbation model controls the random initialization of the perturbation and how the perturbation broadcasts over a batch of data. None of these adjustments require any modification to the process by which we actually solve for the perturbations; i.e., we can continue to use `standard_trainer` or any gradient-based solver.

`~rai_toolbox.optim.ParamTransformingOptimizer`, `~rai_toolbox.perturbations.AdditivePerturbation`, and `~rai_toolbox.perturbations.gradient_ascent` represent concrete implementations of this design; the reader is advised to consult their reference documentation for further insights into the rAI-toolbox's approach to solving for data perturbations.


Common data-related workflows supported by rAI-toolbox
======================================================

A wide range of responsible AI techniques involve optimizing parameters of data
perturbations (or more generally, *transformations*), often in addition to optimizations over model parameters:

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

The rAI-toolbox is designed to support all of the flavors of analysis represented by
the above workflows. Users can immediately leverage the toolbox's perturbation
:ref:`models <pert-models>`, :ref:`optimizers <optim-reference>`,
and :ref:`solvers <pert-solvers>`, or build their own in a manner that can be easily composed with other existing tools from the PyTorch ecosystem for creating distributed and scalable Responsible AI workflows.
