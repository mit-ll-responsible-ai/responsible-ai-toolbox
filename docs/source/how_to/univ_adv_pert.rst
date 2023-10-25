.. meta::
   :description: A guide for using the rAI-toolbox to solve for a universal adversarial perturbation

.. _how-to-univ-pert:

==================================
Solve for a Universal Perturbation
==================================

A universal adversarial perturbation (UAP) is a *single* perturbation that can be applied *uniformly to any data* in order to substantially reduce the quality of a machine learning model's predictionsf on that data.
In the case of an additive perturbation (i.e., :math:`x \rightarrow x + \delta`), the single perturbation, :math:`\delta`, is optimized over a data distribution:

.. math::

    \max\limits_{\delta \in \Delta} \mathbb{E}_{(x,y)\sim D} [\mathcal{L}(f_\theta(x + \delta,y)]

The paper `Universal Adversarial Training <https://arxiv.org/pdf/1811.11304.pdf>`_ by Shafahi et al propose a simple and efficient gradient-based optimization method for solving for such a UAP.

In this How-To, we demonstrate how the rAI-toolbox's perturbation models, optimizers, and solvers naturally accommodate this approach to solving for UAPs.
Specifically, we will

1. Create some toy-data and a trivial model.
2. Initialize a perurbation model that is designed to apply a single perturbation across a batch of data.
3. Configure a perturbation-optimizer to normalize the gradient of the single perturbation, and to project the updated perturbation to fall within an :math:`\epsilon`-sized :math:`L^2` ball.
4. Configure our solver to aggregate the loss such that the gradient is invariante to batch-size.

It should be noted that the data and model that we are using here are meaningless, and thus nor will the resulting UAP be of any particular interest; this How-To is only meant to demonstrate mechanics rather than a real-world application.


Let's assume that our model is a classifier that classifies images among ten classes.
For our toy data, let's create a batch of image-like tensors; we'll create a shape-`(N=9, C=3, H=4, W=4)` tensor, which represents a batch of nine 4x4 images with three color channels. Accordingly, we'll also create a shape-`(N=9,)` tensor of integer-values labels. Our toy model will consist a `conv-relu-dense` sequence of layers.


.. code-block:: python
   :caption: 1 Creating toy data and model

   import torch as tr
   from torch.nn import Conv2d, Linear, Sequential, ReLU, Flatten
   
   tr.manual_seed(0)
   
   data = tr.rand(9, 3, 4, 4)
   truth = tr.randint(low=0, high=10, size=(9,))
   model = Sequential(Conv2d(3, 5, 4), Flatten(), ReLU(), Linear(5, 10))

We will use an additive perturbation, which adds as single shape-`(C=3, H=4, W=4)` perturbation to the batch of data (broadcasting over the batch dimension).
Rather than specifying the precise shape of the perturbation, we can simply indicate `delta_ndim=-1` to indicate that we want to drop the batch-dimension from `data`


.. code-block:: python
   :caption: 2 Initialize a universal perturbation model

   from rai_toolbox.perturbations import AdditivePerturbation

   pert_model = AdditivePerturbation(data, delta_ndim=-1)

.. code-block:: pycon
   :caption: Inspecting the initialized perturbation parameter, :math:`\delta`
   
   >>> pert_model.delta.shape
   torch.Size([3, 4, 4])

   >>> pert_model.delta
   Parameter containing:
   tensor([[[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]],
   
           [[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]],
   
           [[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]]], requires_grad=True)


Now we will initialize an optimizer that will update our perturbation. `~rai_toolbox.optim.L2ProjectedOptim` is designed to normalize a gradient by its :math:`L^2` norm prior to performing the parameter updated (using a `SGD` step by default). The updated perturbation is then projected into a :math:`\epsilon`-sized :math:`L^2` ball.
We will use an `SGD` step with `lr=0.1` and a momentum of 0.9; we pick :math:`\epsilon=2`. In order to ensure that these gradient and parameter-transforming operations apply appropriately to our UAP, we specify `param_ndim=None` (see :ref:`these docs <param-ndim-add>` for more details).


.. code-block:: python
   :caption: 3 Initialize the perturbation optimizer

   from rai_toolbox.optim import L2ProjectedOptim

   optim = L2ProjectedOptim(
       pert_model.parameters(),
       epsilon=2,
       param_ndim=None,
       lr=0.1,
       momentum=0.9,
   )

.. code-block:: pycon
   :caption: Inspecting the optimizer
   
   >>> optim
   L2ProjectedOptim [SGD](
   Parameter Group 0
       dampening: 0
       epsilon: 2
       grad_bias: 0.0
       grad_scale: 1.0
       lr: 0.1
       maximize: False
       momentum: 0.9
       nesterov: False
       param_ndim: None
       weight_decay: 0
   )


Finally, we run `~rai_toolbox.perturbations.gradient_ascent` for ten steps to solve for our UAP.
By default, this uses cross-entropy loss.
Note that we must reduce our loss using `torch.mean`, not the default `torch.sum`, so that the gradient of our single perturbation is not scaled by batch-size.


.. code-block:: pycon
   :caption: 4 Solving for the UAP
   
   >>> from rai_toolbox.perturbations import gradient_ascent

   >>> xadv, losses = gradient_ascent(
   ...  model=model,
   ...  data=data,
   ...  target=truth,
   ...  perturbation_model=pert_model,
   ...  optimizer=optim,
   ...  reduction_fn=tr.mean, 
   ...  steps=10,
   ... )

   >>> pert_model.delta  # the UAP solution
   Parameter containing:
   tensor([[[-0.0670,  0.0970,  0.1258,  0.4220],
            [ 0.5926, -0.0525, -0.4427, -0.1195],
            [ 0.5156, -0.0750, -0.0547, -0.1155],
            [-0.0163,  0.5255,  0.1064, -0.4517]],
   
           [[-0.0072, -0.4236,  0.4355, -0.2047],
            [-0.0961,  0.4003, -0.1891, -0.1592],
            [-0.1453,  0.0062, -0.4630,  0.6092],
            [-0.1998, -0.1265,  0.4888,  0.0515]],
   
           [[-0.0886,  0.2541,  0.1003,  0.0585],
            [ 0.3680,  0.1455, -0.4263, -0.0196],
            [-0.1726,  0.1401, -0.5161, -0.1914],
            [ 0.1375,  0.3058,  0.1123,  0.0247]]], requires_grad=True)

   >>> tr.linalg.norm(pert_model.delta.flatten(), ord=2)  # verify that δ falls in eps-2 ball
   tensor(2.0000, grad_fn=<CopyBackwards>)

Let's check that the loss of the clean batch of data is less than the loss of the perturbed batch

.. code-block:: pycon
   :caption: Verifying the attack
   
   >>> from torch.nn.functional import cross_entropy
   
   >>> cross_entropy(model(data), truth)
   tensor(2.2578, grad_fn=<NllLossBackward0>)
   
   >>> pert_data = pert_model(data)
   >>> cross_entropy(model(pert_data), truth)
   tensor(2.4227, grad_fn=<NllLossBackward0>)

Great! Our UAP for this toy problem reduces the average performance of our model on uniformally-perturbed data.

.. note:: 

    The paper `Universal Adversarial Training <https://arxiv.org/pdf/1811.11304.pdf>`_ utilizes a clamped cross-entropy loss. In the workflow presented here, one would pass a clamped version of the loss via the `criterion` argument to `~rai_toolbox.perturbations.gradient_ascent`.

