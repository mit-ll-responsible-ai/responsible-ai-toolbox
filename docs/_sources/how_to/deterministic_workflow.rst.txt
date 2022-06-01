.. meta::
   :description: A guide for using the rAI-toolbox to design a deterministic workflow

.. _how-to-deterministic:

===============================
Design a Deterministic Workflow
===============================

This How-To steps us through the process of designing a simple `~MultiRunMetricsWorkflow`-based workflow whose task depends on randomly-generated numbers, such that the task can be run in a deterministic manner.

We will:

1. Define a "pre-task" function that seeds the state of our random number generator
2. Write our task, which will draw from said random number generator
3. Run our workflow multiple times for various seeds to demonstrate that its behavior is indeed deterministic

Creating a non-deterministic workflow
=====================================
Our workflow will represent a simplified scenario in which our task accepts an instantiated PyTorch model and processes some data.

.. code-block:: python
   :caption: Our basic workflow

   import torch as tr
   from hydra_zen import builds, make_config, instantiate
   from rai_toolbox.mushin import MultiRunMetricsWorkflow, multirun

   class TorchWorkflow(MultiRunMetricsWorkflow):
       @staticmethod
       def task(module: tr.nn.Module, data: tr.Tensor):
           result = module(data)
           return {"out": result.detach()}

Let's assume that we are not loading ``module`` from a checkpoint, but instead are initializing it from scratch (we might be interested in training the module).
Now we will create a config that describes how to instantiate our module and data


.. code-block:: python
   :caption: Creating our config

   Config = make_config(
       module=builds(tr.nn.Linear, 10, 1),  # a single linear layer
       data=builds(tr.arange, 10.0),  # tensor([0., 1., ..., 9.])
   )

Lastly, we will run our task three independent times.

.. code-block:: pycon
   :caption: Running our task 3 times

   >>> wf = TorchWorkflow(Config)
   >>> wf.run(trial_id=multirun("abc"))
   [2022-05-26 14:49:42,656][HYDRA] Launching 3 jobs locally
   [2022-05-26 14:49:42,657][HYDRA] 	#0 : +trial_id=a
   [2022-05-26 14:49:42,717][HYDRA] 	#1 : +trial_id=b
   [2022-05-26 14:49:42,776][HYDRA] 	#2 : +trial_id=c

Accumulating and printing our results reveals that our output varies randomly from trial to trial.

.. code-block:: pycon
   :caption: Viewing the results

   >>> wf.to_xarray().out
   <xarray.DataArray 'out' (trial_id: 3, out_dim0: 1)>
   array([[ 4.452395 ],
          [ 3.0512874],
          [-2.3163323]], dtype=float32)
   Coordinates:
     * trial_id  (trial_id) <U1 'a' 'b' 'c'
     * out_dim0  (out_dim0) int64 0


This random variation is caused by the fact that instantiating ``torch.nn.Linear(10, 1)`` uses PyTorch's global random number generator to draw random values for the weights and bias for this module.
Each task execution calls ``instantiate(builds(Linear, 10, 1))`` prior to populating the ``module`` argument of our task function, thus we see three randomly-varying outputs across all trials.

Making our workflow deterministic
=================================
Adding a ``pre_task`` method to our workflow enables us to seed PyTorch's global random number generator before the linear layer is instantiated and passed to the task function.

.. code-block:: python
   :caption: A deterministic version of the workflow
   
   import torch as tr
   from hydra_zen import builds, make_config, instantiate, MISSING
   from rai_toolbox.mushin import MultiRunMetricsWorkflow, multirun

   class TorchWorkflow(MultiRunMetricsWorkflow):
       @staticmethod
       def pre_task(torch_seed: int):
           tr.manual_seed(torch_seed)
   
       @staticmethod
       def task(module: tr.nn.Module, data: tr.Tensor):
           result = module(data)
           return {"out": result.detach()}

   # We add `torch_seed` to our config
   Config = make_config(
       torch_seed=MISSING,
       module=builds(tr.nn.Linear, 10, 1),
       data=builds(tr.arange, 10.0),
   )

To verify that the results are deterministic for a given seed, we will run our workflow three times for each of two seeds.


.. code-block:: pycon
   :caption: Demonstrating that our workflow is deterministic for a given seed

   >>> wf = TorchWorkflow(Config)
   >>> wf.run(torch_seed=multirun([0, 1]), trial_id=multirun("abc"))
   [2022-05-26 16:40:09,964][HYDRA] Launching 6 jobs locally
   [2022-05-26 16:40:09,965][HYDRA] 	#0 : torch_seed=0 +trial_id=a
   [2022-05-26 16:40:10,025][HYDRA] 	#1 : torch_seed=0 +trial_id=b
   [2022-05-26 16:40:10,085][HYDRA] 	#2 : torch_seed=0 +trial_id=c
   [2022-05-26 16:40:10,242][HYDRA] 	#3 : torch_seed=1 +trial_id=a
   [2022-05-26 16:40:10,303][HYDRA] 	#4 : torch_seed=1 +trial_id=b
   [2022-05-26 16:40:10,365][HYDRA] 	#5 : torch_seed=1 +trial_id=c

   >>> wf.to_xarray().out
   <xarray.DataArray 'out' (torch_seed: 2, trial_id: 3, out_dim0: 1)>
   array([[[1.0383023 ],
           [1.0383023 ],
           [1.0383023 ]],
   
          [[0.76723164],
           [0.76723164],
           [0.76723164]]], dtype=float32)
   Coordinates:
     * torch_seed  (torch_seed) int64 0 1
     * trial_id    (trial_id) <U1 'a' 'b' 'c'
     * out_dim0    (out_dim0) int64 0

Great! Now our workflow always produces the same result when it is configured with a consistent seed.
  
.. note:: 
    
    It is inadvisable to rely on global random state in your code, as we have done in this How-To. Please
    refer to `this guide to good practices with random number generation <https://albertcthomas.github.io/good-practices-random-number-generators/>`_ for a better alternative.
