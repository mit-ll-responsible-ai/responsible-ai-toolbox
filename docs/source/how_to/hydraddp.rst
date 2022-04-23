.. meta::
   :description: A description of how-to run PyTorch Lightning's DDP strategy with Hydra using rai-toolbox.

.. admonition:: TL;DR
   
   Create a PyTorch Lightning configuration with ``trainer`` and ``module`` fields, e.g.,

   .. code-block:: python

      from pytorch_lightning import Trainer
      from hydra_zen import builds, make_config
      from rai_toolbox.mushin import HydraDDP
      
      MyLightningModule = # load/create your lightning module

      Config = make_config(
          module=builds(MyLightningModule)
          trainer=builds(Trainer, gpus=2, strategy=builds(HydraDDP)),
      )

   Define a task function: 
   
   .. code-block:: python

      from hydra_zen import instantiate

      def task_fn(cfg):
          obj = instantiate(cfg)
          obj.trainer.fit(obj.module))

   Simply launch a PyTorch Lightning job, e.g., ``launch(Config, task_fn)``,
   or `create a command line interface <https://mit-ll-responsible-ai.github.io/hydra-zen/tutorials/add_cli.html>`_ to run your job.

.. tip::

    Using :func:`~rai_toolbox.mushin.HydraDDP`, PyTorch Lightning's ddp-mode `Trainer <https://pytorch-lightning.readthedocs.io/en/latest/api_references.html#trainer/>`_
    **becomes compatible** with interactive environments such as Jupyter Notebooks!

.. _hydraddp:

===================================
Run PyTorch Lightning DDP in Hydra
===================================

Executing Hydra with `PyTorch Lightning's Distributed Data Parallel (DDP) Strategy <https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu_expert.html#what-is-a-strategy/>`_ 
often `has issues <https://github.com/PyTorchLightning/pytorch-lightning/issues/11300>`_
, in part because the strategy launches subprocesses where the command is derived from 
values in `sys.argv`.

The rai-toolbox comes with a custom strategy, :func:`~rai_toolbox.mushin.HydraDDP`, 
that addresses the challenge of running Hydra and Lightning together using DDP.

In this How-To we will

1. Define the requirements for a Hydra configuration
2. Build a `hydra-zen <https://github.com/mit-ll-responsible-ai/hydra-zen/>`_ configuration to execute a PyTorch Lightning multi-GPU training task
3. Execute the training task
4. Examing the logged files in the Hydra working directory

First, in order to use :func:`~rai_toolbox.mushin.HydraDDP`, the Hydra configuration 
must contain the following two configurations:

.. code-block:: reStructuredText
   :caption: 1: Define requirements for Hydra configuration
   
   ├── Config
   │    ├── trainer: A ``pytorch_lightning.Trainer`` configuration
   │    ├── module: A ``pytorch_lightning.LightningModule`` configuration


This configuration requirement allows us to define a nicely behaved task function (``rai_toolbox.mushin.lightning._pl_main.py``) that is launched for each subprocess:

.. code-block:: python
   :caption: Task Function For HydraDDP Subprocess

   def task(trainer: Trainer, module: LightningModule, pl_testing: bool, pl_local_rank: int) -> None:
       if pl_testing:
           log.info(f"Rank {pl_local_rank}: Launched subprocess using Training.test")
           trainer.test(module)
       else:
           log.info(f"Rank {pl_local_rank}: Launched subprocess using Training.fit")
           trainer.fit(module)

The configuration flags for ``pl_testing`` and ``pl_local_rank`` are automatically applied in :func:`~rai_toolbox.mushin.HydraDDP` before execution.

Next lets create an example configuration and task function using `hydra-zen <https://github.com/mit-ll-responsible-ai/hydra-zen/>`_:

.. code-block:: python
   :caption: 2: hydra-zen configuration for HydraDDP
   
   import pytorch_lightning as pl

   from hydra_zen import builds, make_config, instantiate, launch
   from rai_toolbox.mushin import HydraDDP
   from rai_toolbox.mushin.testing.lightning import TestLightningModule

   TrainerConfig = builds(
       pl.Trainer,
       accelerator="auto",
       gpus=2,
       max_epochs=1,
       fast_dev_run=True,
       strategy=builds(HydraDDP),
       populate_full_signature=True
   )

   ModuleConfig = builds(TestLightningModule)

   Config = make_config(
       trainer=TrainerConfig,
       module=ModuleConfig
   )

   def task_function(cfg):
       obj = instantiate(cfg)
       obj.trainer.fit(obj.module)

Next execute the training job

.. code-block:: python
   :caption: 3: Execute Task

   >> job = launch(Config, task_function)
   GPU available: True, used: True
   ...

Lastly, the Hydra working directory will contain these two items

- The Hydra directory, ``.hydra``, storing the YAML configuration files
- The file, ``zen_launch.log``, storing any logging outputs from the run

The log file should contain the following information:

.. code-block:: text
   :caption: 4: Output of zen_launch.log

   [2022-04-21 20:35:40,794][__main__][INFO] - Rank 1: Launched subprocess using Training.fit
   [2022-04-21 20:35:42,800][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 1
   [2022-04-21 20:35:42,801][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 0
   [2022-04-21 20:35:42,802][torch.distributed.distributed_c10d][INFO] - Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 2 nodes.
   [2022-04-21 20:35:42,810][torch.distributed.distributed_c10d][INFO] - Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 2 nodes.

Here you can see that the first line in the logged output indicates that the subprocess was launched for the second (Rank 1) GPU as expected.