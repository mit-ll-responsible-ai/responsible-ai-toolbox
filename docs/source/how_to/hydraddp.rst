.. meta::
   :description: A description of how-to run PyTorch Lightning's DDP strategy with Hydra using rAI-toolbox.

.. admonition:: TL;DR
   
   Create a minimal PyTorch Lightning configuration with ``trainer`` and ``module`` 
   fields, e.g.,

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
    becomes compatible with interactive environments such as Jupyter Notebooks!

.. _hydraddp:

===================================
Run PyTorch Lightning DDP in Hydra
===================================

Using Hydra to run `PyTorch Lightning's Distributed Data Parallel (DDP) Strategy <https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu_expert.html#what-is-a-strategy/>`_ 
often `has issues <https://github.com/PyTorchLightning/pytorch-lightning/issues/11300>`_
, in part because the strategy launches subprocesses where the command is derived from 
values in `sys.argv`.

The rAI-toolbox comes with a custom strategy, :func:`~rai_toolbox.mushin.HydraDDP`,
that addresses the challenge of running Hydra and Lightning together using DDP.

In this How-To we will:

1. Define the requirements for a Hydra configuration.
2. Build a `hydra-zen <https://github.com/mit-ll-responsible-ai/hydra-zen/>`_ configuration to execute a PyTorch Lightning multi-GPU training task.
3. Launch the training task.
4. Examine the logged files in the Hydra working directory.

First, in order to use :func:`~rai_toolbox.mushin.HydraDDP`, the Hydra configuration 
must contain the following two sub-configurations:

.. code-block:: reStructuredText
   :caption: 1: Define requirements for Hydra configuration
   
   Config
    ├── trainer: A ``pytorch_lightning.Trainer`` configuration
    ├── module: A ``pytorch_lightning.LightningModule`` configuration
    ├── datamodule: [OPTIONAL] A `pytorch_lightning.LightningDataModule` configuration


This configuration requirement enables :func:`~rai_toolbox.mushin.HydraDDP` to use a 
toolbox-provided task function (``rai_toolbox.mushin.lightning._pl_main.py``) that is 
launched for each subprocess:

.. code-block:: python
   :caption: The task function automatically run by each HydraDDP subprocess

   def task(trainer: Trainer, module: LightningModule, pl_testing: bool, pl_predicting: bool, pl_local_rank: int) -> None:
       if pl_testing:
           log.info(f"Rank {pl_local_rank}: Launched subprocess using Training.test")
           trainer.test(module)
        elif pl_predicting:
            log.info(f"Rank {pl_local_rank}: Launched subprocess using Training.predict")
            trainer.predict(module, datamodule=datamodule)
       else:
           log.info(f"Rank {pl_local_rank}: Launched subprocess using Training.fit")
           trainer.fit(module)

Note that the configuration flags for ``pl_testing``, ``pl_predicting``, and ``pl_local_rank`` are 
automatically set by :func:`~rai_toolbox.mushin.HydraDDP` before execution.

Next let's create an example configuration and task function using `hydra-zen <https://github.com/mit-ll-responsible-ai/hydra-zen/>`_:

.. code-block:: python
   :caption: 2: Creating hydra-zen configuration and task function for leveraging HydraDDP
   
   import pytorch_lightning as pl

   from hydra_zen import builds, make_config, instantiate, launch
   from rai_toolbox.mushin import HydraDDP
   from rai_toolbox.mushin.testing.lightning import SimpleLightningModule

   TrainerConfig = builds(
       pl.Trainer,
       strategy=builds(HydraDDP),
       populate_full_signature=True,
   )

   ModuleConfig = builds(SimpleLightningModule, populate_full_signature=True)

   Config = make_config(
       trainer=TrainerConfig,
       module=ModuleConfig
   )

   def task_function(cfg):
       obj = instantiate(cfg)
       obj.trainer.fit(obj.module)

Next, we launch the training job. For the purpose of this How-To, we will run only for 
a single epoch and in "fast dev run" mode.  

.. code-block:: python
   :caption: 3: Execute a Hydra-compatible ddp job using two gpus

   >>> job = launch(Config, task_function, 
   ...              overrides=["trainer.gpus=2", 
   ...                         "trainer.max_epochs=1",
   ...                         "trainer.fast_dev_run=True",
   ...                        ]
   ...              )
   GPU available: True, used: True
   ...

Lastly, the Hydra working directory will contain these two items:

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



Bonus: Adding Some Bells & Whistles to Our Hydra Application
============================================================

There are a couple of enhancements that we can add to our Hydra-based application, 
which are beyond the scope of this How-To; it is simple to `add a command line interface to our code <https://mit-ll-responsible-ai.github.io/hydra-zen/tutorials/add_cli.html>`_ and to make the :func:`~rai_toolbox.mushin.HydraDDP` strategy available 
as `a swappable configuration group <https://mit-ll-responsible-ai.github.io/hydra-zen/tutorials/config_groups.html>`_. We refer the reader to the linked tutorials for 
further explanation and instruction.

The code from this How-To has been modified accordingly and placed in the script 
``pl_trainer.py``: 

.. code-block:: python
   :caption: Contents of ``pl_trainer.py``
   
   import hydra
   from hydra.core.config_store import ConfigStore

   import pytorch_lightning as pl

   from hydra_zen import builds, make_config, instantiate
   from rai_toolbox.mushin import HydraDDP
   from rai_toolbox.mushin.testing.lightning import SimpleLightningModule

   TrainerConfig = builds(pl.Trainer, populate_full_signature=True)
   ModuleConfig = builds(SimpleLightningModule, populate_full_signature=True)

   Config = make_config(trainer=TrainerConfig, module=ModuleConfig)

   cs = ConfigStore.instance()
   cs.store(group="trainer/strategy", 
            name="hydra_ddp", 
            node=builds(HydraDDP),
   )
   cs.store(name="pl_app", node=Config)


   @hydra.main(config_path=None, config_name="pl_app")
   def task_function(cfg):
       obj = instantiate(cfg)
       obj.trainer.fit(obj.module)
   
   if __name__ == "__main__":
      task_function()

We can configure and run this code from the command line:

.. code-block:: console

   $ python pl_trainer.py +trainer/strategy=hydra_ddp trainer.gpus=2 trainer.max_epochs=1 trainer.fast_dev_run=True
   GPU available: True, used: True
   ...
