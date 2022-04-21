.. _hydraddp:

===================================
PyTorch Lightning HydraDDP Strategy
===================================

Using Hydra with PyTorch Lightning's Distributed Data Parallel (DDP)
`Strategy <https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu_expert.html#what-is-a-strategy/>`_
has challenges due to two specific issues:

1. Hydra changes directories to a local `working directory <https://hydra.cc/docs/1.0/tutorials/basic/running_your_app/working_directory/>`_
2. PyTorch Lightning ``DDP`` strategy launches subprocesses for each local GPU using the values in `sys.argv`.

The rai-toolbox comes with a custom strategy, :func:`~rai_toolbox.mushin.HydraDDP`, that aims to help address
the challenge of running Hydra and Lightning together for multiple GPUs.

In this How-To we will

1. Define the requirements for a Hydra configuration to run with :func:`~rai_toolbox.mushin.HydraDDP`.
2. Build a hydra-zen configuration to execute a PyTorch Lightning mult-GPU training task
3. Execute the training task
4. Examing the logging directory structure

First, in order for :func:`~rai_toolbox.mushin.HydraDDP` to properly execute the Hydra configuration must
contain the following to configuration:

- trainer: A ``pytorch_lightning.Trainer`` configuration
- module: A ``pytorch_lightning.LightningModule`` configuration

This is because each child subprocess launched by PyTorch Lightning will execute the following task function
defined in ``rai_toolbox.mushin.lightning._pl_main.py``:

.. code-block:: python
   :caption: Task Function For DDP Child Processes

    def task(trainer: Trainer, module: LightningModule, pl_testing: bool) -> None:
        if pl_testing:
            trainer.test(module)
        else:
            trainer.fit(module)

The configuration flag for ``pl_testing`` is automatically applied to the configuration before execution.

Next lets define an example of configuration using this custom strategy:

.. code-block:: python
   :caption: hydra-zen configuration for HydraDDP
   
   import pytorch_lightning as pl

   from hydra_zen import builds, make_config, instantiate, launch
   from rai_toolbox.mushin import HydraDDP
   from rai_toolbox.testing import TestLightningModule

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

Next lets execute the training job

.. code-block:: python
   :caption: Execute Task

   >> job = launch(Config, task_function)
   GPU available: True, used: True
   ...

Lastly, the Hydra working directory will now have these two items

1. The Hydra directory, ``.hydra``, storing the YAML configuration files
2. The logging file, ``zen_launch.log`` which should have the following output

.. code-block:: text
   :caption: Output of zen_launch.log

   [2022-04-21 16:51:29,767][__main__][INFO] - Launching child process using Training.fit
   [2022-04-21 16:51:31,773][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 1
   [2022-04-21 16:51:31,776][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 0
   [2022-04-21 16:51:31,776][torch.distributed.distributed_c10d][INFO] - Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 2 nodes.
   [2022-04-21 16:51:31,783][torch.distributed.distributed_c10d][INFO] - Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 2 nodes.

Here you can see that the first line in the logged output indicates that the child process was launched.