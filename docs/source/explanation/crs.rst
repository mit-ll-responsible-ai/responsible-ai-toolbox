.. meta::
   :description: Our Approach to Configurable, Reproducible, and Scalable AI.

===========================================================
Our Approach to Configurable, Reproducible, and Scalable AI
===========================================================

One major objective of `rai_toolbox` is to provide a framework that not only enables the evaulation and 
enhancement of responsible and explanaible AI, but to provide the tools via `rai_toolbox.mushin` and 
`hydra-zen <https://github.com/mit-ll-responsible-ai/hydra-zen/>`_ that are:

- **Configurable**: All aspects of an application—including deeply nested components—are configured from a single interface. 
- **Repeatable**: Each run of the application is self-documenting; the full configuration of the software is logged for each run.
- **Scalable**: Multiple configurations of the application can be launched—to sweep or search over configuration subspaces—using local or distributed job launcher methods.

Configurable
============

`Hydra's <https://hydra.cc/>`_ and `hydra-zen's <https://github.com/mit-ll-responsible-ai/hydra-zen/>`_ role is to

- Remove boilerplate for Command Line Interfaces (CLI), configuration files, and logging
- Provide a powerful configuration management that is extendable and dynamically composable to support large-scale, repeatable, experimentation
- Support pluggable architectures to launch jobs on distributed clusters or sweep parameters and optimize hyper-parameters of a Machine Learning (ML) model

`Hydra <https://hydra.cc/>`_ provides the tools to describe and configure hierarchical sturctures of Python-based software applications using YAML-based configuration.
Nested components of one's AI system can be configured with a single interface and a complete configuration of the entire application can be represented via a YAML file.
Additionally, one can define "configuration groups" using Hydra's `ConfigStore API <https://hydra.cc/docs/tutorials/structured_config/config_store/>`_ allowing for configurations
to be "swapped" in an ergonomic way.  

Here is an example using `hydra-zen` to build a configuration and show how the configuration is serialized to a YAML file:

.. code-block:: python
   :caption: Using hydra-zen to generate configurations and serialize to a YAML file

   >>> from hydra_zen import builds, to_yaml
   >>> from torch.optim import Adam
   >>> OptimConfig = builds(Adam, ...)
   >>> print(to_yaml(OptimConfig))
   _target_: torch.optim.adam.Adam
   params: ???
   lr: 0.001
   betas:
   - 0.9
   - 0.999
   eps: 1.0e-08
   weight_decay: 0
   amsgrad: false

To demonstrate the power of "swappable" configurations we will build an example application below
to allow for different configurations for the "optim" configuration.

.. code-block:: python
   :caption: `experiement.py`: A Hydra application with swappable configurations

   import hydra
   from hydra.core.config_store import ConfigStore
   from hydra_zen import builds, make_config, MISSING, launch
   from torch.optim import Adam, SGD
   
   # Build multiple configurations for optimizers
   AdamConfig = builds(Adam, lr=0.1, zen_partial=True)
   SGDConfig = builds(SGD, lr=0.01, zen_partial=True

   # Create experiment configuration
   Config = make_config(optim=MISSING)

   # Store configs in Hydra's ConfigStore
   cs = ConfigStore.instance()
   cs.store(name="adam", group="optim", node=AdamConfig)
   cs.store(name="sgd", group="optim", node=SGDConfig)

   cs.store(name="experiment_config", node=Config)

   @hydra.main(config_path=None, config_name="experiment_config")
   def task_fn(cfg):
      optim = instantiate(cfg.optim)
      print(optim.__name__)
   
   if __name__ == "__main__":
      task_fn()
   
Using Hydra's CLI the application can be executed using different values for the ``optim`` configuration.

.. code-block:: bash
   :caption: Using Hydra CLI to execute with different configurations

   $ python experiment.py +optim=sgd
   $ python experiment.py +optim=adam


.. tip::

   `hydra-zen <https://github.com/mit-ll-responsible-ai/hydra-zen/>`_ provides elegant tools for generating and customizing Hydra-compatible configurations without
   writing YAML configurations for the entire (and often complex) software system. hydra-zen eliminates this cost by enabling a Python-centric, ergonomic
   workflow for dynamically populating and automatically validating configurations for one's entire software application.


Repeatable
==========

Reproducibility is a natural consequence of the configurability: each job launched by Hydra is documented by—and can be fully
replicated by—the YAML configuration that is automatically recorded for that job.  The YAML configuration is stored within the 
experiment directory::

   ├── <experiment directory name>
   |    ├── <hydra configuration subdirectory: (default: .hydra)>
   |    |    ├── config.yaml
   |    |    ├── hydra.yaml
   |    |    ├── overrides.yaml
   |    ├── <logged data>

One method for repeating the experiment is to use `Hydra's CLI <https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/>`_:

.. code-block:: bash
   :caption: Repeating an experiment using Hydra CLI and saved YAML configuration

   $ python experiment.py --config-path <YAML configuration directory> -config-name config


Scalable
========

Scalability can be achieved in two ways: 1) using Hydra's ``multirun`` to run experiments using multiple
configurations and configuration parameters and 2) by utilizing Hydra's ``launcher`` architecture to launch
multiprocessing and distributed jobs locally, on cluster architectures, and in the cloud.

Hydra ``multirun`` allows one to launch multiple experiments via a simple interface.  For example, to launch experiments
for multiple configurations and parameters, simply run

.. code-block:: bash
   :caption: Using Hydra ``multirun`` to launch 4 different experiments.

   $ python experiment.py +optim=sgd,adam optim.lr=0.1,0.2 --multirun

Each experiment configuration and data will be logged in indvidual directories and therefore each experiment
is repeatable without running all the experiments::

   ├── <multirun directory>
   │    ├── <experiment directory name: 0>
   │    |    ├── <hydra output subdirectory: (default: .hydra)>
   |    |    |    ├── config.yaml
   |    |    |    ├── hydra.yaml
   |    |    |    ├── overrides.yaml
   │    |    ├── <metrics_filename>
   │    ├── <experiment directory name: 1>
   |    |    ...


For Hydra ``launcher`` capabilities, here a couple useful examples

- Launching multiple parallel jobs using `JobLib.Parallel`: `Hydra JobLib Launcher <https://hydra.cc/docs/plugins/joblib_launcher/>`_
- Launching on a `SLURM <https://slurm.schedmd.com/documentation.html/>`_ cluster: `Hydra Submitit Launcher <https://hydra.cc/docs/plugins/submitit_launcher/>`_

