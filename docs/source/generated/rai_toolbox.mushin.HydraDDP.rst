rai\_toolbox.mushin.HydraDDP
============================

.. currentmodule:: rai_toolbox.mushin

.. autoclass:: HydraDDP

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~HydraDDP.__init__
      ~HydraDDP.all_gather
      ~HydraDDP.backward
      ~HydraDDP.barrier
      ~HydraDDP.batch_to_device
      ~HydraDDP.block_backward_sync
      ~HydraDDP.broadcast
      ~HydraDDP.configure_ddp
      ~HydraDDP.connect
      ~HydraDDP.determine_ddp_device_ids
      ~HydraDDP.dispatch
      ~HydraDDP.lightning_module_state_dict
      ~HydraDDP.load_checkpoint
      ~HydraDDP.load_model_state_dict
      ~HydraDDP.load_optimizer_state_dict
      ~HydraDDP.model_sharded_context
      ~HydraDDP.model_to_device
      ~HydraDDP.on_predict_end
      ~HydraDDP.on_predict_start
      ~HydraDDP.on_test_end
      ~HydraDDP.on_test_start
      ~HydraDDP.on_train_batch_start
      ~HydraDDP.on_train_end
      ~HydraDDP.on_train_start
      ~HydraDDP.on_validation_end
      ~HydraDDP.on_validation_start
      ~HydraDDP.optimizer_state
      ~HydraDDP.optimizer_step
      ~HydraDDP.post_backward
      ~HydraDDP.post_dispatch
      ~HydraDDP.post_training_step
      ~HydraDDP.pre_backward
      ~HydraDDP.pre_configure_ddp
      ~HydraDDP.predict_step
      ~HydraDDP.process_dataloader
      ~HydraDDP.reconciliate_processes
      ~HydraDDP.reduce
      ~HydraDDP.reduce_boolean_decision
      ~HydraDDP.register_strategies
      ~HydraDDP.remove_checkpoint
      ~HydraDDP.save_checkpoint
      ~HydraDDP.set_world_ranks
      ~HydraDDP.setup
      ~HydraDDP.setup_distributed
      ~HydraDDP.setup_environment
      ~HydraDDP.setup_optimizers
      ~HydraDDP.setup_precision_plugin
      ~HydraDDP.teardown
      ~HydraDDP.test_step
      ~HydraDDP.test_step_end
      ~HydraDDP.training_step
      ~HydraDDP.training_step_end
      ~HydraDDP.validation_step
      ~HydraDDP.validation_step_end
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~HydraDDP.accelerator
      ~HydraDDP.checkpoint_io
      ~HydraDDP.distributed_sampler_kwargs
      ~HydraDDP.global_rank
      ~HydraDDP.handles_gradient_accumulation
      ~HydraDDP.is_distributed
      ~HydraDDP.is_global_zero
      ~HydraDDP.launcher
      ~HydraDDP.lightning_module
      ~HydraDDP.lightning_restore_optimizer
      ~HydraDDP.local_rank
      ~HydraDDP.model
      ~HydraDDP.node_rank
      ~HydraDDP.num_nodes
      ~HydraDDP.num_processes
      ~HydraDDP.optimizers
      ~HydraDDP.parallel_devices
      ~HydraDDP.precision_plugin
      ~HydraDDP.process_group_backend
      ~HydraDDP.restore_checkpoint_after_setup
      ~HydraDDP.root_device
      ~HydraDDP.strategy_name
      ~HydraDDP.torch_distributed_backend
      ~HydraDDP.world_size
   
   