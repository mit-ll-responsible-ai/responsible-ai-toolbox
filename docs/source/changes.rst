.. meta::
   :description: The changelog for rAI-toolbox, including what's new.

=========
Changelog
=========

This is a record of all past rAI-toolbox releases and what went into them, in reverse 
chronological order. All previous releases should still be available on pip.

.. _v0.3.0:

------------------
0.3.0 - 2023-XX-XX
------------------

.. note:: This is documentation for an unreleased version of the rai-toolbox.

Improvements
------------
- Python 3.10 is officially supported.
- `rai_experiments` v0.2.0 is available via pypi.
    - `rai_experiments.models.pretrained.load_model` was added as a means download/cache model weights for rai-toolbox examples and tutorials.
- Updated `Optimizer` protocol for compatibility with torch 1.12.0+.
- Improved documentation formatting and fixed typos.
- Added format and spell checking to CI.

Compatibility-Breaking changes
------------------------------
- Support for `MultiRunMetricsWorkflow.evaluation_task` was deprecated in v0.2.0 and is now removed.
- `ParamTransformingOptimizer.project` was deprecated in v0.2.0 and is now removed.
- The minimum supported version of PyTorch is now 1.10.0.
- The minimum supported version of `hydra-zen <https://github.com/mit-ll-responsible-ai/hydra-zen>`_ is now 0.9.0.

Deprecations
------------
- `rai_toolbox.mushin.zen` has been deprecated. Use `hydra_zen.zen` instead.


.. _v0.2.1:

------------------
0.2.1 - 2022-06-16
------------------

This patch fixes the following bugs:

- ``TopQGradient`` device mismatch with user-specified RNG (see :pull:`64`)
- `MultiRunMetricsWorkflow.to_xarray` raises `ValueError` when `target_job_dirs` points to job that performed a multirun over sequence-type values (see :pull:`68`)

.. _v0.2.0:

------------------
0.2.0 - 2022-06-01
------------------

This release predominantly focuses on improvements to `rai_toolbox.mushin.MultiRunMetricsWorkflow`, which is still in early-beta and may be subject to substantial future changes

Improvements to `MultiRunMetricsWorkflow`
-----------------------------------------
- A `pre_task` step can be defined for a workflow; this is useful for seeding random number generators prior to the task's instantiation phase. See :ref:`this how-to guide <how-to-deterministic>` for examples.
- Loaded workflow overrides now roundtrip appropriately. See :pull:`61`.
- `metric_load_fn` can be overridden to customize how `~MultiRunMetricsWorkflow` loads metric files; the default behavior is to use `torch.load`. See :pull:`63`.
- `working_subdir` can be included as a data-variable in a workflow's xarray; this enables users to lookup subdirs by override values. See :pull:`52`.
- `to_xarray` works on lists of array-likes, not just lists of numpy arrays
- `load_metrics` can be called directly from `~MultiRunMetricsWorkflow`.
- `load_metrics` and `to_xarray` support loading multiple files; multiple file names can be specified as a sequence and/or a glob patterns can be provided.


Other Improvements
------------------
- The `~HydraDDP` callback now supports `pytorch_lightning.Trainer.predict`
- The project's CI now has a nightly job that runs our test suite against pre-releases of its dependencies

Compatibility-Breaking changes
------------------------------
`rai_toolbox.mushin.hydra.launch` was removed. Use `hydra_zen.launch` instead.

Deprecations
------------
- `MultiRunMetricsWorkflow.evaluation_task` is deprecated in favor of `MultiRunMetricsWorkflow.task`. It will be removed in rAI-toolbox v0.3.0. See :pull:`62`.
- `ParamTransformingOptimizer.project` is now deprecated in favor of `ParamTransformingOptimizer._post_step_transform_`. It will be removed in rAI-toolbox v0.3.0. See :pull:`54` and :pull:`59` for details.


.. _v0.1.1:

------------------
0.1.1 - 2022-05-10
------------------


This patch fixes two bugs in ``rai_toolbox.perturbations.init``:

- `~rai_toolbox.perturbations.uniform_like_l1_n_ball_` was not correctly symmeterized; the drawn values only had components in the direction of the positive hemisphere of the :math:`L^1` ball.
- Passing an on-gpu tensor to the in-place init functions would cause a device mismatch error with the default random number generator, which is on CPU.


.. _v0.1.0:

------------------
0.1.0 - 2022-05-04
------------------


This is rAI-toolbox's first stable release on PyPI! Please check out the rest of our 
docs to see what the toolbox has to offer.

We plan to have an aggressive release schedule for compatibility-preserving patches of 
bug-fixes and quality-of-life improvements (e.g. improved type annotations), and to 
regularly add features. Experimental parts of the toolbox's API that may undergo 
significant changes in future releases are documented as such.
