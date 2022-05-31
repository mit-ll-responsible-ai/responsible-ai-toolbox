.. meta::
   :description: The changelog for rAI-toolbox, including what's new.

=========
Changelog
=========

This is a record of all past rAI-toolbox releases and what went into them, in reverse 
chronological order. All previous releases should still be available on pip.

.. _v0.2.0:

------------------
0.2.0 - 2022-XX-XX
------------------

.. note:: This is documentation for an unreleased version of the toolbox


Deprecations
------------
`~ParamTransformingOptimizer.project` is now deprecated in favor of `~ParamTransformingOptimizer._post_step_transform_`. It will be removed in rAI-toolbox v0.3.0. See :pull:`54` for details.


.. _v0.1.1:

------------------
0.1.1 - 2022-05-10
------------------


This patch fixes two bugs in ``rai_toolbox.perturbations.init``:

- `~rai_toolbox.perturbations.uniform_like_l1_n_ball_` was not correctly symmeterized; the drawn values only had components in the direction of the positive hemisphere of the :math:`L^1` ball.
- Passing an on-gpu tensor to the in-place init functions would cause a device mis-match error with the default random number generator, which is on CPU.


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
