# rai-experiments

This directory contains experiments and analyses that leverage the capabilities of the rai-toolbox.

Each entry in `experiments/` is an independent "project" that reproduces certain research results, or
that demonstrates particular features of the toolbox in an applied setting.


## Installing rai-experiments utilities

`rai-experiments` is also a pip-installable package that includes utilities that are leveraged by `rai-toolbox`'s tutorials and docs.
Install these via:

```console
$ pip install rai-experiments
```

## Running experiment notebooks

Follow the above installation instructions, then clone this repo and navigate to the navigate to the `experiments/` directory.

Some of the experiments contain Jupyter notebooks (`.ipynb` files). In order to open and run these, you must [install
either jupyterlab or the "classic" jupyter notebook](https://jupyter.org/install).


# For Maintainers

To publish a new version of `rai-experiments` do the following:

1. Update `__version__` in `experiments/src/rai_experiments/__init__.py`
2. Open a PR into the branch `rai-exps-publish`
3. Review and merge.


This will automatically trigger GitHub Actions, which will publish this new version to PyPI.

# Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

A portion of this research was sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

© 2023 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
