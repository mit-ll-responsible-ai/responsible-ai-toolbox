# Building the Docs


## Deploying the docs to our website

To deploy the docs to our website, you must push to the `docs` branch. In GitHub create a pull request from `main` into `docs`.


## Building the docs locally
Navigate your console to the `responsible-ai-toolbox/docs/` directory. In your conda environment, presumed to be named `my_env` here, run the following to install doc dependencies:

```console
$ conda env update -n my_env --file conda_env.yml
```

In the same conda environment [install the toolbox](https://mit-ll-responsible-ai.github.io/responsible-ai-toolbox/#installation) 
and install the docs-requirements:

```console
$ pip install -r requirements.txt
```

Then build the docs:
 
```console
$ python -m sphinx source build
```

The resulting HTML files will be in `responsible-ai-toolbox/docs/build`.


To add docs for a new function/object, add that objects name under the rst-file's toc-tree and within the correct module, e.g.

```
.. currentmodule:: rai_toolbox.perturbations

.. autosummary::
   :toctree: generated/

   gradient_ascent
```

and then generate a documentation stub for that object. Change directories to `responsible-ai-toolbox/docs/source` and run:

```shell
sphinx-autogen -o generated/ *.rst
```

Now you can build the documentation as documented above, and that object's documentation will be included in that table of contents. (Note: you may need to delete the `builds/` directory before re-generating the HTML to see the change take effect)