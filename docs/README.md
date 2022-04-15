# Building the Docs

Install `responsible-ai-toolbox` and install the docs-requirements:

```shell
/rai_toolbox/docs> pip install -r requirements.txt
```

Then build the docs:
 
```shell script
/rai_toolbox/docs> python -m sphinx source build
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