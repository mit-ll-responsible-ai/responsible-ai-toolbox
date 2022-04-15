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
