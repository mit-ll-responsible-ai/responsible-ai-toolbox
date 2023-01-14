Thanks for your interest in contributing to LL's robustness-toolkit. Please read 
through the following resources before you begin working on any contributions to this 
code base.


- [Contributor Basics](#contributor-basics)
  - [Installing rAI toolkit for development](#installing-rai-toolkit-for-development)
  - [Pre-Commit Hooks (Required)](#pre-commit-hooks-required)
    - [What does this do?](#what-does-this-do)
  - [Running Tests](#running-tests)
- [Code Quality](#code-quality)
  - [PEP 8](#pep-8)
  - [Naming Conventions](#naming-conventions)
  - [Type-Hints](#type-hints)
  - [Documentation Strings](#documentation-strings)
  - [Using Descriptive Data Structures](#using-descriptive-data-structures)
- [Validating Type Correctness](#validating-type-correctness)
- [Maintaining the rai-toolbox](#maintaining-the-rai-toolbox)
  - [CI/CD Overview](#cicd-overview)
  - [Project dependencies, metadata, and versioning](#project-dependencies-metadata-and-versioning)
  - [Tooling configuration](#tooling-configuration)
  - [Creating a new release and publishing to PyPI](#creating-a-new-release-and-publishing-to-pypi)
  
## Contributor Basics
### Installing rAI toolkit for development

Install the toolkit along with its test dependencies; checkout the repo, navigate to its top level and run

```shell
pip install -e .[tests]
```

the `-e` option ensures that any changes that you make to the project's source code will be reflected in your local install – you need not reinstall the package in order for your modifications to take effect.

### Pre-Commit Hooks (Required)

We provide contributors with pre-commit hooks, which will apply auto-formatters and 
linters to your code before your commit takes effect. You must install these in order to contribute to the repo.

First install pre-commit in your Python environment. Run:

```console
pip install pre-commit
```

Then, in the top-level of the `rai_toolbox` repo, run:

```console
pre-commit install
pre-commit run
```

Great! You can read more about pre-commit hooks in general here: https://pre-commit.com/

#### What does this do?

Our pre-commit hooks run the following auto-formatters on all commits:
- [black](https://black.readthedocs.io/en/stable/)
- [isort](https://pycqa.github.io/isort/)

It also runs [flake8](https://github.com/PyCQA/flake8) to enforce PEP8 standards.


### Running Tests

The most basic use case of `pytest` is trivial: it will look for files with the word "test" in their name, and will look for functions that also have "test" in their name, and it will simply run those functions.

Navigate to the top-level of `rai_toolbox` and run:

```console
pytest tests/
```

If you want to quickly run through the test suite, just to verify that it runs without error, you can run:


```console
pytest tests/ --hypothesis-profile fast
```

Additional Resources to Learn About Our Approach to Automated Testing, see: https://github.com/rsokl/testing-tutorial

## Code Quality
A more thorough discussion of the following items can be found in [module 5 of Python Like You Mean It](https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/Writing_Good_Code.html).

### PEP 8
Our code should adhere to the PEP 8 Style Guide. A general, brief overview of this style guide can be found [here](https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/Writing_Good_Code.html#The-PEP8-Style-Guide-to-Python-Code). [This is a link to the formal PEP8 specification](https://www.python.org/dev/peps/pep-0008/#code-lay-out).

```python
# Example of PEP-8 enforcement

# Adheres to PEP8:
x = {1: "a", 2: "b", 3: "c"}

# Violates PEP8 (excess whitespace):
x = {1 : "a", 2 : "b", 3 : "c"}
```

### Naming Conventions
The basic [naming conventions for Python](https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/Writing_Good_Code.html#Naming-Conventions) are quite simple. They are:
- Class names: `CamelCase`
- Constants: `ALL_CAPS`
- Literally everything else: `snake_case`

For example:

```python
class ClassName:
    pass


def function_name():
    pass


variable_name = [1, 2, 3]

CONSTANT = 42
```

Note that Python package and module names should also follow `snake_case` conventions:

```
my_project/
  | setup.py
  | src/
     | my_project/
        | __init__.py
        | module_name.py
        | package_name/
           | __init__.py
           | sub_module_name.py
  | tests/
```

### Type-Hints

It is recommended that large-scale projects and projects that make heavy use of custom classes consider incorporating [function and variable annotations with type-hints](https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/Writing_Good_Code.html#Type-Hinting). These greatly augment an IDE's ability to enrich your development environment with typing information. This serves to highlight bugs and inconsistencies as you are writing your code, rather than at runtime.

```python
# Example of type hinting


def count_vowels(x: str, include_y: bool = False) -> int:
    """Returns the number of vowels contained in `in_string`"""
    vowels = set("aeiouAEIOU")
    if include_y:
        vowels.update("yY")
    return sum(1 for char in x if char in vowels)
```

### Documentation Strings

Documentation strings should adhere to the [NumPy Documentation Style](https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/Writing_Good_Code.html#Documentation-Styles). Beyond the occasional trivial, self-evident function, every function should have a doc-string. This includes functions that are strictly internal to the codebase.

```python
# Example of of a numpy-style docstring
def compute_student_stats(grade_book, stat_function, student_list=None):
    """Computes custom statistics over students' grades.

    Applies ``stat_func`` over a list of each student's grades,
    and accumulates name-stat tuple pairs in a list.

    Parameters
    ----------
    grade_book : Dict[str, List[float]]
        The dictionary (name -> grades) of all of the students'
        grades.

    stat_function: Callable[[Iterable[float]], Any]
        The function used to compute statistics over each student's
        grades.

    student_list : Optional[List[str]]
        A list of names of the students for whom statistics will be
        computed. By default, statistics will be computed for all
        students in the grade book.

    Returns
    -------
    List[Tuple[str, Any]]
        The name-stats tuple pair for each specified student.

    Examples
    --------
    >>> from statistics import mean
    >>> grade_book = dict(Bruce=[90., 82., 92.], Courtney=[100., 85., 78.])
    >>> compute_student_stats(grade_book, stat_function=mean)
    [('Bruce', 88.0), ('Courtney', 87.66666666666667)]
    """
```

### Using Descriptive Data Structures

Strive to leverage data structures with explicitly-named fields that describe your data. For example, if you are working with geographic coordinates in degrees-minutes-seconds, one might be tempted to store these coordinates in a plain tuple:

```python
# bad: using a tuple to store a degrees-minutes-seconds coordinate
>>> coord = (80, 42, 30)  # coordinate in DMS
```

The shortcoming of this is that the meaning of these fields are not self-evident. This data requires an additional source of documentation for context or, worse, relies implicitly on a developer's recollection. 

One can instead define a [named-tuple](https://www.pythonlikeyoumeanit.com/Module2_EssentialsOfPython/DataStructures_III_Sets_and_More.html#Named-Tuple) that explicitly represents the data being handled. 

```python
# good: Using a namedtuple to store a degrees-minutes-seconds coordinate
#       with explicit attribute names
>>> from collections import namedtuple
>>> DMS = namedtuple('DMS', ['degrees', 'minutes', 'seconds'])
>>> coord = DMS(degrees=80, minutes=42, seconds=30)

>>> coord.degrees  # fetch a field by name
80

>>> coord[0]  # fetch a field by index
80
```

Named tuples can even store docstrings to provide the user with additional information:

```python
# setting a docstring on a named-tuple
>>> DMS.__doc__ = """
Stores a number in degrees/minutes/seconds.

Parameters
----------
degrees : int
minutes : int ∈ [0, 60]
seconds : int ∈ [0, 60]
"""
```
If you are using Python 3.7 or later, you can also use the new [dataclass](https://docs.python.org/3/library/dataclasses.html) to a similar effect. 

For users performing larger-scale numerical experiments, numpy-arrays fall victim to the same ambiguity as tuples. The [`pandas`](https://pandas.pydata.org/) and [`xarray`](http://xarray.pydata.org/en/stable/) libraries should be used to store tabular and higher-dimensional arrays of data with explicitly-named fields and metadata.

```python
# bad: Using comments to describe heterogeneous data
>>> import numpy as np
>>> data = np.array([[.25, .01],  # row: time-of-day; col: sensor
...                  [1.58, .82],
...                  [1.01, .91]])

# good: Using an xarray to explicitly name dimensions and label coordinates
#       for a heterogenous array of data
>>> dims = ["data_collect", "sensor_type"]
>>> coords = {"data_collect": ["Morning", "Noon", "Night"],
...           "sensor_type": ["EO", "IR"]}
>>> xr.DataArray(data, dims=dims, coords=coords)
<xarray.DataArray (data_collect: 3, sensor_type: 2)>
array([[0.25, 0.01],
       [1.58, 0.82],
       [1.01, 0.91]])
Coordinates:
  * data_collect  (data_collect) <U7 'Morning' 'Noon' 'Night'
  * sensor_type   (sensor_type) <U2 'EO' 'IR'
  ```
Using explicit data structures when processing data is critical to ensuring that an algorithm can have a long shelf-life and will be useful to people other than the algorithm's author(s).


## Validating Type Correctness

Our CI runs the `pyright` type-checker in basic mode against hydra-zen's entire code base and against specific test files; this ensures that our type-annotations are complete and accurate.

If you use VSCode with Pylance, then make sure that `Type Checking Mode` is set to `basic` for your hydra-zen workspace. Your IDE will then mark any problematic code.Other IDEs can leverage the pyright language server to a similar effect. 

While this is helpful for getting immediate feedback about your code, it is no substitute for running `pyright` from the commandline. To do so, run the following tox job:

```console
tox -e pyright
```

## Maintaining the rai-toolbox

The following lays out the essentials for maintaining the rai-toolbox. It is recommended that you read the previous "Contributor Basics" before proceeding.

### CI/CD Overview

We use GitHub Actions for our CI/CD needs. It is responsible for:
- Running our test suite against multiple platforms, Python versions, and dependency matrices
- Scanning the project for consistent and up-to-date headers
- Running a spell checker on our docs and code base
- Running the pyright type checker on our code base, tests, and docs
- Publishing new build artifacts to PyPI
- Building and publishing our docs
- Managing our pinned dependencies via dependabot

As described above, We use `tox` to normalize the Python environment creation and command running for the majority of these tasks, so that they can be run locally as well.

### Project dependencies, metadata, and versioning

The project's build tooling (e.g. that we use setuptools to builds the installable artifacts), metadata (e.g. author list), and dependencies are all specified in the [pyproject.toml](https://github.com/mit-ll-responsible-ai/responsible-ai-toolbox/blob/main/pyproject.toml) file.

The `project > dependencies` section is where the project's minimum dependencies are specified.
In the case that a new dependency is added, a minimum version must be specified. `project > optional-dependencies` includes the dependencies that are needed to run our test suite, dependencies for the `mushin` and `datasets` submodules.

The project's version (e.g. `v0.3.0`) is managed by [setuptools_scm](https://github.com/pypa/setuptools_scm), meaning that the `rai_toolbox.__version__` attribute is not set manually, rather it is derived from the project's latest git-commit tag of the form `vX.Y.Z`. See [Creating a new release and publishing to PyPI](#creating-a-new-release-and-publishing-to-pypi) for more details.


### Tooling configuration

The repo's [pyproject.toml](https://github.com/mit-ll-responsible-ai/responsible-ai-toolbox/blob/main/pyproject.toml) file is responsible for storing the configurations for the project's tools (e.g. isort, pyright, codespell) whenever possible. Some tools, such as flake8 and pre-commit, do not support this file format and have separate config files.


Our pinned pyright dependency is managed by [dependabot](https://docs.github.com/en/code-security/dependabot/dependabot-version-updates) in 

### Creating a new release and publishing to PyPI

We use GitHub Actions to automate the process of publishing to the Python Package Index. To publish a new release, simply create a new tag to reflect the new version `v<major>.<minor>.<patch>[rc<iter>]` (e.g. `v1.0.2` or `v1.0.2rc3` -- for the 3rd release candidate of `v1.0.2`). Then, in the repo, create a new release using that tag. This will trigger the action for publishing to PyPI; within minutes users will be able to install the latest version using pip!
