# Development

This document explains how to set up a development environment for
[contributing](CONTRIBUTING.md) to cleanlab.

## Setting up a virtual environment

While this is not required, we recommend that you do development and testing in
a virtual environment. There are a number of tools to do this, including
[virtualenv](https://virtualenv.pypa.io/), [pipenv](https://pipenv.pypa.io/),
and [venv](https://docs.python.org/3/library/venv.html). You can
[compare](https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe)
the tools and choose what is right for you. Here, we'll explain how to get set
up with venv, which is built in to Python 3.

```shell
python3 -m venv ./ENV  # create a new virtual environment in the directory ENV
source ./ENV/bin/activate  # switch to using the virtual environment
```

You only need to create the virtual environment once, but you will need to
activate it every time you start a new shell. Once the virtual environment is
activated, the `pip install` commands below will install dependencies into the
virtual environment rather than your system Python installation.

## Installing dependencies and cleanlab

Run the following commands in the repository's root directory.

1. Install development requirements
```shell
pip install -r requirements-dev.txt
```

2. Install cleanlab as an editable package
```shell
pip install -e .
```

For Macs with Apple silicon: replace `tensorflow` in requirements-dev.txt with: `tensorflow-macos==2.9.2` and `tensorflow-metal==0.5.1`

### Handling optional dependencies

When designing a class that relies on an optional, domain-specific runtime dependency, it is better to use lazy-importing to avoid forcing users to install the dependency if they do not need it.

Depending on the coupling of your class to the dependency, you may want to consider importing it at the module-level or as an instance variable of the class or a function that uses the dependency.

If the dependency is used by many methods in the module or other classes, it is better to import it at the module-level.
On the other hand, if the dependency is only used by a handful of methods, then it's better to import it inside the method. If the dependency is not installed, an ImportError should be raised when the method is called, along with instructions on how to install the dependency.

Here is an example of a class that lazily imports CuPy and has a sum method (element-wise) that can be used on both CPU and GPU devices.

Unless an alternative implementations of the sum method is available, an `ImportError` should be raised when the method is called with instructions on how to install the dependency.

<details> <summary>Example code</summary>

```python
def lazy_import_cupy():
  try:
    import cupy
  except ImportError as error:
    # If the dependency is required for the class to work,
    # replace this block with a raised ImportError containing instructions
    print("Warning: cupy is not installed. Please install it with `pip install cupy`.")
    cupy = None
  return cupy

class Summation:
  def __init__(self):
    self.cupy = lazy_import_cupy()
  def sum(self, x) -> float:
    if self.cupy is None:
      return sum(x)
    return self.cupy.sum(x)
```
</details>


For the build system to recognize the optional dependency, you should add it to the `EXTRAS_REQUIRE` constant in **setup.py**:

<details> <summary>Example code</summary>

```python
EXTRAS_REQUIRE = {
    ...
    "gpu": [
      # Explain why the dependency below is needed,
      # e.g. "for performing summation on GPU"
      "cupy",
    ],
}
```


Or assign to a separate variable and add it to `EXTRAS_REQUIRE`

```python	
GPU_REQUIRES = [
  # Explanation ...
  "cupy",
]

EXTAS_REQUIRE = {
    ...
    "gpu": GPU_REQUIRES,
}
```
</details>


The package can be installed with the optional dependency (here called `gpu`) via:

1. PyPI installation

```shell
pip install -r "cleanlab[gpu]"
```

2. Editable installation

```shell
pip install -e ".[gpu]"
```

## Testing


**Download test data**
The test data for cleanlab resides in the assets repository. Use the following commands to download test data before running tests.
```shell
git clone https://github.com/cleanlab/assets.git
mv assets/cleanlab_test_data cleanlab/tests/datalab/data
```

**Run all the tests:**

```shell
pytest
```

**Run a specific file or test:**

```shell
pytest -k <filename or filter expression>
```

**Run with verbose output:**

```shell
pytest --verbose
```

**Run with code coverage:**

```shell
pytest --cov=cleanlab/ --cov-config .coveragerc --cov-report=html
```

The coverage report will be available in `coverage_html_report/index.html`,
which you can open with your web browser.

### Type checking

Cleanlab uses [mypy](https://mypy.readthedocs.io/en/stable/) typing. Type checking happens automatically during CI but can be run locally.

**Check typing in all files:**

```shell
mypy cleanlab
```

The above is just a simplified command for demonstration, do NOT run this for testing your own type annotations!
Our CI adds a few additional flags to the `mypy` command it uses in the file:
**.github/workflows/ci.yml**.
To exactly match the `mypy` command that is executed in CI, copy these flags, and also ensure your version of `mypy` and related packages like `pandas-stubs` match the latest released versions (used in our CI).

### Examples

You can check that the [examples](https://github.com/cleanlab/examples) still
work with changes you make to cleanlab by manually running the notebooks.
You can also run all example notebooks as follows:

```shell
git clone https://github.com/cleanlab/examples.git
```

Then specify your local version of cleanlab source in the first line of: **examples/requirements.txt**.
E.g. you can edit this line to point to your local version of cleanlab as a relative path such as `../cleanlab` if the `cleanlab` and `examples` repos are sibling directories on your computer.

Finally execute the bash script:

```shell
examples/run_all_notebooks.sh
```


## How to style new code contributions

cleanlab follows the [Black](https://black.readthedocs.io/) code style (see [pyproject.toml](pyproject.toml)). This is
enforced by CI, so please format your code by invoking `black` before submitting a pull request.

Generally aim to follow the [PEP-8 coding style](https://peps.python.org/pep-0008/).
Please do not use wildcard `import *` in any files, instead you should always import the specific functions that you need from a module.

All cleanlab code should have a maximum line length of 100 characters.

### Pre-commit hook

This repo uses the [pre-commit framework](https://pre-commit.com/) to easily
set up code style checks that run automatically whenever you make a commit.
You can install the git hook scripts with:

```shell
pre-commit install
```

### EditorConfig

This repo uses [EditorConfig](https://editorconfig.org/) to keep code style
consistent across editors and IDEs. You can install a plugin for your editor,
and then your editor will automatically ensure that indentation and line
endings match the project style.


## Adding new modules into the source code

  You should go through the following checklist if you intend to add new functionality to the package in a separate module.
- [x] Add brief description of the module’s purpose in a comment at the top of file and docstrings for every function.
- [x] Import the module `my_module.py` into main [``__init__.py``](cleanlab/__init__.py)
- [x] Create detailed unit tests (typically in a new file `tests/test_my_module.py`)
- [x] Add new module to docs index pages (**docs/source/index.rst**) and create .rst file in **docs/source/cleanlab/** (so that module appears on [docs.cleanlab.ai](https://docs.cleanlab.ai/stable/index.html) -- please verify its documentation also looks good there)
- [x] Create a QuickStart (**docs/source/tutorials**) notebook that runs main module functionality in 5min or less and add it to index pages (**docs/source/tutorials/index.rst**, **docs/source/index.rst**). Clear cell output before pushing.
- [x] Create an [examples](https://github.com/cleanlab/examples) notebook that runs more advanced module functionality with a more real-world application (can have a longer run time). Push with printed cell output.

## Contributing new issue types to Datalab

To contribute a new type of issue that Datalab can automatically detect in any dataset, refer to our guide on [Creating Your Own Issues Manager](https://docs.cleanlab.ai/master/cleanlab/datalab/guide/custom_issue_manager.html).

Do not add your new issue type to the set of issues that Datalab detects by default, our team can add it to this default set later on once it's utility has been thoroughly validated.

Don't forget to update the [issue type descriptions guide](https://github.com/cleanlab/cleanlab/blob/master/docs/source/cleanlab/datalab/guide/issue_type_description.rst) with a brief description of your new issue type.

Try to add tests for this new issue type. It's a good idea to start with some tests in a separate module in the [issue manager test directory](https://github.com/cleanlab/cleanlab/tree/master/tests/datalab/issue_manager). 

## Documentation

You can build the docs from your local cleanlab version by following [these
instructions](./docs/README.md#build-the-cleanlab-docs-locally).

If editing existing docs or adding new tutorials, please first read through our [guidelines](./docs/README.md#tips-for-editing-docstutorials).


## Documentation style

cleanlab uses [NumPy
style](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)
docstrings
([example](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html)).

Aspects that are not covered in the NumPy style or that are different from the
NumPy style are documented below:

- **Referring to the cleanlab package**: we refer to cleanlab without any
  special formatting, so no `cleanlab`, just cleanlab.
- **Cross-referencing**: when mentioning functions/classes/methods, always
  [cross-reference](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects)
  them to create a clickable link. Cross-referencing code from Jupyter
  notebooks is not currently supported.
- **Variable, module, function, and class names**: when not cross-references,
  should be written between single back-ticks, like `` `pred_probs` ``. Such
  names in Jupyter notebooks (Markdown) can be written between single
  back-ticks as well.
- **Math**: We support [LaTeX
  math](https://sphinxcontrib-katex.readthedocs.io/en/v0.8.6/examples.html)
  with the inline `` :math:`x+y` `` or the block:

  ```
  .. math::

     \sum_{0}^{n} 2n+1
  ```
- **Pseudocode vs math**: Prefer pseudocode in double backticks over LaTeX math.
- **Bold vs italics**: Use italics when defining a term, and use bold sparingly
  for extra emphasis.
- **Shapes**: Do not include shapes in the type of parameters, instead use
  `np.array` or `array_like` as the type and specify allowed shapes in the
  description. See, for example, the documentation for
  `cleanlab.classification.CleanLearning.fit()`. Format for 1D shape: `(N,1)`
- **Optional arguments**: for the most part, just put `, optional` in the type.
- **Type unions**: if a parameter or return type is something like "a numpy
  array or None", you can use "or" to separate types, e.g. `np.array or None`,
  and it'll be parsed correctly.
- **Parameterized types**: Use [standard Python type
  hints](https://docs.python.org/3/library/typing.html) for referring to
  parameters and parameterized types in docs, e.g. `Iterable[int]` or
  `list[float]`.

### Common variable names / terminology used throughout codebase

- `N` - the number of examples/datapoints in a dataset.
  - `num_examples` may also be used when additional clarity is needed.
- `K` - the number of classes (unique labels) for a dataset.
  - `num_classes` may also be used when additional clarity is needed.
- `labels` - a label for each example, length should be N (sample-size of dataset)
- `classes` - set of possible labels for any one example, length should be K (number of possible categories in classification problem)

Try to adhere to this standardized terminology unless you have good reason not to!

### Relative Link Formatting Instructions

Use relative linking to connect information between docs and jupyter notebooks, and make sure links will remain valid in the future as new cleanlab versions are released! Sphinx/html works with relative paths so try to specify relative paths if necessary. For specific situations:

- Link another function or class from within a source code docstring: 
  - If you just want to specify the function/class name (ie. the function/class is unique throughout our library): `` `~cleanlab.file.function_or_class_name` ``. 
  
    This uses the [Sphinx's](https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-default_role) `default_role = "py:obj"` setting, so the leading tilde shortens the link to only display `function_or_class_name`.
  - If you want to additionally specify the module which the function belongs to: 
      - `` :py:func:`file.function_name <cleanlab.file.function_name>` `` for functions 
      - ``:py:class:`file.class_name <cleanlab.file.class_name>` `` for classes

    Here you have more control over the text that is displayed to display the module name.  When referring to a function that is alternatively defined in other modules as well, always use this option to be more explicit about which module you are referencing.
- Link a tutorial (rst file) from within a source code docstring or rst file: ``:ref:`tutorial_name <tutorial_name>` ``
- Link a tutorial notebook (ipynb file) from within a source code docstring or rst file: `` `notebook_name <tutorials/notebook_name.ipynb>`_ `` . (If the notebook is not the in the same folder as the source code, use a relative path)
- Link a function from within a tutorial notebook: `[function_name](../cleanlab/file.html#cleanlab.file.function_name)`

  Links from master branch tutorials will reference master branch functions, similarly links from tutorials in stable branch will reference stable branch functions since we are using relative paths.
- Link a specific section of a notebook from within the notebook: `[section title](#section-title)`
- Link a different tutorial notebook from within a tutorial notebook: `[another notebook](another_notebook.html)`. (Note this only works when the other notebook is in same folder as this notebook, otherwise may need to try relative path)
- Link another specific section of different notebook from within a tutorial notebook: `[another notebook section title](another_notebook.html#another-notebook-section-title)`
- Linking examples notebooks from inside tutorial notebooks can be simply done by linking global url of the example notebook in master branch of github.com/cleanlab/examples/
