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

```console
$ python3 -m venv ./ENV  # create a new virtual environment in the directory ENV
$ source ./ENV/bin/activate  # switch to using the virtual environment
```

You only need to create the virtual environment once, but you will need to
activate it every time you start a new shell. Once the virtual environment is
activated, the `pip install` commands below will install dependencies into the
virtual environment rather than your system Python installation.

## Installing dependencies and cleanlab

Run the following commands in the repository's root directory.

1. Install development requirements with `pip install -r requirements-dev.txt`

1. Install cleanlab as an editable package with `pip install -e .`

## Testing

**Run all the tests:**

```console
$ pytest
```

**Run a specific file or test:**

```
$ pytest -k <filename or filter expression>
```

**Run with verbose output:**

```
$ pytest --verbose
```

**Run with code coverage:**

```
$ pytest --cov=cleanlab/ --cov-config .coveragerc --cov-report=html
```

The coverage report will be available in `coverage_html_report/index.html`,
which you can open with your web browser.

### Examples

You can check that the [examples](https://github.com/cleanlab/examples) still
work with changes you make to cleanlab by manually running the notebooks.
You can also run all example notebooks as follows:

```console
git clone https://github.com/cleanlab/examples.git
```

Then specify your local version of cleanlab source in the first line of: **examples/requirements.txt**.
E.g. you can edit this line to point to your local version of cleanlab as a relative path such as `../cleanlab` if the `cleanlab` and `examples` repos are sibling directories on your computer.

Finally execute the bash script:

```console
examples/run_all_notebooks.sh
```

## Documentation

You can build the docs from your local cleanlab version by following [these
instructions](docs/README.md#build-the-cleanlab-docs-locally).

## Code style

cleanlab follows the [Black](https://black.readthedocs.io/) code style. This is
enforced by CI, so please format your code before submitting a pull request.

### Pre-commit hook

This repo uses the [pre-commit framework](https://pre-commit.com/) to easily
set up code style checks that run automatically whenever you make a commit.
You can install the git hook scripts with:

```console
$ pre-commit install
```

### EditorConfig

This repo uses [EditorConfig](https://editorconfig.org/) to keep code style
consistent across editors and IDEs. You can install a plugin for your editor,
and then your editor will automatically ensure that indentation and line
endings match the project style.

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
- **Pseudocode vs math**: prefer pseudocode in double backticks over LaTeX math.
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

### Common variable names used in documentation

- `N` - the number of examples/datapoints in a dataset.
  - `num_examples` may also be used when additional clarity is needed.
- `K` - the number of classes (unique labels) for a dataset.
  - `num_classes` may also be used when additional clarity is needed.
