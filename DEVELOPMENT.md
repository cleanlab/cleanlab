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
work with changes you make to cleanlab by manually testing them.

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
  them to create a clickable link.
- **Math**: We support [LaTeX
  math](https://sphinxcontrib-katex.readthedocs.io/en/v0.8.6/examples.html)
  with the inline `` :math:`x+y` `` or the block:

  ```
  .. math::

     \sum_{0}^{n} 2n+1
  ```
