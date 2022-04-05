#!/usr/bin/env python

"""
Lint Jupyter notebooks being checked in to this repo.

Currently, this "linter" only checks one property, that the notebook's output
cells are empty, to avoid bloating the repository size.
"""


import argparse
import json
import os
import sys


def main():
    opts = get_opts()
    notebooks = find_notebooks(opts.dir)
    for notebook in notebooks:
        check(notebook)


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directories to search for notebooks", type=str, nargs="+")
    return parser.parse_args()


def find_notebooks(dirs):
    notebooks = set()
    for d in dirs:
        for dirname, _, filenames in os.walk(d):
            for filename in filenames:
                if not filename.endswith(".ipynb"):
                    continue
                full_path = os.path.join(dirname, filename)
                notebooks.add(full_path)
    return notebooks


def check(notebook):
    with open(notebook) as f:
        contents = json.load(f)
    check_outputs_empty(notebook, contents)
    check_no_trailing_newline(notebook, contents)


def check_outputs_empty(path, contents):
    for i, cell in enumerate(contents["cells"]):
        if "outputs" in cell and cell["outputs"] != []:
            fail(path, "output is not empty", i)


def check_no_trailing_newline(path, contents):
    """
    Checks that the last line of a code cell doesn't end with a newline, which
    produces an unnecessarily newline in the doc rendering.
    """
    for i, cell in enumerate(contents["cells"]):
        if cell["cell_type"] != "code":
            continue
        if "source" not in cell or len(cell["source"]) == 0:
            fail(path, "code cell is empty", i)
        if cell["source"][-1].endswith("\n"):
            fail(path, "unnecessary trailing newline", i)


def fail(path, message, cell=None):
    cell_msg = f" [cell {cell}]" if cell is not None else ""
    print(f"{path}{cell_msg}: {message}")
    sys.exit(1)


if __name__ == "__main__":
    main()
