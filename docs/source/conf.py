# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import datetime
import shutil

sys.path.insert(0, os.path.abspath("../../cleanlab"))

# -- Project information -----------------------------------------------------

project = "cleanlab"
copyright = f"{datetime.datetime.now().year}, Cleanlab Inc."
author = "Cleanlab Inc."

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "autodocsumm",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx_tabs.tabs",
    "sphinx_multiversion",
    "sphinx_copybutton",
    "sphinxcontrib.katex",
]

numpy_show_class_members = True

# Don't add .txt suffix to source files:
html_sourcelink_suffix = ""

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

autosummary_generate = True

# -- Options for apidoc extension ----------------------------------------------

# apidoc_module_dir = "cleanlab/cleanlab"

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for Napoleon extension -------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for autodoc extension -------------------------------------------

# This value selects what content will be inserted into the main body of an autoclass
# directive
#
# http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#directive-autoclass
autoclass_content = "class"


# Default options to an ..autoXXX directive.
autodoc_default_options = {
    "autosummary": True,
    "members": None,
    "inherited-members": None,
    "show-inheritance": None,
    "special-members": "__call__",
}

# Subclasses should show parent classes docstrings if they don't override them.
autodoc_inherit_docstrings = True

# -- Options for katex extension -------------------------------------------

if os.getenv("CI") or shutil.which("katex") is not None:
    # requires that the machine have `katex` installed: `npm install -g katex`
    katex_prerender = True

# -- Variables Setting ---------------------------------------------------

# Determine doc site URL (DOCS_SITE_URL)
# Check if it's running in production repo
if os.getenv("GITHUB_REPOSITORY") == "cleanlab/cleanlab":
    DOCS_SITE_URL = "/"
else:
    DOCS_SITE_URL = "/cleanlab-docs/"

gh_env_file = os.getenv("GITHUB_ENV")
if gh_env_file is not None:
    with open(gh_env_file, "a") as f:
        f.write(f"\nDOCS_SITE_URL={DOCS_SITE_URL}")  # Set to Environment Var

GITHUB_REPOSITORY_OWNER = os.getenv("GITHUB_REPOSITORY_OWNER") or "cleanlab"
GITHUB_REF_NAME = os.getenv("GITHUB_REF_NAME") or "master"

# Pass additional variables to Jinja templates
html_context = {
    "DOCS_SITE_URL": DOCS_SITE_URL,
}

# -- nbsphinx Configuration ---------------------------------------------------

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = (
    """
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: html

    <style>
        .nbinput .prompt,
        .nboutput .prompt {
            display: none;
        }

        .output_area {
            max-height: 300px;
            overflow: auto;
        }

        .dataframe {
            background: #D7D7D7;
        }
    
        th {
            color:black;
        }
    </style>

    <script type="text/javascript">
        window.addEventListener('load', () => {
            const h1_element = document.getElementsByTagName("h1");
            h1_element[0].insertAdjacentHTML("afterend", `
            <p>
                <a style="background-color:white;color:black;padding:4px 12px;text-decoration:none;display:inline-block;border-radius:8px;box-shadow:0 2px 4px 0 rgba(0, 0, 0, 0.2), 0 3px 10px 0 rgba(0, 0, 0, 0.19)" href="https://colab.research.google.com/github/"""
    + GITHUB_REPOSITORY_OWNER
    + """/cleanlab-docs/blob/master/"""
    + GITHUB_REF_NAME
    + """/{{ docname|e }}" target="_blank">
                <img src="https://colab.research.google.com/img/colab_favicon_256px.png" alt="" style="width:40px;height:40px;vertical-align:middle">
                <span style="vertical-align:middle">Run in Google Colab</span>
                </a>
            </p>
            `);
        })

    </script>
"""
)

# Change this to "always" before running in the doc's CI/CD server
if os.getenv("CI"):
    nbsphinx_execute = "always"
if os.getenv("SKIP_NOTEBOOKS", "0") != "0":
    nbsphinx_execute = "never"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_favicon = "https://raw.githubusercontent.com/cleanlab/assets/a4483476d449f2f05a4c7cde329e72358099cc07/cleanlab/cleanlab_favicon.svg"
html_title = "cleanlab"
html_logo = (
    "https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/cleanlab_logo_only.png"
)
html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/cleanlab/cleanlab",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_sidebars = {
    "**": [
        "brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "versioning.html",
        "sidebar/scroll-end.html",
    ],
}
