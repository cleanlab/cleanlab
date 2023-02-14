datalab
=======

.. warning::
    Methods in this ``datalab`` module are bleeding edge and may have sharp edges. They are not guaranteed to be stable between different ``cleanlab`` versions.

.. automodule:: cleanlab.experimental.datalab
   :autosummary:
   :members:
   :undoc-members:
   :show-inheritance:

Getting Started
---------------

This package has additional dependencies that are not required for the core ``cleanlab`` package. To install them, run:

.. code-block:: console

    $ pip install cleanlab[datalab]

For the developmental version of the package, install from source:

.. code-block:: console

    $ pip install git+https://github.com/cleanlab/cleanlab.git#egg=cleanlab[datalab]

Guides
------

.. toctree::
    :maxdepth: 2

    guide/index


API Reference
-------------

.. toctree::
    :maxdepth: 2

    datalab
    data
    factory
    data_issues
    issue_manager/index
