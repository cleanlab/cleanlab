neighbor
========

The `neighbor` modules provide functionality for performing nearest neighbor search and pairwise distance calculations in those searches.

This submodule consists of the following modules:

- `neighbor.knn_graph`: Contains functions for setting up a nearest neighbor search index and constructing knn graphs.
- `neighbor.search`: Contains a helper function that wraps the default implementation of nearest neighbor searches.
- `neighbor.metric`: Contains functions for selecting distance metrics for nearest neighbor searches.

.. automodule:: cleanlab.internal.neighbor
   :autosummary:
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
    knn_graph
    metric
    search
