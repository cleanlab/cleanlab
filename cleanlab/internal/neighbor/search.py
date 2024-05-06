from __future__ import annotations
from typing import TYPE_CHECKING

from sklearn.neighbors import NearestNeighbors


if TYPE_CHECKING:

    from cleanlab.typing import Metric


def construct_knn(n_neighbors: int, metric: Metric, **knn_kwargs) -> NearestNeighbors:
    """
    Constructs a k-nearest neighbors search object. You can implement a similar method to run cleanlab with your own approximate-KNN library.

    Parameters
    ----------
    n_neighbors :
        The number of nearest neighbors to consider.
    metric :
        The distance metric to use for computing distances between points.
        See :py:mod:`~cleanlab.internal.neighbor.metric` for more information.
    **knn_kwargs:
        Additional keyword arguments to be passed to the search index constructor.
        See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html for more details on the available options.

    Returns:
        NearestNeighbors: A k-nearest neighbors search object.

    Note:
        The `metric` argument should be a callable that takes two arguments (the two points) and returns the distance between them.
        The additional keyword arguments (`**knn_kwargs`) are passed directly to the underlying k-nearest neighbors search algorithm.

    """
    sklearn_knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, **knn_kwargs)

    return sklearn_knn
