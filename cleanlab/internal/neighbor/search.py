from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, Tuple, cast

from sklearn.neighbors import NearestNeighbors


if TYPE_CHECKING:
    import numpy as np
    from scipy.sparse import csr_matrix

    from cleanlab.internal.neighbor.types import Metric
    from cleanlab.internal.neighbor.types import FeatureArray


class NeighborSearch(Protocol):
    def fit(self, X: FeatureArray) -> "NeighborSearch": ...

    def kneighbors(self, X: FeatureArray) -> Tuple[np.ndarray, np.ndarray]: ...

    def kneighbors_graph(self, X: FeatureArray) -> csr_matrix: ...


def construct_knn(n_neighbors: int, metric: Metric, **knn_kwargs) -> NeighborSearch:
    """
    Constructs a k-nearest neighbors search object.

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
        NeighborSearch: A k-nearest neighbors search object.

    Note:
        The `metric` argument should be a callable that takes two arguments (the two points) and returns the distance between them.
        The additional keyword arguments (`**knn_kwargs`) are passed directly to the underlying k-nearest neighbors search algorithm.

    """
    sklearn_knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, **knn_kwargs)
    # Trust me, I know sklearn_knn acts like a NeighborSearch.
    # NearestNeighbors.fit(X, y) ignores y, but keeps it for API consistency by convention.
    return cast(NeighborSearch, sklearn_knn)
