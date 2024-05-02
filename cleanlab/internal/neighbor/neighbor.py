from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cleanlab.internal.neighbor.types import FeatureArray, Metric
    from cleanlab.internal.neighbor.search import NeighborSearch

from cleanlab.internal.neighbor.metric import decide_metric
from cleanlab.internal.neighbor.search import construct_knn


DEFAULT_K = 10
"""Default number of neighbors to consider in the k-nearest neighbors search,
unless the size of the feature array is too small or the user specifies a different value.
"""


def features_to_knn(
    features: FeatureArray,
    *,
    n_neighbors: Optional[int] = None,
    metric: Optional[Metric] = None,
    **knn_kwargs,
) -> NeighborSearch:
    """Build and fit a k-nearest neighbors search object from an array of numerical features.

    Parameters
    ----------
    features :
        The input feature array, with shape (N, M), where N is the number of samples and M is the number of features.
    n_neighbors :
        The number of nearest neighbors to consider. If None, a default value is determined based on the feature array size.
    metric :
        The distance metric to use for computing distances between points. If None, the metric is determined based on the feature array shape.
    **knn_kwargs :
        Additional keyword arguments to be passed to the search index constructor.

    Returns
    -------
    knn :
        A k-nearest neighbors search object fitted to the input feature array.

    Examples
    --------

    >>> import numpy as np
    >>> from cleanlab.internal.neighbor import features_to_knn
    >>> features = np.random.rand(100, 10)
    >>> knn = features_to_knn(features)
    >>> knn
    NearestNeighbors(metric='cosine', n_neighbors=10)
    """
    # Use provided metric if available, otherwise decide based on the features.
    metric = metric or decide_metric(features)

    # Decide the number of neighbors to use in the KNN search.
    n_neighbors = _configure_num_neighbors(features, n_neighbors)

    knn = construct_knn(n_neighbors, metric, **knn_kwargs)
    return knn.fit(features)


def _configure_num_neighbors(features: FeatureArray, k: Optional[int]):
    if k is not None and k >= features.shape[0]:
        raise ValueError(
            f"Number of neighbors k={k} must be less than the number of samples {features.shape[0]}."
        )

    k = min(k or DEFAULT_K, features.shape[0] - 1)
    return k
