from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from cleanlab.typing import FeatureArray, Metric

from cleanlab.internal.neighbor.metric import decide_default_metric
from cleanlab.internal.neighbor.search import construct_knn


DEFAULT_K = 10  # Value is set for issue type that requires the largest number of neighbors. Most of the issue types require 10 neighbors by default.
"""Default number of neighbors to consider in the k-nearest neighbors search,
unless the size of the feature array is too small or the user specifies a different value.

This should be the largest desired value of k for all desired issue types that require a KNN graph.

E.g. if near duplicates wants k=1 but outliers wants 10, then DEFAULT_K should be 10.
"""


def features_to_knn(
    features: Optional[FeatureArray],
    *,
    n_neighbors: Optional[int] = None,
    metric: Optional[Metric] = None,
    **sklearn_knn_kwargs,
) -> NearestNeighbors:
    """Build and fit a k-nearest neighbors search object from an array of numerical features.

    Parameters
    ----------
    features :
        The input feature array, with shape (N, M), where N is the number of samples and M is the number of features.
    n_neighbors :
        The number of nearest neighbors to consider. If None, a default value is determined based on the feature array size.
    metric :
        The distance metric to use for computing distances between points. If None, the metric is determined based on the feature array shape.
    **sklearn_knn_kwargs :
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
    if features is None:
        raise ValueError("Both knn and features arguments cannot be None at the same time.")
    # Use provided metric if available, otherwise decide based on the features.
    metric = metric or decide_default_metric(features)

    # Decide the number of neighbors to use in the KNN search.
    n_neighbors = _configure_num_neighbors(features, n_neighbors)

    knn = construct_knn(n_neighbors, metric, **sklearn_knn_kwargs)
    return knn.fit(features)


def construct_knn_graph_from_index(knn: NearestNeighbors) -> csr_matrix:
    """Construct a KNN graph from a fitted NearestNeighbors search object.

    Parameters
    ----------
    knn :
        A NearestNeighbors object that has been fitted to a feature array.
        The knn graph is constructed based on the distances and indices of each feature row's nearest neighbors.

    Returns
    -------
    knn_graph :
        A sparse, weighted adjacency matrix representing the KNN graph of the feature array.

    Note
    ----
    This is *not* intended to construct a KNN graph of test data. It is only used to construct a KNN graph of the data used to fit the NearestNeighbors object.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.internal.neighbor.neighbor import features_to_knn, construct_knn_graph_from_index
    >>> features = np.array([
        [0.701, 0.701],
        [0.900, 0.436],
        [0.000, 1.000],
    ])
    >>> knn = features_to_knn(features, n_neighbors=1)
    >>> knn_graph = construct_knn_graph_from_index(knn)
    >>> knn_graph.toarray()  # For demonstration purposes only. It is generally a bad idea to transform to dense matrix for large graphs.
    array([[0.        , 0.33140006, 0.        ],
           [0.33140006, 0.        , 0.        ],
           [0.76210367, 0.        , 0.        ]])
    """

    distances, indices = knn.kneighbors(return_distance=True)

    N, K = distances.shape

    # Pointers to the row elements distances[indptr[i]:indptr[i+1]],
    # and their corresponding column indices indices[indptr[i]:indptr[i+1]].
    indptr = np.arange(0, N * K + 1, K)

    return csr_matrix((distances.reshape(-1), indices.reshape(-1), indptr), shape=(N, N))


def construct_knn_graph_from_features(
    features: Optional[FeatureArray],
    *,
    n_neighbors: Optional[int] = None,
    metric: Optional[Metric] = None,
    **sklearn_knn_kwargs,
) -> csr_matrix:
    """Calculate the KNN graph from the features if it is not provided in the kwargs.

    Parameters
    ----------
    features :
        The input feature array, with shape (N, M), where N is the number of samples and M is the number of features.
    n_neighbors :
        The number of nearest neighbors to consider. If None, a default value is determined based on the feature array size.
    metric :
        The distance metric to use for computing distances between points. If None, the metric is determined based on the feature array shape.
    **sklearn_knn_kwargs :
        Additional keyword arguments to be passed to the search index constructor.

    Returns
    -------
    knn_graph :
        A sparse, weighted adjacency matrix representing the KNN graph of the feature array.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.internal.neighbor.neighbor import construct_knn_graph_from_features
    >>> features = np.array([
        [0.701, 0.701],
        [0.900, 0.436],
        [0.000, 1.000],
    ])
    >>> knn_graph = construct_knn_graph_from_features(features, n_neighbors=1)
    >>> knn_graph.toarray()  # For demonstration purposes only. It is generally a bad idea to transform to dense matrix for large graphs.
    array([[0.        , 0.33140006, 0.        ],
           [0.33140006, 0.        , 0.        ],
           [0.76210367, 0.        , 0.        ]])
    """
    # Construct NearestNeighbors object
    knn = features_to_knn(features, n_neighbors=n_neighbors, metric=metric, **sklearn_knn_kwargs)
    # Build graph from NearestNeighbors object
    return construct_knn_graph_from_index(knn)


def _configure_num_neighbors(features: FeatureArray, k: Optional[int]):
    # Error if the provided value is greater or equal to the number of examples.
    N = features.shape[0]
    if k is not None and k >= N:
        raise ValueError(
            f"Number of nearest neighbors k={k} cannot exceed the number of examples N={len(features)} passed into the estimator (knn)."
        )

    # Either use the provided value or select a default value based on the feature array size.
    k = k or min(DEFAULT_K, N - 1)
    return k
