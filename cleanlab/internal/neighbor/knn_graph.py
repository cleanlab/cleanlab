from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from cleanlab.internal.neighbor.types import FeatureArray, Metric

from cleanlab.internal.neighbor.metric import decide_metric
from cleanlab.internal.neighbor.search import construct_knn


DEFAULT_K = 10
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
    metric = metric or decide_metric(features)

    # Decide the number of neighbors to use in the KNN search.
    n_neighbors = _configure_num_neighbors(features, n_neighbors)

    knn = construct_knn(n_neighbors, metric, **sklearn_knn_kwargs)
    return knn.fit(features)


def knn_to_knn_graph(knn: NearestNeighbors) -> csr_matrix:
    distances, indices = knn.kneighbors(return_distance=True)

    N, K = distances.shape

    # Pointers to the row elements distances[indptr[i]:indptr[i+1]],
    # and their corresponding column indices indices[indptr[i]:indptr[i+1]].
    indptr = np.arange(0, N * K + 1, K)

    return csr_matrix((distances.reshape(-1), indices.reshape(-1), indptr), shape=(N, N))


def correct_knn_distances_and_indices(
    features: FeatureArray, knn_graph: csr_matrix
) -> tuple[np.ndarray, np.ndarray]:
    N = features.shape[0]
    distances, indicies = knn_graph.data.reshape(N, -1), knn_graph.indices.reshape(N, -1)

    # Number of neighbors
    k = distances.shape[1]

    # Prepare the output arrays
    corrected_distances = np.zeros_like(distances)
    corrected_indices = np.zeros_like(indicies, dtype=int)

    # Use np.unique to catch inverse indices of all unique feature sets
    _, unique_inverse = np.unique(features, return_inverse=True, axis=0)

    # Map each unique feature set to its indices across the dataset
    feature_map = {u: np.where(unique_inverse == u)[0] for u in set(unique_inverse)}

    for i, (dists, inds) in enumerate(zip(distances, indicies)):
        # Find all indices where the points are the same as point i. This set is already sorted
        same_point_indices = feature_map[unique_inverse[i]]

        # Remove the current index i, 1D array of values in same_point_indices that are not in i.
        # The result of np.setdiff1d can be sorted with the argument assume_unique=True,
        # but that is always the case if same_point_indices is sorted.
        same_point_indices = np.setdiff1d(same_point_indices, i)

        # Determine the number of same points to include, respecting the limit of k
        num_same = len(same_point_indices)
        num_same_included = min(num_same, k)  # ensure we do not exceed k neighbors

        # Include same points in the results
        corrected_indices[i, :num_same_included] = same_point_indices[:num_same_included]
        # Distances are already zero for these points

        # Fill the rest of the slots with different points
        num_remaining_slots = k - num_same_included
        if num_remaining_slots > 0:
            # Get indices and distances from knn that are not the same as i
            different_point_mask = np.isin(inds, same_point_indices, invert=True)

            corrected_distances[i, num_same_included:k] = dists[different_point_mask][
                :num_remaining_slots
            ]
            corrected_indices[i, num_same_included:k] = inds[different_point_mask][
                :num_remaining_slots
            ]

    return corrected_distances, corrected_indices


def _configure_num_neighbors(features: FeatureArray, k: Optional[int]):
    if k is not None and k >= features.shape[0]:
        raise ValueError(
            f"Number of nearest neighbors k={k} cannot exceed the number of examples N={len(features)} passed into the estimator (knn)."
        )

    k = min(k or DEFAULT_K, features.shape[0] - 1)
    return k
