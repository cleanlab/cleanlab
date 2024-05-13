from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Tuple
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from cleanlab.typing import FeatureArray, Metric

from cleanlab.internal.neighbor.metric import decide_default_metric
from cleanlab.internal.neighbor.search import construct_knn


DEFAULT_K = 10
"""Default number of neighbors to consider in the k-nearest neighbors search,
unless the size of the feature array is too small or the user specifies a different value.

This should be the largest desired value of k for all desired issue types that require a KNN graph.

E.g. if near duplicates wants k=1 but outliers wants 10, then DEFAULT_K should be 10. This way, all issue types can rely on the same KNN graph.
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
    """Construct a sparse distance matrix representation of KNN graph out of a fitted NearestNeighbors search object.

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
    >>> from cleanlab.internal.neighbor.knn_graph import features_to_knn, construct_knn_graph_from_index
    >>> features = np.array([
    ...    [0.701, 0.701],
    ...    [0.900, 0.436],
    ...    [0.000, 1.000],
    ... ])
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


def create_knn_graph_and_index(
    features: Optional[FeatureArray],
    *,
    n_neighbors: Optional[int] = None,
    metric: Optional[Metric] = None,
    correct_exact_duplicates: bool = True,
    **sklearn_knn_kwargs,
) -> Tuple[csr_matrix, NearestNeighbors]:
    """Calculate the KNN graph from the features if it is not provided in the kwargs.

    Parameters
    ----------
    features :
        The input feature array, with shape (N, M), where N is the number of samples and M is the number of features.
    n_neighbors :
        The number of nearest neighbors to consider. If None, a default value is determined based on the feature array size.
    metric :
        The distance metric to use for computing distances between points. If None, the metric is determined based on the feature array shape.
    correct_exact_duplicates :
        Whether to correct the KNN graph to ensure that exact duplicates have zero mutual distance, and they are correctly included in the KNN graph.
    **sklearn_knn_kwargs :
        Additional keyword arguments to be passed to the search index constructor.

    Returns
    -------
    knn_graph :
        A sparse, weighted adjacency matrix representing the KNN graph of the feature array.
    knn :
        A k-nearest neighbors search object fitted to the input feature array. This object can be used to query the nearest neighbors of new data points.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.internal.neighbor.knn_graph import create_knn_graph_and_index
    >>> features = np.array([
    ...    [0.701, 0.701],
    ...    [0.900, 0.436],
    ...    [0.000, 1.000],
    ... ])
    >>> knn_graph, knn = create_knn_graph_and_index(features, n_neighbors=1)
    >>> knn_graph.toarray()  # For demonstration purposes only. It is generally a bad idea to transform to dense matrix for large graphs.
    array([[0.        , 0.33140006, 0.        ],
           [0.33140006, 0.        , 0.        ],
           [0.76210367, 0.        , 0.        ]])
    >>> knn
    NearestNeighbors(metric=<function euclidean at ...>, n_neighbors=1)  # For demonstration purposes only. The actual metric may vary.
    """
    # Construct NearestNeighbors object
    knn = features_to_knn(features, n_neighbors=n_neighbors, metric=metric, **sklearn_knn_kwargs)
    # Build graph from NearestNeighbors object
    knn_graph = construct_knn_graph_from_index(knn)

    # Ensure that exact duplicates found with np.unique aren't accidentally missed in the KNN graph
    if correct_exact_duplicates:
        assert features is not None
        knn_graph = correct_knn_graph(features, knn_graph)
    return knn_graph, knn


def correct_knn_graph(features: FeatureArray, knn_graph: csr_matrix) -> csr_matrix:
    N = features.shape[0]
    distances, indices = knn_graph.data.reshape(N, -1), knn_graph.indices.reshape(N, -1)

    corrected_distances, corrected_indices = correct_knn_distances_and_indices(
        features, distances, indices
    )
    N = features.shape[0]
    return csr_matrix(
        (corrected_distances.reshape(-1), corrected_indices.reshape(-1), knn_graph.indptr),
        shape=(N, N),
    )


def correct_knn_distances_and_indices(
    features: FeatureArray, distances: np.ndarray, indices: np.ndarray, enable_warning: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Corrects the distances and indices of a k-nearest neighbors (knn) graph
    based on all exact duplicates detected in the feature array.

    Parameters
    ----------
    features :
        The feature array used to construct the knn graph.
    distances :
        The distances between each point and its k nearest neighbors.
    indices :
        The indices of the k nearest neighbors for each point.
    enable_warning :
        Whether to enable warning messages if any row underestimates the number of exact duplicates.

    Returns
    -------
    corrected_distances :
        The corrected distances between each point and its k nearest neighbors. Exact duplicates (based on the feature array) are ensured to have zero mutual distance.
    corrected_indices :
        The corrected indices of the k nearest neighbors for each point. Exact duplicates are ensured to be included in the k nearest neighbors, unless the number of exact duplicates exceeds k.

    Raises
    ------
    UserWarning :
        A warning may be raised if there were some slots available for an exact duplicate that were missed.
        This may happen if the number of exact duplicates in the existing knn graph is lower than k,
        but the set of exact duplicates is larger than what was included in the knn graph.
        This warning may be disabled by setting enable_warning=False.


    Example
    -------
    >>> import numpy as np
    >>> X = np.array(
    ...     [
    ...         [0, 0],
    ...         [0, 0], # Exact duplicate of the previous point
    ...         [1, 1], # The distances between this point and the others is sqrt(2) (equally distant from both)
    ...     ]
    ... )
    >>> distances = np.array(  # Distance to the 1-NN of each point
    ...     [
    ...         [np.sqrt(2)],  # Should be [0]
    ...         [1e-16],       # Should be [0]
    ...         [np.sqrt(2)],
    ...     ]
    ... )
    >>> indices = np.array(  # Index of the 1-NN of each point
    ...     [
    ...         [2],  # Should be [1]
    ...         [0],
    ...         [1],  # Might be [0] or [1]
    ...     ]
    ... )
    >>> corrected_distances, corrected_indices = correct_knn_distances_and_indices(X, distances, indices)
    >>> corrected_distances
    array([[0.], [0.], [1.41421356]])
    >>> corrected_indices
    array([[1], [0], [0]])


    Clearly, the first point misses its exact duplicate in the knn graph. To raise a warning for such cases, set enable_warning=True.

    >>> corrected_distances, corrected_indices = correct_knn_distances_and_indices(X, distances, indices, enable_warning=True)
    UserWarning: There were some slots available for an exact duplicate that were missed.
    """

    # Number of neighbors
    k = distances.shape[1]

    # Prepare the output arrays
    corrected_distances = np.zeros_like(distances)
    corrected_indices = np.zeros_like(indices, dtype=int)

    corrected_distances = np.copy(distances)
    corrected_indices = np.copy(indices)

    # Use np.unique to catch inverse indices of all unique feature sets
    _, unique_inverse, unique_counts = np.unique(
        features, return_inverse=True, return_counts=True, axis=0
    )

    # Collect different sets of exact duplicates in the dataset
    exact_duplicate_sets = [
        np.where(unique_inverse == u)[0] for u in set(unique_inverse) if unique_counts[u] > 1
    ]

    points_missing_exact_duplicates = []

    for duplicate_inds in exact_duplicate_sets:
        for i in duplicate_inds:
            dists = distances[i]
            inds = indices[i]

            same_point_indices = np.setdiff1d(duplicate_inds, i)

            # Figure out how many were already included in the original knn graph
            pre_existing_same_point_indices = np.intersect1d(same_point_indices, inds)

            # Optionally warn the user if there are more identical points than slots available in the existing knn graph
            same_point_indices_set_is_larger = len(pre_existing_same_point_indices) < len(
                same_point_indices
            )
            slots_larger = len(pre_existing_same_point_indices) < k
            if enable_warning and same_point_indices_set_is_larger and slots_larger:
                points_missing_exact_duplicates.append(i)

            # Determine the number of same points to include, respecting the limit of k
            num_same = len(same_point_indices)
            num_same_included = min(num_same, k)  # ensure we do not exceed k neighbors

            # Include same points in the results
            corrected_indices[i, :num_same_included] = same_point_indices[:num_same_included]

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

            # Finally, set the distances between exact duplicates to zero
            corrected_distances[i, :num_same_included] = 0

    if enable_warning and points_missing_exact_duplicates:
        # If the set of same points is larger than the number of slots available in the knn graph
        # and the number of same points already included in the knn graph is less than k,
        # there were some slots available for an exact duplicate that were missed. This should
        # not happen in practice, so the user should be warned.
        warnings.warn("There were some slots available for an exact duplicate that were missed.")

    return corrected_distances, corrected_indices


def _configure_num_neighbors(features: FeatureArray, k: Optional[int]):
    # Error if the provided value is greater or equal to the number of examples.
    N = features.shape[0]
    k_larger_than_dataset = k is not None and k >= N
    if k_larger_than_dataset:
        raise ValueError(
            f"Number of nearest neighbors k={k} cannot exceed the number of examples N={len(features)} passed into the estimator (knn)."
        )

    # Either use the provided value or select a default value based on the feature array size.
    k = k or min(DEFAULT_K, N - 1)
    return k
