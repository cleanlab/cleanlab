from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import circulant
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


def construct_knn_graph_from_index(
    knn: NearestNeighbors,
    correction_features: Optional[FeatureArray] = None,
) -> csr_matrix:
    """Construct a sparse distance matrix representation of KNN graph out of a fitted NearestNeighbors search object.

    Parameters
    ----------
    knn :
        A NearestNeighbors object that has been fitted to a feature array.
        The KNN graph is constructed based on the distances and indices of each feature row's nearest neighbors.
    correction_features :
        The input feature array used to fit the NearestNeighbors object.
        If provided, the function the distances and indices of the neighbors will be corrected based on exact
        duplicates in the feature array.
        If not provided, no correction will be applied.

        Warning
        -------
        This function is designed to handle a specific case where a KNN index is used to construct a KNN graph by itself,
        and there is a need to detect and correct for exact duplicates in the feature array. However, relying on this
        function for such corrections is generally discouraged. There are other functions in the module that handle
        KNN graph construction with feature corrections in a more flexible and robust manner. Use this function only
        when there is a special need to correct distances and indices based on the feature array provided.

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

    # Perform self-querying to get the distances and indices of the nearest neighbors
    distances, indices = knn.kneighbors(X=None, return_distance=True)

    # Correct the distances and indices if the correction_features array is provided
    if correction_features is not None:
        distances, indices = correct_knn_distances_and_indices(
            features=correction_features, distances=distances, indices=indices
        )

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

    Raises
    ------
    ValueError :
        If `features` is None, as it's required to construct a KNN graph from scratch.

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
    """
    Corrects a k-nearest neighbors (KNN) graph by handling exact duplicates in the feature array.

    This utility function takes a precomputed KNN graph and the corresponding feature array,
    identifies sets of exact duplicate feature vectors, and corrects the KNN graph to properly
    reflect these duplicates. The corrected KNN graph is returned as a sparse CSR matrix.

    Parameters
    ----------
    features : np.ndarray
        The input feature array, with shape (N, M), where N is the number of samples and M is the number of features.
    knn_graph : csr_matrix
        A sparse matrix of shape (N, N) representing the k-nearest neighbors graph.
        The graph is expected to be in CSR (Compressed Sparse Row) format.

    Returns
    -------
    csr_matrix
        A corrected KNN graph in CSR format with adjusted distances and indices to properly handle
        exact duplicates in the feature array.

    Notes
    -----
    - This function assumes that the input `knn_graph` is already computed and provided in CSR format.
    - The function modifies the KNN graph to ensure that exact duplicates are represented with zero distance
      and correctly updated neighbor indices.
    - This function is useful for post-processing a KNN graph when exact duplicates were not handled during
      the initial KNN computation.

    """
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


def _compute_exact_duplicate_sets(features: FeatureArray) -> List[np.ndarray]:
    """
    Computes the sets of exact duplicate points in the feature array.

    This function groups indices of points that have identical feature vectors.
    It returns a list of arrays, where each array contains the indices of points that are exact duplicates
    of each other.

    Parameters
    ----------
    features : np.ndarray
        The input feature array, with shape (N, M), where N is the number of samples and M is the number of features.

    Returns
    -------
    exact_duplicate_sets
        A list of 1D arrays, where each array contains the indices of exact duplicate points in the dataset.
        Only sets with two or more duplicates are included in the list. If no exact duplicates are found, an empty list is returned.

    Examples
    --------
    >>> features = np.array([[1, 2], [3, 4], [1, 2], [5, 6], [3, 4]])
    >>> _compute_exact_duplicate_sets(features)
    [array([0, 2]), array([1, 4])]  # The row value [1, 2] appears in rows 0 and 2, and [3, 4] appears in rows 1 and 4.

    Notes
    -----
    - This function uses `np.unique` to find unique feature vectors and their inverse indices.
    - This function is intended to be used internally within this module.
    """
    # Use np.unique to catch inverse indices of all unique feature sets
    _, unique_inverse, unique_counts = np.unique(
        features, return_inverse=True, return_counts=True, axis=0
    )

    # Collect different sets of exact duplicates in the dataset
    exact_duplicate_sets = [
        np.where(unique_inverse == u)[0] for u in set(unique_inverse) if unique_counts[u] > 1
    ]

    return exact_duplicate_sets


def correct_knn_distances_and_indices_with_exact_duplicate_sets_inplace(
    distances: np.ndarray,
    indices: np.ndarray,
    exact_duplicate_sets: List[np.ndarray],
) -> None:
    """
    Corrects the distances and indices arrays of k-nearest neighbors (KNN) graphs by handling sets
    of exact duplicates explicitly. This function modifies the input arrays in-place.

    This function ensures that exact duplicates are correctly represented in the KNN graph.
    It modifies the `distances` and `indices` arrays so that each set of exact duplicates
    points to itself with zero distance, and adjusts the nearest neighbors accordingly.

    Parameters
    ----------
    distances :
        A 2D array of shape (N, k) representing the distances between each point of the N points and their k-nearest neighbors.
        This array will be modified in-place to reflect the corrections for exact duplicates (whose mutual distances are explicitly set to zero).
    indices :
        A 2D array of shape (N, k) representing the indices of the nearest neighbors for each of the N points.
        This array will be modified in-place to reflect the corrections for exact duplicates.
    exact_duplicate_sets :
        A list of 1D arrays, each containing the indices of points that are exact duplicates of each other.
        These sets will be used to correct the KNN graph by ensuring that duplicates are reflected as nearest neighbors
        with zero distance.

    High-Level Overview
    -------------------
    The function operates in two main scenarios based on the size of the duplicate sets relative to k:

    1. **Duplicate Set Size >= k + 1**:
       - All nearest neighbors are exact duplicates.
       - The `indices` array is updated such that the first k+1 entries for each duplicate set point are used to represent the nearest neighbors
          of all points in the duplicate set.
       - The rows of the `distances` array belonging to the duplicate set are set to zero.

    2. **Duplicate Set Size < k + 1**:
       - Some of the nearest neighbors are not exact duplicates.
       - Non-duplicate neighbors are shifted to the back of the list.
       - The `indices` and `distances` arrays are updated accordingly to reflect the duplicates at the front with zero distance.

    User Considerations
    -------------------
    - **Input Validity**: Ensure that the `distances` and `indices` arrays have the correct shape and correspond to the same KNN graph.
    - **In-Place Modifications**: The function modifies the input arrays directly. If the original data is needed, make a copy before calling the function.
    - **Duplicate Set Size**: The function is optimized for cases where the number of exact duplicates can be larger than k. Ensure the duplicate sets are accurately identified.
    - **Performance**: The function uses efficient NumPy operations, but performance can be affected by the size of the input arrays and the number of duplicate sets.

    Capabilities
    ------------
    - Handles exact duplicate sets efficiently, ensuring correct KNN graph representation.
    - Maintains zero distances for exact duplicates.
    - Adjusts neighbor indices to reflect the presence of duplicates.

    Limitations
    -----------
    - Assumes that the input arrays (`distances` and `indices`) come from a precomputed KNN graph.
    - Does not handle near-duplicates or merge non-duplicate neighbors.
    - Requires careful construction of `exact_duplicate_sets` to avoid misidentification.
    """

    # Number of neighbors
    k = distances.shape[1]

    for duplicate_inds in exact_duplicate_sets:
        # Determine the number of same points to include, respecting the limit of k
        num_same = len(duplicate_inds)
        num_same_included = min(num_same - 1, k)  # ensure we do not exceed k neighbors

        sorted_first_k_duplicate_inds = _prepare_neighborhood_of_first_k_duplicates(
            duplicate_inds, num_same_included
        )

        if num_same >= k + 1:
            # All nearest neighbors are exact duplicates

            # We only pass in the ciruclant matrix of nearest neighbors
            indices[duplicate_inds[: k + 1]] = sorted_first_k_duplicate_inds
            # But the rest will just take the k first duplicate ids
            indices[duplicate_inds[k + 1 :]] = duplicate_inds[:k]

            # Finally, set the distances between exact duplicates to zero
            distances[duplicate_inds] = 0
        else:
            # Some of the nearest neighbors aren't exact duplicates, move those to the back

            # Get indices and distances from knn that are not the same as i
            different_point_mask = np.isin(indices[duplicate_inds], duplicate_inds, invert=True)

            # Get the indices of the first m True values in each row of the mask
            true_indices = np.argsort(~different_point_mask, axis=1)[:, :-num_same_included]

            # Copy the values to the last m columns in dists
            distances[duplicate_inds, -(k - num_same_included) :] = distances[
                duplicate_inds, true_indices.T
            ].T
            indices[duplicate_inds, -(k - num_same_included) :] = indices[
                duplicate_inds, true_indices.T
            ].T

            # We can pass the circulant matrix to a slice
            indices[duplicate_inds, :num_same_included] = sorted_first_k_duplicate_inds

            # Finally, set the distances between exact duplicates to zero
            distances[duplicate_inds, :num_same_included] = 0

    return None


def _prepare_neighborhood_of_first_k_duplicates(duplicate_inds, num_same_included):
    """
    Prepare a matrix representing the neighborhoods of duplicate items.

    This function constructs a matrix where each row corresponds to an item
    and contains the indices of its nearest neighbors (excluding itself), up
    to a specified number `k`.

    Parameters:
    -----------
    duplicate_inds : list
        A list of indices that represent duplicate items.

    num_same_included : int
        An integer `k` representing the number of neighbors to include for
        each item.

    Returns:
    --------
    np.ndarray
        A matrix where each row contains the sorted indices of the nearest
        neighbors for the corresponding item.

    Explanation:
    ------------
    1. Extract the Base for the Circulant Matrix:
       - The function extracts the first `k+1` elements from `duplicate_inds`
         to form the base of the circulant matrix. This approach ensures that
         even if the set of duplicate items is larger, we only need to consider
         the first `k` duplicates as the nearest neighbors, avoiding conflicts
         with the items themselves.

    2. Create the Circulant Matrix:
       - A circulant matrix is generated from the base, where each row is a
         cyclic permutation of the previous row.

    3. Slice the Matrix to Exclude the First Column:
       - The first column is removed to ensure each row represents the neighbors
         without including the item itself.

    4. Sort the Neighborhood Indices:
       - The rows of the sliced matrix are sorted to ensure a consistent order
         of neighbors.

    Example:
    --------
    Given a set of 5 duplicate items `[A, B, C, D, E]` and `k=2`, the function
    processes this as follows:

    1. `circulant_base` for `k=2` would be `[A, B, C]`.
    2. The `circulant_matrix` might look like:
       ```
       [A B C]
       [B C A]
       [C A B]
       ```
    3. Removing the first column results in:
       ```
       [B C]
       [C A]
       [A B]
       ```
    4. Sorting each row gives the final matrix:
       ```
       [B C]
       [A C]
       [A B]
       ```

    This matrix indicates that:
    - The nearest neighbors of `A` are `[B, C]`.
    - The nearest neighbors of `B` are `[A, C]`.
    - The nearest neighbors of `C` are `[A, B]`.

    For `k=2`, the neighbors of `D`, `E`, onwards could be any of the above.

    The function constructs a sorted matrix of nearest neighbors for a list of
    duplicate items, ensuring an equal distribution of neighbors up to a specified
    number `k`. This process is necessary for tasks requiring an understanding of
    the local neighborhood structure among duplicate examples. By using only the first
    `k+1` elements, the function avoids the need to construct a larger circulant
    matrix, simplifying the computation and ensuring no conflicts among the rest of the items.
    """
    circulant_base = duplicate_inds[: num_same_included + 1]
    circulant_matrix = circulant(circulant_base)
    sliced_circulant_matrix = circulant_matrix[:, 1:]
    sorted_first_k_duplicate_inds = np.sort(sliced_circulant_matrix, axis=1)
    return sorted_first_k_duplicate_inds


def correct_knn_distances_and_indices(
    features: FeatureArray,
    distances: np.ndarray,
    indices: np.ndarray,
    exact_duplicate_sets: Optional[List[np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Corrects the distances and indices of a k-nearest neighbors (KNN) graph
    based on all exact duplicates detected in the feature array.

    Parameters
    ----------
    features :
        The feature array used to construct the KNN graph.
    distances :
        The distances between each point and its k nearest neighbors.
    indices :
        The indices of the k nearest neighbors for each point.
    exact_duplicate_sets:
        A list of numpy arrays, where each array contains the indices of exact duplicates in the feature array. If not provided, it will be computed from the feature array.

    Returns
    -------
    corrected_distances :
        The corrected distances between each point and its k nearest neighbors. Exact duplicates (based on the feature array) are ensured to have zero mutual distance.
    corrected_indices :
        The corrected indices of the k nearest neighbors for each point. Exact duplicates are ensured to be included in the k nearest neighbors, unless the number of exact duplicates exceeds k.

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
    """

    if exact_duplicate_sets is None:
        exact_duplicate_sets = _compute_exact_duplicate_sets(features)

    # Prepare the output arrays
    corrected_distances = np.copy(distances)
    corrected_indices = np.copy(indices)

    correct_knn_distances_and_indices_with_exact_duplicate_sets_inplace(
        distances=corrected_distances,
        indices=corrected_indices,
        exact_duplicate_sets=exact_duplicate_sets,
    )

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
