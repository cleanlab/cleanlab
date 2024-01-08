from hypothesis import strategies as st
import numpy as np
from scipy.sparse import csr_matrix


@st.composite
def knn_graph_strategy(draw, num_samples, k_neighbors, min_distance=0.0, max_distance=100.0):
    """
    Generate a K-nearest neighbors (KNN) graph based on the given parameters.

    Parameters
    ----------
    draw: A function used to draw values from search strategies.

    num_samples (int or SearchStrategy): The number of samples in the graph.
        If a SearchStrategy is provided, a value will be drawn from it.

    k_neighbors (int or SearchStrategy): The number of nearest neighbors to consider for each sample.
        If a SearchStrategy is provided, a value will be drawn from it.

    Returns
    -------
    knn_graph : csr_matrix
        The KNN graph represented as a sparse matrix.

    Notes
    -----
    - The KNN graph is generated based on a symmetric distance matrix.
    - The distance matrix is computed using randomly generated upper triangle values.
    - The diagonal of the distance matrix is set to infinity to avoid selecting a point as its own neighbor.
    - The K-nearest neighbors are computed based on the distance matrix.
    - The resulting KNN graph is returned as a sparse matrix in csr format.
    - The number of samples must be greater than the number of neighbors.
    - The KNN graph is not guaranteed to be connected (i.e. there may be isolated subgraphs).
    - The KNN graph is a directed graph (i.e. the edges are not symmetric).
    - The neighbors are sorted by distance in the CSR-formatted sparse matrix,
        so the first neighbor is the closest neighbor.
    """
    # If the argument is a strategy, draw a value from it.
    if isinstance(num_samples, st.SearchStrategy):
        num_samples = draw(num_samples)

    if isinstance(k_neighbors, st.SearchStrategy):
        k_neighbors = draw(k_neighbors)

    # Generate a symmetric distance matrix
    upper_triangle = [
        draw(
            st.lists(
                st.floats(
                    min_value=min_distance,
                    max_value=max_distance,
                    allow_nan=False,
                    allow_infinity=False,
                    allow_subnormal=False,
                ),
                min_size=i,
                max_size=i,
                unique=True,
            )
        )
        for i in range(1, num_samples + 1)
    ]

    distance_matrix = np.zeros((num_samples, num_samples))
    for i, row in enumerate(upper_triangle):
        distance_matrix[i, : i + 1] = row
        distance_matrix[: i + 1, i] = row

    np.fill_diagonal(
        distance_matrix, np.inf
    )  # To ensure we don't select a point as its own neighbor

    # Compute k-nearest neighbors based on the distance matrix
    sorted_indices = np.argsort(distance_matrix, axis=1)
    kneighbor_indices = sorted_indices[:, :k_neighbors]
    kneighbor_distances = np.array(
        [distance_matrix[i, kneighbor_indices[i]] for i in range(num_samples)]
    )

    knn_graph = csr_matrix(
        (
            kneighbor_distances.flatten(),
            kneighbor_indices.flatten(),
            np.arange(0, (kneighbor_distances.shape[0] * k_neighbors + 1), k_neighbors),
        ),
        shape=(num_samples, num_samples),
    )
    return knn_graph
