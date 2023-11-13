from hypothesis import strategies as st
import numpy as np
from scipy.sparse import csr_matrix


@st.composite
def knn_graph_strategy(draw, num_samples, k_neighbors):
    # If the argument is a strategy, draw a value from it.
    if isinstance(num_samples, st.SearchStrategy):
        num_samples = draw(num_samples)

    if isinstance(k_neighbors, st.SearchStrategy):
        k_neighbors = draw(k_neighbors)

    # Generate a symmetric distance matrix
    upper_triangle = [
        draw(st.lists(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False), 
                     min_size=i, max_size=i, unique=True))
        for i in range(1, num_samples+1)
    ]

    distance_matrix = np.zeros((num_samples, num_samples))
    for i, row in enumerate(upper_triangle):
        distance_matrix[i, :i+1] = row
        distance_matrix[:i+1, i] = row

    np.fill_diagonal(distance_matrix, np.inf)  # To ensure we don't select a point as its own neighbor

    # Compute k-nearest neighbors based on the distance matrix
    sorted_indices = np.argsort(distance_matrix, axis=1)
    kneighbor_indices = sorted_indices[:, :k_neighbors]
    kneighbor_distances = np.array([distance_matrix[i, kneighbor_indices[i]] for i in range(num_samples)])

    knn_graph = csr_matrix((kneighbor_distances.flatten(), kneighbor_indices.flatten(), np.arange(0,( kneighbor_distances.shape[0] * k_neighbors + 1), k_neighbors)), shape=(num_samples, num_samples))
    return knn_graph