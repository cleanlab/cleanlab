import math
from hypothesis import given, settings, strategies as st
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

class TestKNNGraph:
    @staticmethod
    def assert_distances_sorted(distances):
        """Check distances are sorted in ascending order for each row."""
        for row in distances:
            assert all(row[i] <= row[i+1] for i in range(len(row)-1))

    @staticmethod
    def assert_indices_unique(indices):
        """Check that indices are unique across columns and don't have the row's index."""
        for row_idx, row in enumerate(indices):
            assert len(set(row)) == len(row)  # Check uniqueness
            assert row_idx not in row  # Check that row's index is not in the row

    @staticmethod
    def assert_mutual_neighbors_have_same_distances(distances, indices):
        """Verify that mutual neighbors have the same distances."""
        for i in range(distances.shape[0]):
            for j in indices[i]:
                if i in indices[j]:
                    d_ij = distances[i][list(indices[i]).index(j)]
                    d_ji = distances[j][list(indices[j]).index(i)]
                    assert math.isclose(d_ij, d_ji), f"Distances between {i} and {j} do not match: {d_ij} vs {d_ji}"

    @staticmethod
    def assert_mutual_consistency_of_knn_distances(distances, indices):
        """Verify the mutual consistency of k-NN distances:
        For every point i and its neighbor j, ensure that the distance from i to j
        cannot be smaller than the distance from any other neighbor k of j to j.
        """
        for i in range(distances.shape[0]):
            for j in indices[i]:
                d_ij = distances[i][list(indices[i]).index(j)]
                j_neighbors_distances = distances[j]
                if d_ij < max(j_neighbors_distances):
                    assert i in indices[j], f"Point {i} should be a neighbor of point {j}, it's closer than the farthest neighbor of {j}"

    @given(knn_graph=knn_graph_strategy(num_samples=st.integers(min_value=10, max_value=50), k_neighbors=st.integers(min_value=2, max_value=5)))
    def test_knn_graph(self, knn_graph):
        N = knn_graph.shape[0]
        distances = knn_graph.data.reshape(N, -1)
        indices = knn_graph.indices.reshape(N, -1)

        self.assert_distances_sorted(distances)
        self.assert_indices_unique(indices)
        self.assert_mutual_neighbors_have_same_distances(distances, indices)
        self.assert_mutual_consistency_of_knn_distances(distances, indices)
