import math
from hypothesis import HealthCheck, given, settings, strategies as st
import pytest

from .conftest import knn_graph_strategy


class TestKNNGraph:
    @staticmethod
    def assert_distances_sorted(distances):
        """Check distances are sorted in ascending order for each row."""
        for row in distances:
            assert all(row[i] <= row[i + 1] for i in range(len(row) - 1))

    @staticmethod
    def assert_indices_unique(indices):
        """Check that neighbor indices are unique and don't have the row's index."""
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
                    assert math.isclose(
                        d_ij, d_ji
                    ), f"Distances between {i} and {j} do not match: {d_ij} vs {d_ji}"

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
                    assert (
                        i in indices[j]
                    ), f"Point {i} should be a neighbor of point {j}, it's closer than the farthest neighbor of {j}"

    @pytest.mark.slow
    @given(
        knn_graph=knn_graph_strategy(
            num_samples=st.integers(min_value=6, max_value=10),
            k_neighbors=st.integers(min_value=2, max_value=5),
        )
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=1000, deadline=None)
    def test_knn_graph(self, knn_graph):
        """Run through the property tests defined above."""
        N = knn_graph.shape[0]
        distances = knn_graph.data.reshape(N, -1)
        indices = knn_graph.indices.reshape(N, -1)

        self.assert_distances_sorted(distances)
        self.assert_indices_unique(indices)
        self.assert_mutual_neighbors_have_same_distances(distances, indices)
        self.assert_mutual_consistency_of_knn_distances(distances, indices)
