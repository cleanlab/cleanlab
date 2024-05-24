from typing import cast

from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import pytest
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


from cleanlab.internal.neighbor import features_to_knn
from cleanlab.internal.neighbor.knn_graph import (
    correct_knn_distances_and_indices,
    correct_knn_graph,
    construct_knn_graph_from_index,
    create_knn_graph_and_index,
)


@pytest.mark.parametrize(
    "N",
    [2, 10, 100, 101],
    ids=lambda x: f"N={x}",
)
@pytest.mark.parametrize(
    "M",
    [2, 3, 4, 5, 10, 50, 100],
    ids=lambda x: f"M={x}",
)
def test_features_to_knn(N, M):

    features = np.random.rand(N, M)
    if N >= 100:
        features[-10:] = features[-11]  # Make the last 11 entries all identical, as an edge-case.
    knn = features_to_knn(features)

    assert isinstance(knn, NearestNeighbors)
    knn = cast(NearestNeighbors, knn)
    assert knn.n_neighbors == min(10, N - 1)
    if M > 3:
        metric = knn.metric
        assert metric == "cosine"
    else:
        metric = knn.metric
        if N <= 100:
            assert hasattr(metric, "__name__")
            metric = metric.__name__
        assert metric == "euclidean"

    if N >= 100:
        distances, indices = knn.kneighbors(n_neighbors=10)
        # Assert that the last 10 rows are identical to the 11th last row.
        assert np.allclose(features[-10:], features[-11])
        np.testing.assert_allclose(distances[-11:], 0, atol=1e-15)
        # All the indices belong to the same example, so the set of indices should be the same.
        # No guarantees about the order of the indices, but each point is not considered its own neighbor.
        np.testing.assert_allclose(np.unique(indices[-11:]), np.arange(start=N - 11, stop=N))

    # The knn object should be fitted to the features.
    # TODO: This is not a good test, but it's the best we can do without exposing the internal state of the knn object.
    # Assert is_
    assert knn._fit_X is features


def test_knn_kwargs():
    """Check that features_to_knn passes additional keyword arguments to the NearestNeighbors constructor correctly."""
    N, M = 100, 10
    features = np.random.rand(N, M)
    V = features.var(axis=0)
    knn = features_to_knn(
        features,
        n_neighbors=6,
        metric="seuclidean",
        metric_params={"V": V},
    )

    assert knn.n_neighbors == 6
    assert knn.radius == 1.0
    assert (alg := knn.algorithm) == "auto"
    assert knn.leaf_size == 30
    assert knn.metric == "seuclidean"
    assert knn.metric_params == {"V": V}
    assert knn.p == 2
    assert knn._fit_X is features  # Not a public attribute, bad idea to rely on this attribute.

    # Attributes estimated from fitted data
    assert knn.n_features_in_ == M
    assert knn.effective_metric_params_ == {"V": V}
    assert knn.effective_metric_ == "seuclidean"
    assert knn.n_samples_fit_ == N
    assert (
        knn._fit_method == "ball_tree" if alg == "auto" else alg
    )  # Should be one of ["kd_tree", "ball_tree" and "brute"], set with "algorithm"


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_construct_knn_graph_from_index(metric):
    N, k = 100, 10
    knn = NearestNeighbors(n_neighbors=k, metric=metric)
    X = np.random.rand(N, 10)
    knn.fit(X)
    knn_graph = construct_knn_graph_from_index(knn)

    assert knn_graph.shape == (N, N)
    assert knn_graph.nnz == N * k
    assert knn_graph.dtype == np.float64
    assert np.all(knn_graph.data >= 0)
    assert np.all(knn_graph.indices >= 0)
    assert np.all(knn_graph.indices < 100)

    distances = knn_graph.data.reshape(N, k)
    indices = knn_graph.indices.reshape(N, k)

    # Assert all rows in distances are sorted
    assert np.all(np.diff(distances, axis=1) >= 0)


class TestKNNCorrection:
    def test_knn_graph_corrects_missing_duplicates(self):
        """Test that the KNN graph correction identifies missing duplicates and places them correctly."""
        X = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 1],
            ]
        )

        # k = 2
        retrieved_distances = np.array(
            [
                [0, np.sqrt(2)],
                [0, 0],
                [0, 0],
                [np.sqrt(2)] * 2,
            ]
        )
        retrieved_indices = np.array(
            [
                [1, 3],
                [0, 2],
                [0, 1],
                [0, 1],
            ]
        )

        # Most of the retrieved distances are correct, except for the first row that has two exact duplicates
        expected_distances = np.copy(retrieved_distances)
        expected_distances[0] = [0, 0]
        expected_indices = np.copy(retrieved_indices)
        expected_indices[0] = [1, 2]

        # Simulate an properly ordered KNN graph, which missed an exact duplicate in row 0
        knn_graph = csr_matrix(
            (retrieved_distances.ravel(), retrieved_indices.ravel(), np.arange(0, 9, 2)),
            shape=(4, 4),
        )
        expected_knn_graph = csr_matrix(
            (expected_distances.ravel(), expected_indices.ravel(), np.arange(0, 9, 2)),
            shape=(4, 4),
        )

        # Test that the distances and indices are corrected
        corrected_distances, corrected_indices = correct_knn_distances_and_indices(
            X, retrieved_distances, retrieved_indices
        )
        np.testing.assert_array_equal(corrected_distances, expected_distances)
        np.testing.assert_array_equal(corrected_indices, expected_indices)

        # Test that the knn graph can be corrected as well
        corrected_knn_graph = correct_knn_graph(X, knn_graph)
        np.testing.assert_array_equal(corrected_knn_graph.toarray(), expected_knn_graph.toarray())

    def test_knn_graph_corrects_order_of_duplicates(self):
        """Ensure that KNN correction prioritizes duplicates correctly even when initial indices are out of order."""
        X = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 1],
            ]
        )

        retrieved_distances = np.array(
            [
                [np.sqrt(2), 0],  # Should be [0, 0]
                [0, 0],
                [0, 0],
                [np.sqrt(2)] * 2,
            ]
        )
        retrieved_indices = np.array(
            [
                [3, 1],  # Should be [1, 2]
                [0, 2],
                [1, 0],  # Should be [0, 1]
                [0, 1],
            ]
        )

        expected_distances = np.copy(retrieved_distances)
        expected_distances[0] = [0, 0]

        expected_indices = np.copy(retrieved_indices)
        expected_indices[0] = [1, 2]
        expected_indices[2] = [0, 1]

        # Simulate an IMPROPERLY ordered KNN graph
        knn_graph = csr_matrix(
            (retrieved_distances.ravel(), retrieved_indices.ravel(), np.arange(0, 9, 2)),
            shape=(4, 4),
        )
        expected_knn_graph = csr_matrix(
            (expected_distances.ravel(), expected_indices.ravel(), np.arange(0, 9, 2)),
            shape=(4, 4),
        )

        # Test that the distances and indices are corrected
        corrected_distances, corrected_indices = correct_knn_distances_and_indices(
            X, retrieved_distances, retrieved_indices
        )
        np.testing.assert_array_equal(corrected_distances, expected_distances)
        np.testing.assert_array_equal(corrected_indices, expected_indices)

        # Test that the knn graph can be corrected as well
        corrected_knn_graph = correct_knn_graph(X, knn_graph)
        np.testing.assert_array_equal(corrected_knn_graph.toarray(), expected_knn_graph.toarray())


def noisy_euclidean_distance(x, y):
    """Calculate Euclidean distance and add bias if the distance is exactly zero (points are identical)."""
    distance = np.linalg.norm(x - y)
    if all(x == y):
        distance += 2
    return distance


def test_create_knn_graph_correctness():
    """
    Test to verify that the KNN graph creation and index correction handles duplicate points
    and correctly calculates distances using a modified Euclidean distance metric that adds a large
    bias for data points that are identical.
    """

    # Define a set of points with duplicates
    X = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
        ]
    )

    # Define the expected distances and indices for k=3
    expected_distances = np.array(
        [
            [0, 0, np.sqrt(2)],
            [0, 0, np.sqrt(2)],
            [0, 0, np.sqrt(2)],
            [np.sqrt(2), np.sqrt(2), np.sqrt(2)],
        ]
    )
    expected_indices = np.array(
        [
            [1, 2, 3],
            [0, 2, 3],
            [0, 1, 3],
            [0, 1, 2],
        ]
    )

    ### TESTING graph WITH corrections
    knn_graph_corrected, _ = create_knn_graph_and_index(
        features=X, n_neighbors=3, metric=noisy_euclidean_distance, correct_exact_duplicates=True
    )
    distances_corrected, indices_corrected = knn_graph_corrected.data.reshape(
        4, 3
    ), knn_graph_corrected.indices.reshape(4, 3)

    # Assert the corrected graph matches expected values
    np.testing.assert_array_equal(distances_corrected, expected_distances)
    np.testing.assert_array_equal(indices_corrected, expected_indices)

    ### TESTING graph WITHOUT corrections
    # With the noisy metric, the exact duplicates may be missed
    knn_graph, _ = create_knn_graph_and_index(
        features=X, n_neighbors=3, metric=noisy_euclidean_distance, correct_exact_duplicates=False
    )
    distances, indices = knn_graph.data.reshape(4, 3), knn_graph.indices.reshape(4, 3)

    # Check that all distances in the last row of the *incorrect* graph are identical
    np.testing.assert_array_equal(
        distances[-1], [np.sqrt(2)] * 3
    )  # Don't confuse this with expected_distances[-1]

    # Verify that the first neighbor for the first three points in the incorrect graph is the last point
    np.testing.assert_array_equal(indices[:3, 0], [3] * 3)
    np.testing.assert_array_equal(distances[:3, 0], [np.sqrt(2)] * 3)


@given(
    # A collection of data points to search over<
    X=arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=6, max_value=10), st.integers(min_value=2, max_value=3)
        ),
        elements=st.floats(min_value=-10, max_value=10),
    ),
    # Here are the K nearest neighbors we want to find
    k=st.integers(min_value=1, max_value=5),
)
def test_create_knn_graph_properties(X, k):
    """
    Property-based test to verify that the KNN graph creation handles varying input sizes and
    checks that indices and distances are consistent within the graph.
    """
    knn_graph, _ = create_knn_graph_and_index(
        features=X, n_neighbors=k, metric=noisy_euclidean_distance, correct_exact_duplicates=False
    )
    distances, indices = knn_graph.data.reshape(X.shape[0], k), knn_graph.indices.reshape(
        X.shape[0], k
    )

    # Corrected version to handle exact duplicates
    knn_graph_corrected, _ = create_knn_graph_and_index(
        features=X, n_neighbors=k, metric=noisy_euclidean_distance, correct_exact_duplicates=True
    )
    distances_corrected, indices_corrected = knn_graph_corrected.data.reshape(
        X.shape[0], k
    ), knn_graph_corrected.indices.reshape(X.shape[0], k)

    # Testing properties
    # Ensure no self-references unless k > number of points minus one
    for i in range(X.shape[0]):
        assert i not in indices[i] or k > X.shape[0] - 1

    # Ensure distances are non-negative
    assert np.all(distances >= 0), "All distances should be non-negative"
    # but the corrected distances may be smaller
    assert np.all(distances_corrected <= distances)


@given(
    # This point will be duplicated
    base_point=arrays(
        dtype=np.float64,
        shape=(2,),
        elements=st.floats(min_value=-10, max_value=10, allow_subnormal=False),
    ),
    # This is how many instances there are of the duplicated point
    num_duplicates=st.integers(min_value=2, max_value=5),
    # Here are other points which aren't duplicates, will be post-processed to eliminate exact duplicates,
    # so that they won't affect the duplicate results
    extra_points=arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(min_value=11, max_value=20), st.just(2)),
        elements=st.floats(min_value=15, max_value=20, allow_subnormal=False),
        unique=True,
    ),
    # Here are the K nearest neighbors we want to find
    k=st.integers(min_value=1, max_value=10),
)
def test_knn_graph_duplicate_handling(base_point, num_duplicates, extra_points, k):
    """
    Test to ensure that KNN graph handles duplicates properly by comparing graphs with and without
    exact duplicate corrections.
    """
    # Before the test, ensure that the base_point is not a part of the extra_points (it's ok throw that point out)
    if np.any(_dup := (extra_points == base_point).all(axis=1)):
        extra_points = np.delete(extra_points, np.where(_dup)[0][0], axis=0)

    # Create a dataset with duplicates of a single point and some extra distinct points
    X = np.vstack([np.tile(base_point, (num_duplicates, 1)), extra_points])

    # Run KNN without correcting duplicates
    knn_graph, _ = create_knn_graph_and_index(
        features=X, n_neighbors=k, metric=noisy_euclidean_distance, correct_exact_duplicates=False
    )
    distances, indices = knn_graph.data.reshape(X.shape[0], k), knn_graph.indices.reshape(
        X.shape[0], k
    )

    # Run KNN with correcting duplicates
    knn_graph_corrected, _ = create_knn_graph_and_index(
        features=X, n_neighbors=k, metric=noisy_euclidean_distance, correct_exact_duplicates=True
    )
    distances_corrected, indices_corrected = knn_graph_corrected.data.reshape(
        X.shape[0], k
    ), knn_graph_corrected.indices.reshape(X.shape[0], k)

    # Check two properties of the graphs, once corrected for duplicates
    # 1. Check that duplicate points have the same neighbors in the corrected graph,
    #    they should be their mutual closest neighbors

    # To simplify comparisons across rows, include the row id (omitted in the knn_graphs)
    duplicate_ids = np.arange(num_duplicates)
    # Reshape the duplicate_ids array to a 2D array for later concatenation
    reshaped_duplicate_ids = duplicate_ids.reshape(-1, 1)
    # Get the nearest neighbors for each duplicate, excluding itself
    nearest_neighbors = indices_corrected[:num_duplicates, : (num_duplicates - 1)]

    # Concatenate the duplicate ids with their corresponding nearest neighbors
    # This forms a 2D array where each row represents a duplicate and its nearest neighbors
    # Note that in this test, the neighbors of interest are supposed to be duplicates themselves
    points_and_neighbors = np.hstack((reshaped_duplicate_ids, nearest_neighbors))

    # Define a function to calculate precision
    # Precision is defined as the number of true positives divided by the number of true positives plus the number of false positives
    def calculate_precision(test_set, actual_set):
        true_positives = np.intersect1d(test_set, actual_set)
        return len(true_positives) / len(test_set)

    # All the points and their neighbors belong to the same set of duplicates.
    for row in points_and_neighbors:
        # Assert that the precision of the nearest neighbors with respect to the actual duplicates is 1
        # This means that all nearest neighbors are actual duplicates
        assert calculate_precision(row, duplicate_ids) == 1

    # 2. Distances for duplicates are corrected (should be zeros)
    # Without correcting for duplicates in this test, we assume some distances are non-zero
    # We sum the distances for the first (duplicates-1) items
    # If the distances were correctly calculated, the sum should be zero
    # Therefore, if the sum is greater than zero, the distances were not correctly calculated
    uncorrected_distances_sum = distances[:num_duplicates, : (num_duplicates - 1)].sum(axis=1)
    assert any(
        uncorrected_distances_sum > 0
    ), "Uncorrected distances for duplicates are not greater than zero"
    # With correction for duplicates, the distances should be zero
    # We check this by comparing the corrected distances for the first (duplicates-1) items to zero
    corrected_distances = distances_corrected[: (num_duplicates - 1), : (num_duplicates - 1)]
    np.testing.assert_array_equal(
        corrected_distances, 0, "Corrected distances for duplicates are not zero"
    )


def test_construct_knn_then_correct_knn_graph_does_the_same_work():
    features = np.random.rand(1000, 2)
    features[10:20] = features[10]  # Make the 10th to 20th rows identical
    metric = noisy_euclidean_distance
    n_neighbors = 50

    # Construct the index and knn_graph separately, the correction should happen during the knn_graph construction
    knn = features_to_knn(features, n_neighbors=n_neighbors, metric=metric)
    knn_graph_from_index = construct_knn_graph_from_index(knn)  # Without correction
    knn_graph_from_index_with_correction = construct_knn_graph_from_index(
        knn, correction_features=features
    )
    knn_graph, _ = create_knn_graph_and_index(
        features=features, n_neighbors=n_neighbors, metric=metric
    )

    # knn_graph has correction
    np.testing.assert_array_equal(
        knn_graph_from_index_with_correction.toarray(), knn_graph.toarray()
    )
    # knn_graph_from_index does not have correction
    assert not np.all(knn_graph_from_index.toarray() == knn_graph.toarray())
