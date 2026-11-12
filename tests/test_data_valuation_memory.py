import tracemalloc

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from cleanlab.data_valuation import _knn_shapley_score, data_shapley_knn


def dense_recurrence(neighbors, labels, k):
    """Evaluate the original per-example recurrence before averaging."""
    scores = np.zeros((len(labels), len(labels)))
    for row, indices in enumerate(neighbors):
        for rank in range(k - 2, -1, -1):
            scores[row, indices[rank]] = (
                scores[row, indices[rank + 1]]
                + int(labels[indices[rank]] == labels[row])
                - int(labels[indices[rank + 1]] == labels[row])
            )
    return np.mean(scores / k, axis=0)


@pytest.mark.parametrize("k", [1, 3, 9])
@pytest.mark.parametrize("string_labels", [False, True])
def test_linear_memory_scores_preserve_the_original_recurrence(k, string_labels):
    rng = np.random.default_rng(123)
    labels = rng.integers(0, 4, size=31)
    if string_labels:
        labels = np.array(["class-" + str(label) for label in labels])
    neighbors = np.array(
        [rng.choice(np.delete(np.arange(31), row), size=12, replace=False) for row in range(31)]
    )
    expected = dense_recurrence(neighbors, labels, k)

    actual = _knn_shapley_score(neighbors, labels, k)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-16)


def test_repeated_precomputed_neighbor_indices_keep_per_row_assignment_semantics():
    labels = np.array([0, 1, 0, 2])
    neighbors = np.array([[1, 1, 2], [0, 2, 0], [0, 0, 3], [2, 2, 1]])

    actual = _knn_shapley_score(neighbors, labels, k=3)

    np.testing.assert_allclose(actual, dense_recurrence(neighbors, labels, 3))


def test_public_precomputed_graph_scores_match_hand_calculated_contributions():
    labels = np.array([0, 0, 0, 1])
    neighbors = np.array([[1, 2, 3], [0, 2, 3], [1, 3, 0], [2, 1, 0]])
    graph = csr_matrix(
        (np.tile([1.0, 2.0, 3.0], 4), neighbors.ravel(), np.arange(0, 13, 3)), shape=(4, 4)
    )

    actual = data_shapley_knn(labels, knn_graph=graph, k=3)

    # The four rows contribute totals [1, 1, 2, -1], divided by N*k=12.
    np.testing.assert_allclose(actual, [13 / 24, 13 / 24, 7 / 12, 11 / 24])


@pytest.mark.parametrize("k", [1, 2, 4])
def test_public_feature_path_preserves_neighbor_based_scores(k):
    features = np.array([0.0, 1.0, 3.0, 7.0, 15.0])[:, None]
    labels = np.array([0, 1, 1, 0, 0])
    neighbors = NearestNeighbors(n_neighbors=k).fit(features).kneighbors(return_distance=False)
    expected = 0.5 * (1 + dense_recurrence(neighbors, labels, k))

    actual = data_shapley_knn(labels, features=features, metric="euclidean", k=k)

    np.testing.assert_allclose(actual, expected)


def test_precomputed_graph_scoring_avoids_a_dense_example_by_example_matrix():
    n, k = 1024, 8
    labels = np.arange(n) % 3
    neighbors = (np.arange(n)[:, None] + np.arange(1, k + 1)) % n
    graph = csr_matrix(
        (np.tile(np.arange(1.0, k + 1), n), neighbors.ravel(), np.arange(0, n * k + 1, k)),
        shape=(n, n),
    )

    tracemalloc.start()
    try:
        scores = data_shapley_knn(labels, knn_graph=graph, k=k)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert scores.shape == (n,)
    assert np.isfinite(scores).all()
    # Allow ample linear temporary storage, but not even one N-by-N float64 array.
    assert peak < 128 * n * np.dtype(np.float64).itemsize
