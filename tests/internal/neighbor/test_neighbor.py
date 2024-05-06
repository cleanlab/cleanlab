from typing import cast
import pytest
import numpy as np
from sklearn.neighbors import NearestNeighbors

from cleanlab.internal.neighbor import features_to_knn
from cleanlab.internal.neighbor.neighbor import construct_knn_graph_from_index


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
    features = np.random.rand(100, 10)
    V = features.var(axis=0)
    knn = features_to_knn(features, n_neighbors=6, metric="seuclidean", metric_params={"V": V})

    assert knn.n_neighbors == 6
    assert knn.metric == "seuclidean"
    assert knn._fit_X is features
    assert knn.metric_params == {"V": V}


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
