from typing import cast
import pytest
import numpy as np
from sklearn.neighbors import NearestNeighbors

from cleanlab.internal.neighbor import features_to_knn
from cleanlab.internal.neighbor.neighbor import knn_to_knn_graph


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
def test_knn_to_knn_graph(metric):
    N, k = 100, 10
    knn = NearestNeighbors(n_neighbors=k, metric=metric)
    X = np.random.rand(N, 10)
    knn.fit(X)
    knn_graph = knn_to_knn_graph(knn)

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
