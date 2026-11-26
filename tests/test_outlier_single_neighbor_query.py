import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from cleanlab.outlier import OutOfDistribution


@pytest.mark.parametrize("algorithm", ["brute", "kd_tree", "ball_tree"])
@pytest.mark.parametrize("k", [1, 4])
def test_score_reuses_the_successful_neighbor_query(monkeypatch, algorithm, k):
    rng = np.random.default_rng(46)
    training = rng.normal(size=(30, 4))
    queries = rng.normal(size=(12, 8))[:, ::2]
    queries.setflags(write=False)
    knn = NearestNeighbors(n_neighbors=6, algorithm=algorithm).fit(training)
    expected_distances, _ = knn.kneighbors(queries)
    average_distances = expected_distances[:, :k].mean(axis=1)
    expected = np.exp(-average_distances / np.median(average_distances))

    original_query = knn.kneighbors
    calls = []

    def query(*args, **kwargs):
        calls.append(1)
        return original_query(*args, **kwargs)

    monkeypatch.setattr(knn, "kneighbors", query)
    estimator = OutOfDistribution(params={"knn": knn, "k": k})
    np.testing.assert_allclose(estimator.score(features=queries), expected, rtol=1e-14)
    assert len(calls) == 1
    assert estimator.params["knn"] is knn


def test_fit_score_queries_training_neighbors_once(monkeypatch):
    original_query = NearestNeighbors.kneighbors
    calls = []

    def query(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original_query(self, *args, **kwargs)

    monkeypatch.setattr(NearestNeighbors, "kneighbors", query)
    features = np.array([[0.0], [1.0], [3.0], [10.0]])
    estimator = OutOfDistribution(params={"k": 2})
    result = estimator.fit_score(features=features, verbose=False)
    expected = np.exp(-np.array([2.0, 1.5, 2.5, 8.0]) / 2.25)
    np.testing.assert_allclose(result, expected, rtol=1e-14)
    assert len(calls) == 1
    assert calls[0][0] == (None,)


def test_unfitted_supplied_index_is_fitted_and_queried(monkeypatch):
    features = np.array([[0.0], [1.0], [3.0], [10.0]])
    knn = NearestNeighbors(n_neighbors=2)
    original_query = knn.kneighbors
    original_fit = knn.fit
    successful_queries = []
    fits = []

    def query(*args, **kwargs):
        result = original_query(*args, **kwargs)
        successful_queries.append(1)
        return result

    def fit(*args, **kwargs):
        fits.append(1)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(knn, "kneighbors", query)
    monkeypatch.setattr(knn, "fit", fit)
    estimator = OutOfDistribution(params={"knn": knn})
    np.testing.assert_allclose(
        estimator.score(features=features), np.exp(-np.array([0.5, 0.5, 1.0, 3.5]) / 0.75)
    )
    assert len(fits) == len(successful_queries) == 1


def test_custom_neighbor_index_keeps_its_existing_query_interface():
    class NeighborIndex:
        n_neighbors = 3
        metric = "euclidean"

        def __init__(self):
            self.calls = 0

        def kneighbors(self, features):
            self.calls += 1
            assert features.shape == (2, 1)
            return np.array([[1.0, 3.0, 6.0], [2.0, 4.0, 8.0]]), np.zeros((2, 3), dtype=int)

    knn = NeighborIndex()
    estimator = OutOfDistribution(params={"knn": knn, "k": 2})
    np.testing.assert_array_equal(
        estimator.score(features=np.ones((2, 1))), np.exp(-np.array([2.0, 3.0]) / 2.5)
    )
    assert knn.calls == 1


def test_invalid_queries_do_not_refit_the_index(monkeypatch):
    knn = NearestNeighbors(n_neighbors=2).fit(np.arange(12.0).reshape(6, 2))

    def fit(*args, **kwargs):
        pytest.fail("Invalid queries must not refit an already fitted index")

    monkeypatch.setattr(knn, "fit", fit)
    estimator = OutOfDistribution(params={"knn": knn})
    with pytest.raises(ValueError, match="features"):
        estimator.score(features=np.ones((3, 4)))


def test_neighbor_count_warning_and_clipping_are_preserved():
    knn = NearestNeighbors(n_neighbors=2).fit(np.arange(6.0).reshape(-1, 1))
    queries = np.array([[0.5], [6.0]])
    average_distances = knn.kneighbors(queries)[0].mean(axis=1)
    expected = np.exp(-average_distances / np.median(average_distances))
    estimator = OutOfDistribution(params={"knn": knn, "k": 20})
    with pytest.warns(UserWarning, match="Value of k changed to k=2"):
        np.testing.assert_array_equal(estimator.score(features=queries), expected)
