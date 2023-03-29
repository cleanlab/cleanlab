from typing import Optional, Tuple
from annoy import AnnoyIndex
import numpy as np
import pytest
from timeit import timeit
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors

from cleanlab.experimental.datalab.knn import KNN, KNNInterface


class AnnoyKNN(KNNInterface):
    """Annoy class that is compatible with sklearn API."""

    def __init__(self, n_neighbors: int = 5, metric: str = "angular", n_trees: int = 10):
        self.ann_index = None
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_trees = n_trees

    def fit(self, features: np.ndarray):
        dim = features.shape[1]
        self.ann_index = AnnoyIndex(dim, metric=self.metric)
        for i, x in enumerate(features):
            self.ann_index.add_item(i, x)
        self.ann_index.build(self.n_trees)
        return self

    def kneighbors(self, features: np.ndarray, n_neighbors: int = None):
        distances = []
        indices = []
        for x in features:
            idx, dist = self.ann_index.get_nns_by_vector(
                x, n_neighbors or self.n_neighbors, include_distances=True
            )
            distances.append(dist)
            indices.append(idx)
        distances = np.array(distances)
        indices = np.array(indices)
        return distances, indices


class TestKNN:
    # @pytest.mark.parametrize("index", [MockKNN(n_neighbors=3), AnnoyKNN(n_neighbors=3)])
    @pytest.fixture(
        params=[NearestNeighbors(n_neighbors=3), AnnoyKNN(n_neighbors=3)], ids=["Sklearn", "Annoy"]
    )
    def knn(self, request) -> KNN:
        # Assuming each index has a larger n_neighbors than is passed to KNN

        return KNN(n_neighbors=2, knn=request.param)

    @pytest.fixture
    def features(self) -> np.ndarray:
        return np.random.rand(10, 2)

    def test_init(self, knn: KNN) -> None:
        assert knn.n_neighbors == 3

        assert knn.neighbor_indices is None
        assert knn.neighbor_distances is None

        assert KNN(n_neighbors=4).n_neighbors == 4

        with pytest.warns(UserWarning) as record:
            larger_index: KNNInterface = NearestNeighbors(n_neighbors=5)
            assert (
                KNN(n_neighbors=3, knn=larger_index).n_neighbors == 5
            ), "KNN object should override n_neighbors"
        assert len(record) == 1
        warn_message = record[0].message.args[0]
        assert "n_neighbors 3 does not match n_neighbors 5 used to fit knn" in warn_message
        assert "Using the n_neighbors found in the existing KNN search object." in warn_message

    def test_fit(self, knn: KNN, features: np.ndarray) -> None:
        assert knn._fit_features is None
        fit_object = knn.fit(features)
        assert np.all(knn._fit_features == features)
        assert isinstance(fit_object, KNN)

    def test_kneighbors_raises_error_if_not_fit(self, knn: KNN, features: np.ndarray) -> None:
        with pytest.raises(NotFittedError) as record:
            knn.kneighbors(features)
        expected_message = (
            "KNN search object not fit, cannot find neighbors. Call the fit() method first."
        )
        assert expected_message in record.value.args[0]

    def test_kneighbors(self, knn: KNN, features: np.ndarray) -> None:
        knn.fit(features)
        time_1 = timeit(lambda: knn.kneighbors(), number=1)
        time_2 = timeit(lambda: knn.kneighbors(), number=1)

        assert time_1 > time_2, "KNN should cache results, so second call should be faster"
        neighbor_distances, neighbor_indices = knn.kneighbors()

        assert neighbor_distances.shape == (10, 3)
        assert neighbor_indices.shape == (10, 3)
        assert neighbor_distances.dtype == np.float64
        assert neighbor_indices.dtype == np.int64
        assert np.all(
            neighbor_distances[:, 0] != 0
        ), "First neighbor should not be itself when calling knn.kneighbors()"

        # Query with same features
        neighbor_distances, neighbor_indices = knn.kneighbors(features)
        assert neighbor_distances.shape == (10, 3)
        assert neighbor_indices.shape == (10, 3)
        assert np.all(
            neighbor_distances[:, 0] == 0
        ), "knn.kneighbors(features) includes query points as their own nearest neighbors"

    def test_kneighbors_graph(self, knn: KNN, features: np.ndarray) -> None:
        knn.fit(features)
        graph = knn.kneighbors_graph()
        assert graph.shape == (10, 10)
        assert np.all(graph.diagonal() == 0)
        assert np.all(graph.data >= 0)
        assert graph.nnz == features.shape[0] * knn.n_neighbors

    def test_kneighbors_graph_cache(self, knn: KNN, features: np.ndarray) -> None:
        knn.fit(features)
        time_1 = timeit(lambda: knn.kneighbors_graph(), number=1)
        time_2 = timeit(lambda: knn.kneighbors_graph(), number=1)
        assert time_1 > time_2, "KNN should cache results, so second call should be faster"

    def test_kneighbors_graph_features(self, knn: KNN, features: np.ndarray) -> None:
        knn.fit(features)
        N_new = 5
        new_features = np.random.rand(N_new, 2)
        graph = knn.kneighbors_graph(new_features)
        assert graph.shape == (N_new, N_new)
        assert np.any(
            graph.diagonal() != 0
        ), "Diagonal doesn't have to be zero when querying with different features"
        assert np.all(graph.data >= 0)
        assert graph.nnz == N_new * knn.n_neighbors
