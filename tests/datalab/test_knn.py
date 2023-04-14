from annoy import AnnoyIndex
import numpy as np
import pytest
from timeit import timeit
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors

from cleanlab.datalab.knn import KNN, KNNInterface


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

        larger_index: KNNInterface = NearestNeighbors(n_neighbors=5)
        assert (
            KNN(n_neighbors=3, knn=larger_index).n_neighbors == 5
        ), "KNN object should override n_neighbors"

    def test_default_attributes(self) -> None:
        """Test that the default attributes are set correctly.

        Specifically, the metric and number of neighbors should be set to the same values as the
        knn search object.
        """

        # List of tuples of (kwargs, expected attributes)
        test_kwargs_and_expected_attribute_tuples = [
            # n_neighbors and metric take values from the default knn search object
            ({}, {"n_neighbors": 5, "knn": NearestNeighbors, "metric": "minkowski"}),
            # knn search object is still the default, but it's n_neighbors is overridden
            (
                {"n_neighbors": 10},
                {"n_neighbors": 10, "knn": NearestNeighbors, "metric": "minkowski"},
            ),
            # metric is also overridden
            (
                {"n_neighbors": 7, "metric": "euclidean"},
                {"n_neighbors": 7, "knn": NearestNeighbors, "metric": "euclidean"},
            ),
            # passing a knn search object overrides n_neighbors and metric
            (
                {"n_neighbors": 7, "knn": AnnoyKNN()},
                {"n_neighbors": 5, "knn": AnnoyKNN, "metric": "angular"},
            ),
            # passing a knn search object overrides n_neighbors and metric
            (
                {
                    "n_neighbors": 2,
                    "knn": NearestNeighbors(n_neighbors=8, metric="cosine"),
                    "metric": "euclidean",
                },
                {"n_neighbors": 8, "knn": NearestNeighbors, "metric": "cosine"},
            ),
        ]

        for i, (kwargs, expected_attributes) in enumerate(
            test_kwargs_and_expected_attribute_tuples
        ):
            knn = KNN(**kwargs)
            assert (
                knn.n_neighbors == expected_attributes["n_neighbors"]
            ), f"Failing on n_neighbors for test case {i}"
            assert isinstance(
                knn.knn, expected_attributes["knn"]
            ), f"Failing on knn for test case {i}"
            assert (
                knn.metric == expected_attributes["metric"]
            ), f"Failing on metric for test case {i}"

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
        N_new = 25
        new_features = np.random.rand(N_new, 2)
        graph = knn.kneighbors_graph(new_features)
        assert graph.shape == (N_new, N_new)
        assert np.any(
            graph.diagonal() != 0
        ), "Diagonal doesn't have to be zero when querying with different features"
        assert np.all(graph.data >= 0)
        assert graph.nnz == N_new * knn.n_neighbors

    def test_radius_neighbors(self, features: np.ndarray) -> None:
        knn = KNN()
        knn.fit(features)
        knn.radius_neighbors()  # Default values should work
        neighbor_indices = knn.radius_neighbors(radius=0.2)
        assert neighbor_indices.shape == (10,)
        for indices in neighbor_indices:
            assert isinstance(indices, np.ndarray)
            assert indices.dtype == np.int64

        # Annoy doesn't support radius search at the moment
        knn = KNN(knn=AnnoyKNN())
        knn.fit(features)
        with pytest.raises(NotImplementedError) as record:
            knn.radius_neighbors(radius=0.2)
        error_message = record.value.args[0]
        assert "KNN search object does not have a radius_neighbors method." in error_message
        assert (
            "For now, only sklearn.neighbors.NearestNeighbors objects support the radius_neighbors method out of the box."
            in error_message
        )

    def test_add_item(self, knn: KNN) -> None:
        """Test that add_item() adds a new item to the index (without modifying existing items)"""
        assert knn._fit_features is None
        features = np.random.rand(2, 2)
        # Incrementally adding vectors creates a matrix
        for i, eg in enumerate(features):
            knn.add_item(eg)
            assert knn._fit_features.shape == (i + 1, 2)
            assert np.all(knn._fit_features[-1] == eg)

        # Adding a matrix of vectors with the same number of features works
        new_features = np.random.rand(3, 2)
        knn.add_item(new_features)
        assert knn._fit_features.shape == (5, 2)

    def test_add_item_raises_error(self, knn: KNN) -> None:
        features = np.random.rand(2, 2)
        knn.add_item(features)

        # Adding a vector/matrix with a different number of features raises an error
        incorrect_num_features = np.random.rand(3, 3)
        with pytest.raises(ValueError) as record:
            knn.add_item(incorrect_num_features[0])
        expected_message = "New item has a different number of features than the fit features."
        assert expected_message in record.value.args[0]

        with pytest.raises(ValueError) as record:
            knn.add_item(incorrect_num_features[1:])
        assert expected_message in record.value.args[0]
