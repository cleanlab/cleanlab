import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from cleanlab.datalab.internal.issue_manager.knn_graph_helpers import (
    _process_knn_graph_from_inputs as _test_fn_1,  # Rename for testing purposes
    num_neighbors_in_knn_graph as _get_num_neighbors,
    set_knn_graph as _test_fn_2,  # Rename for testing purposes
)
from cleanlab.internal.neighbor.knn_graph import create_knn_graph_and_index as _make_knn


class TestProcessKNNGraphFromInputs:

    def test_knn_graph_provided_in_kwargs(self):
        """
        Test case 1: Verify that the function uses the knn_graph provided in kwargs.
        """
        kwargs = {"knn_graph": csr_matrix(np.random.random((10, 10)))}
        statistics = {"weighted_knn_graph": None}
        result = _test_fn_1(kwargs, statistics, k_for_recomputation=5)
        assert isinstance(result, csr_matrix)
        assert result.shape == (10, 10)

    def test_knn_graph_stored_in_statistics(self):
        """
        Test case 2: Verify that the function uses the knn_graph stored in statistics
        when not provided in kwargs.
        """
        kwargs = {"knn_graph": None}
        statistics = {"weighted_knn_graph": csr_matrix(np.random.random((10, 10)))}
        result = _test_fn_1(kwargs, statistics, k_for_recomputation=5)
        assert isinstance(result, csr_matrix)
        assert result.shape == (10, 10)

    def test_knn_graph_precedence_of_kwargs_over_statistics(self):
        """
        Test case 3: Verify that the knn_graph provided in kwargs takes precedence
        over the one stored in statistics. The knn_graph provided in kwargs is
        ALWAYS used if available.
        """
        kwargs = {"knn_graph": csr_matrix(np.random.random((10, 10))), "k": 5}
        statistics = {"weighted_knn_graph": csr_matrix(np.random.random((5, 5)))}
        result = _test_fn_1(kwargs, statistics, k_for_recomputation=5)
        assert isinstance(result, csr_matrix)
        assert result.shape == (10, 10)

        # Even if the statistics knn_graph is larger, the user-provided knn_graph is preferred
        statistics = {"weighted_knn_graph": csr_matrix(np.random.random((15, 15)))}
        result = _test_fn_1(kwargs, statistics, k_for_recomputation=5)
        assert result.shape == (10, 10)

        # Even if k is larger than the user-provided knn_graph, the user-provided knn_graph is preferred
        kwargs = {"knn_graph": csr_matrix(np.random.random((10, 10)))}
        statistics = {"weighted_knn_graph": csr_matrix(np.random.random((11, 11)))}
        result = _test_fn_1(kwargs, statistics, k_for_recomputation=15)
        assert result.shape == (10, 10)

    def test_no_knn_graph_provided(self):
        """
        Test case 4: Verify that the function returns None when no knn_graph is provided
        in either kwargs or statistics.
        """
        kwargs = {"knn_graph": None}
        statistics = {"weighted_knn_graph": None}
        result = _test_fn_1(kwargs, statistics, k_for_recomputation=0)
        assert result is None

    def test_insufficient_knn_graph(self):
        """
        Test case 5: Verify the behavior of the function when the knn_graph in
        statistics is insufficient for the given k value.
        """
        k = 20
        kwargs = {"knn_graph": csr_matrix(np.random.random((10, 10)))}
        statistics = {"weighted_knn_graph": None}
        result = _test_fn_1(kwargs, statistics, k)
        assert result.shape == (10, 10)

        kwargs = {"knn_graph": None}
        statistics = {"weighted_knn_graph": csr_matrix(np.random.random((10, 10)))}
        result = _test_fn_1(kwargs, statistics, k)
        assert result is None

        statistics = {"weighted_knn_graph": csr_matrix(np.random.random((21, 21)))}
        result = _test_fn_1(kwargs, statistics, k)
        assert result.shape == (21, 21)

        # With sufficiently small k, the kwargs graph is preferred as it's explicitly provided by the user
        k = 5
        kwargs = {"knn_graph": csr_matrix(np.random.random((10, 10)))}
        result = _test_fn_1(kwargs, statistics, k)
        assert result.shape == (10, 10)

        kwargs = {"knn_graph": None}
        result = _test_fn_1(kwargs, statistics, k)
        assert result.shape == (21, 21)


class TestSetKNNGraph:

    @pytest.fixture
    def small_knn_graph(self):
        knn_graph, _ = _make_knn(np.random.random((10, 5)), n_neighbors=5, metric="euclidean")
        return knn_graph

    @pytest.fixture
    def mid_knn_graph(self):
        knn_graph, _ = _make_knn(np.random.random((10, 5)), n_neighbors=7, metric="euclidean")
        return knn_graph

    @pytest.fixture
    def large_knn_graph(self):
        knn_graph, _ = _make_knn(np.random.random((10, 5)), n_neighbors=9, metric="euclidean")
        return knn_graph

    @pytest.fixture
    def manhattan_knn_graph(self):
        knn_graph, _ = _make_knn(np.random.random((10, 5)), n_neighbors=5, metric="manhattan")
        return knn_graph

    def test_knn_graph_provided_in_kwargs(self, small_knn_graph):
        """
        Test case 1: Verify that the function uses the knn_graph provided in kwargs.
        """
        features = np.random.random((10, 5))
        find_issues_kwargs = {"knn_graph": small_knn_graph}
        statistics = {"weighted_knn_graph": None}
        result_graph, result_metric, _ = _test_fn_2(
            features, find_issues_kwargs, metric="euclidean", k=5, statistics=statistics
        )
        assert isinstance(result_graph, csr_matrix)
        assert _get_num_neighbors(result_graph) == 5
        assert result_metric == "euclidean"

    def test_knn_graph_stored_in_statistics(self, small_knn_graph):
        """
        Test case 2: Verify that the function uses the knn_graph stored in statistics
        when not provided in kwargs.
        """
        features = np.random.random((10, 5))
        find_issues_kwargs = {"knn_graph": None}
        statistics = {"weighted_knn_graph": small_knn_graph, "knn_metric": "euclidean"}
        result_graph, result_metric, _ = _test_fn_2(
            features, find_issues_kwargs, metric="euclidean", k=5, statistics=statistics
        )
        assert isinstance(result_graph, csr_matrix)
        assert _get_num_neighbors(result_graph) == 5
        assert result_metric == "euclidean"

        # Even if k is smaller than what is in statitics, the metric will cause a recompute
        statistics = {"weighted_knn_graph": small_knn_graph, "knn_metric": "euclidean_outdated"}
        result_graph, result_metric, _ = _test_fn_2(
            features, find_issues_kwargs, metric="euclidean", k=4, statistics=statistics
        )
        assert _get_num_neighbors(result_graph) == 4
        assert result_metric == "euclidean"

        # If the metric hasn't changed, but the value of k is larger than the stored knn_graph, the knn_graph is recomputed
        statistics = {"weighted_knn_graph": small_knn_graph, "knn_metric": "euclidean"}
        result_graph, result_metric, _ = _test_fn_2(
            features, find_issues_kwargs, metric="euclidean", k=6, statistics=statistics
        )
        assert _get_num_neighbors(result_graph) == 6
        assert result_metric == "euclidean"

    def test_knn_graph_precedence_of_kwargs_over_statistics(
        self, small_knn_graph, mid_knn_graph, large_knn_graph
    ):
        """
        Test case 3: Verify that the knn_graph provided in kwargs takes precedence
        over the one stored in statistics. The knn_graph provided in kwargs is
        ALWAYS used if available.
        """
        features = np.random.random((10, 5))
        find_issues_kwargs = {"knn_graph": mid_knn_graph}
        statistics = {"weighted_knn_graph": small_knn_graph, "knn_metric": "euclidean"}
        result_graph, result_metric, _ = _test_fn_2(
            features, find_issues_kwargs, metric="euclidean", k=5, statistics=statistics
        )
        assert isinstance(result_graph, csr_matrix)
        assert _get_num_neighbors(result_graph) == 7
        assert result_metric == "euclidean"

        # Even if the statistics knn_graph is larger, the user-provided knn_graph is preferred
        statistics = {"weighted_knn_graph": large_knn_graph, "knn_metric": "euclidean"}
        result_graph, result_metric, _ = _test_fn_2(
            features, find_issues_kwargs, metric="euclidean", k=5, statistics=statistics
        )
        assert _get_num_neighbors(result_graph) == 7
        assert result_metric == "euclidean"

        # Even if k is larger than the user-provided knn_graph, the user-provided knn_graph is preferred
        result_graph, result_metric, _ = _test_fn_2(
            features, find_issues_kwargs, metric="euclidean", k=8, statistics=statistics
        )
        assert _get_num_neighbors(result_graph) == 7
        assert result_metric == "euclidean"

    def test_no_knn_graph_provided(self):
        """
        Test case 4: Verify that the function creates a new knn_graph when no knn_graph
        is provided in either kwargs or statistics. Features are required.
        """
        features = np.random.random((10, 5))
        find_issues_kwargs = {"knn_graph": None}
        statistics = {"weighted_knn_graph": None}
        result_graph, result_metric, _ = _test_fn_2(
            features, find_issues_kwargs, metric="cosine", k=3, statistics=statistics
        )
        assert _get_num_neighbors(result_graph) == 3
        assert result_metric == "cosine"

        with pytest.raises(
            AssertionError, match="Features must be provided to compute the knn graph."
        ):
            _test_fn_2(None, find_issues_kwargs, metric="cosine", k=0, statistics=statistics)

    def test_metric_change_requires_new_knn_graph(self, manhattan_knn_graph):
        """
        Test case 5: Verify that the function creates a new knn_graph if the metric has changed.
        """
        features = np.random.random((10, 5))
        find_issues_kwargs = {"knn_graph": None}
        statistics = {"weighted_knn_graph": manhattan_knn_graph, "knn_metric": "manhattan"}
        result_graph, result_metric, _ = _test_fn_2(
            features, find_issues_kwargs, metric="euclidean", k=2, statistics=statistics
        )
        assert _get_num_neighbors(result_graph) == 2
        assert result_metric == "euclidean"

    def test_knn_graph_with_insufficient_graph(self, small_knn_graph, large_knn_graph):
        """
        Test case 6: Verify the behavior of the function when the knn_graph in
        statistics is insufficient for the given k value.
        """
        features = np.random.random((10, 5))
        k = 8
        find_issues_kwargs = {"knn_graph": small_knn_graph}
        statistics = {"weighted_knn_graph": None}
        result_graph, result_metric, _ = _test_fn_2(
            features, find_issues_kwargs, metric="euclidean", k=k, statistics=statistics
        )
        assert _get_num_neighbors(result_graph) == 5
        assert result_metric == "euclidean"

        # The small graph doesn't have enough neighbors, so it should be recomputed
        find_issues_kwargs = {"knn_graph": None}
        statistics = {"weighted_knn_graph": small_knn_graph}
        result_graph, result_metric, _ = _test_fn_2(
            features, find_issues_kwargs, metric="euclidean", k=k, statistics=statistics
        )
        assert _get_num_neighbors(result_graph) == 8
        assert result_metric is "euclidean"

        # The large graph has more than enough neighbors, so it should be used
        statistics = {"weighted_knn_graph": large_knn_graph}
        result_graph, result_metric, _ = _test_fn_2(
            features, find_issues_kwargs, metric="euclidean", k=k, statistics=statistics
        )
        assert _get_num_neighbors(result_graph) == 9
        assert result_metric is "euclidean"

    def test_knn_returned(self, small_knn_graph):
        features = np.random.random((10, 5))
        k = 3
        result_graph, result_metric, result_knn = _test_fn_2(
            features, {"knn_graph": None}, metric="cosine", k=k, statistics={}
        )
        assert isinstance(result_knn, NearestNeighbors)
        assert result_knn.n_neighbors == k
        assert result_knn.metric == "cosine"

        result_graph, result_metric, result_knn = _test_fn_2(
            features, {"knn_graph": small_knn_graph}, metric="euclidean", k=k, statistics={}
        )
        assert result_knn == None
        assert result_metric == "euclidean"
        np.testing.assert_array_equal(result_graph.toarray(), small_knn_graph.toarray())

        result_graph, result_metric, result_knn = _test_fn_2(
            features,
            {"knn_graph": None},
            metric="euclidean",
            k=k,
            statistics={"weighted_knn_graph": small_knn_graph, "knn_metric": "euclidean"},
        )
        assert result_knn == None
        assert result_metric == "euclidean"
        np.testing.assert_array_equal(result_graph.toarray(), small_knn_graph.toarray())
