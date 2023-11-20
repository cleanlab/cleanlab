import numpy as np
import pytest

from scipy.sparse import csr_matrix

from cleanlab.datalab.internal.issue_manager.utils import ConstructedKNNGraph


class TestConstructedKNNGraph:
    @pytest.fixture
    def sparse_matrix(self):
        X = np.random.RandomState(0).rand(5, 5)
        return csr_matrix(X)

    @pytest.fixture
    def constructed_knn_graph_instance(self, lab):
        return ConstructedKNNGraph(lab)

    def test_process_knn_graph_from_inputs_valid_graph(
        self, constructed_knn_graph_instance, sparse_matrix
    ):

        # Check when knn_graph is present in "statistics"
        lab = constructed_knn_graph_instance.datalab
        lab.info["statistics"]["weighted_knn_graph"] = sparse_matrix  # Set key in statistics
        knn_graph = constructed_knn_graph_instance.process_knn_graph_from_inputs(kwargs={})
        assert isinstance(knn_graph, csr_matrix)  # Assert type
        assert knn_graph is sparse_matrix

        # Check when knn_graph is present in kwargs
        kwargs = {"knn_graph": sparse_matrix}
        knn_graph = constructed_knn_graph_instance.process_knn_graph_from_inputs(kwargs)
        assert isinstance(knn_graph, csr_matrix)  # Assert type
        assert (
            knn_graph is sparse_matrix
        )  # Assert that passed sparse matrix is same as returned knn graph

        # Check when knn_graph is passed with k
        kwargs = {"knn_graph": sparse_matrix, "k": 5}
        knn_graph = constructed_knn_graph_instance.process_knn_graph_from_inputs(kwargs)
        assert isinstance(knn_graph, csr_matrix)  # Assert type
        assert (
            knn_graph is sparse_matrix
        )  # Assert that passed sparse matrix is same as returned knn graph

    # Any other cases where valid knn_graph is returned

    def test_process_knn_graph_from_inputs_return_None(
        self, constructed_knn_graph_instance, sparse_matrix
    ):

        # First check for no knn graph (not present in kwargs or "weighted_knn_graph")
        kwargs = {}
        knn_graph = constructed_knn_graph_instance.process_knn_graph_from_inputs(kwargs)
        assert knn_graph is None
        # Pass knn_graph with larger k
        kwargs = {"knn_graph": sparse_matrix, "k": 10}
        knn_graph = constructed_knn_graph_instance.process_knn_graph_from_inputs(kwargs)
        assert knn_graph is None

        # Any other cases where returned knn_graph is None
