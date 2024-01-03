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
        lab = constructed_knn_graph_instance.datalab
        lab.info["statistics"]["weighted_knn_graph"] = sparse_matrix
        knn_graph = constructed_knn_graph_instance.process_knn_graph_from_inputs(kwargs={})
        assert isinstance(knn_graph, csr_matrix)
        assert knn_graph is sparse_matrix
        kwargs = {"knn_graph": sparse_matrix}
        knn_graph = constructed_knn_graph_instance.process_knn_graph_from_inputs(kwargs)
        assert isinstance(knn_graph, csr_matrix)
        assert knn_graph is sparse_matrix
        kwargs = {"knn_graph": sparse_matrix, "k": 5}
        knn_graph = constructed_knn_graph_instance.process_knn_graph_from_inputs(kwargs)
        assert isinstance(knn_graph, csr_matrix)
        assert knn_graph is sparse_matrix

    def test_process_knn_graph_from_inputs_return_None(
        self, constructed_knn_graph_instance, sparse_matrix
    ):
        kwargs = {}
        knn_graph = constructed_knn_graph_instance.process_knn_graph_from_inputs(kwargs)
        assert knn_graph is None
        kwargs = {"knn_graph": sparse_matrix, "k": 10}
        knn_graph = constructed_knn_graph_instance.process_knn_graph_from_inputs(kwargs)
        assert knn_graph is None
