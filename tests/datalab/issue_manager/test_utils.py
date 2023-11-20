import numpy as np
import pytest

# from scipy.sparse import csr_matrix

from cleanlab.datalab.internal.issue_manager.utils import ConstructedKNNGraph
from cleanlab.datalab.datalab import Datalab


class TestConstructedKNNGraph:
    @pytest.fixture
    def datalab_instance(self):
        y = np.random.randint(0, 2, size=30)
        lab = Datalab(data={"y": y}, label_name="y")
        return lab

    def test_process_knn_graph_from_inputs_with_knn_graph(self, datalab_instance):
        constructed_knn_graph_instance = ConstructedKNNGraph(
            datalab_instance
        ).process_knn_graph_from_inputs()
        ## Todo
        # assert (result_knn_graph != None)
