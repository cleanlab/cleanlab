import numpy as np
import pandas as pd
import pytest

from cleanlab.datalab.internal.issue_manager.outlier import OutlierIssueManager
from cleanlab.datalab.internal.issue_manager.data_valuation import DataValuationIssueManager
from cleanlab.datalab.internal.issue_manager.duplicate import NearDuplicateIssueManager
from cleanlab.internal.neighbor.knn_graph import create_knn_graph_and_index

SEED = 42


class TestDataValuationIssueManager:
    @pytest.fixture
    def issue_manager(self, lab):
        return DataValuationIssueManager(datalab=lab, k=3)

    @pytest.fixture
    def outlier_issue_manager(self, lab):
        return OutlierIssueManager(datalab=lab, k=3)

    @pytest.fixture
    def embeddings(self, lab):
        np.random.seed(SEED)
        embeddings_array = 0.5 + 0.1 * np.random.rand(lab.get_info("statistics")["num_examples"], 2)
        embeddings_array[4, :] = -1
        return {"embedding": embeddings_array}

    def test_find_issues_with_input(self, issue_manager, embeddings):
        knn_graph, _ = create_knn_graph_and_index(
            embeddings["embedding"],
            n_neighbors=3,
        )
        issue_manager.find_issues(knn_graph=knn_graph)
        issues, summary, info = issue_manager.issues, issue_manager.summary, issue_manager.info
        assert isinstance(issues, pd.DataFrame), "Issues should be a dataframe"

        assert isinstance(summary, pd.DataFrame), "Summary should be a dataframe"
        assert summary["issue_type"].values[0] == "data_valuation"

        assert isinstance(info, dict), "Info should be a dict"
        assert isinstance(issues, pd.DataFrame), "Issues should be a dataframe"
        info_keys = info.keys()
        expected_keys = [
            "num_low_valuation_issues",
            "average_data_valuation",
        ]
        assert all(
            [key in info_keys for key in expected_keys]
        ), f"Info should have the right keys, but is missing {set(expected_keys) - set(info_keys)}"

    def test_find_issues_with_stats(self, issue_manager, embeddings):
        issue_manager.datalab.find_issues(
            features=embeddings["embedding"], issue_types={"outlier": {"k": 3}}
        )
        issue_manager.find_issues(issue_types={"data_valuation": {"k": 3}})
        issues, summary, info = issue_manager.issues, issue_manager.summary, issue_manager.info
        assert isinstance(issues, pd.DataFrame), "Issues should be a dataframe"

        assert isinstance(summary, pd.DataFrame), "Summary should be a dataframe"
        assert summary["issue_type"].values[0] == "data_valuation"

        assert isinstance(info, dict), "Info should be a dict"
        assert isinstance(issues, pd.DataFrame), "Issues should be a dataframe"
        info_keys = info.keys()
        expected_keys = [
            "num_low_valuation_issues",
            "average_data_valuation",
        ]
        assert all(
            [key in info_keys for key in expected_keys]
        ), f"Info should have the right keys, but is missing {set(expected_keys) - set(info_keys)}"

    def test_find_issues_replaces_knn_graph_with_larger_one(self, lab, embeddings) -> None:
        """
        Tests that a new, larger KNN graph replaces an existing smaller one.

        This test covers the specific branch in `_build_statistics_dictionary`
        where an existing `weighted_knn_graph` is updated because the new
        graph has more non-zero elements (i.e., a larger `k`).
        """
        # Note: The `embeddings` fixture provides a dict.
        features: np.ndarray = embeddings["embedding"]

        # 1. Run with a smaller k to establish an initial graph.
        initial_manager = NearDuplicateIssueManager(
            datalab=lab,
            metric="euclidean",
            k=2,  # A small k
        )
        initial_manager.find_issues(features=features)

        # The computed graph should be in the manager's .info "report".
        initial_info = initial_manager.info
        initial_stats = initial_info.get("statistics", {})
        assert "weighted_knn_graph" in initial_stats, "Initial graph was not in the report."
        old_graph = initial_stats["weighted_knn_graph"]
        assert old_graph.nnz == len(features) * 2

        # 2. Manually update the lab's state to simulate the Datalab runner.
        lab.get_info("statistics")["weighted_knn_graph"] = old_graph
        lab.get_info("statistics")["knn_metric"] = initial_manager.metric

        # 3. Run again with a larger k. This should trigger the update logic.
        new_manager = NearDuplicateIssueManager(
            datalab=lab,
            metric="euclidean",
            k=4,  # A larger k
        )
        new_manager.find_issues(features=features)

        # 4. Verify the new manager's report contains the new, larger graph.
        new_info_stats = new_manager.info.get("statistics", {})
        assert "weighted_knn_graph" in new_info_stats, "New graph was not computed."
        new_graph = new_info_stats["weighted_knn_graph"]

        assert new_graph.nnz > old_graph.nnz, "New graph should be larger."
        assert new_graph.nnz == len(features) * 4

    def test_get_larger_k_than_knn_graph(self, issue_manager, embeddings, outlier_issue_manager):
        outlier_issue_manager.find_issues(features=embeddings["embedding"])
        knn_graph, _ = create_knn_graph_and_index(
            embeddings["embedding"],
            n_neighbors=3,
        )
        issue_manager.k = 4
        expected_error_msg = (
            "The provided knn graph has 3 neighbors, which is less than the required 4 neighbors. "
            "Please ensure that the knn graph you provide has at least as many neighbors as the required value of k."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            issue_manager.find_issues(knn_graph=knn_graph)
