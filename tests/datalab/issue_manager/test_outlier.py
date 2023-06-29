import numpy as np
import pandas as pd
import pytest

from cleanlab.datalab.issue_manager.outlier import OutlierIssueManager
from cleanlab.outlier import OutOfDistribution

SEED = 42


class TestOutlierIssueManager:
    @pytest.fixture
    def embeddings(self, lab):
        np.random.seed(SEED)
        embeddings_array = 0.5 + 0.1 * np.random.rand(lab.get_info("statistics")["num_examples"], 2)
        embeddings_array[4, :] = -1
        return {"embedding": embeddings_array}

    @pytest.fixture
    def issue_manager(self, lab):
        return OutlierIssueManager(datalab=lab, k=3)

    @pytest.fixture
    def issue_manager_with_threshold(self, lab):
        return OutlierIssueManager(datalab=lab, k=2, threshold=0.5)

    def test_init(self, issue_manager, issue_manager_with_threshold):
        assert isinstance(issue_manager.ood, OutOfDistribution)
        assert issue_manager.ood.params["k"] == 3
        assert issue_manager.threshold == None

        assert issue_manager_with_threshold.ood.params["k"] == 2
        assert issue_manager_with_threshold.threshold == 0.5

    def test_find_issues(self, issue_manager, issue_manager_with_threshold, embeddings):
        issue_manager.find_issues(features=embeddings["embedding"])
        issues, summary, info = issue_manager.issues, issue_manager.summary, issue_manager.info
        expected_issue_mask = np.array([False] * 4 + [True])
        assert np.all(
            issues["is_outlier_issue"] == expected_issue_mask
        ), "Issue mask should be correct"
        assert summary["issue_type"][0] == "outlier"
        assert summary["score"][0] == pytest.approx(expected=0.7732146, abs=1e-7)

        assert info.get("knn", None) is not None, "Should have knn info"
        assert issue_manager.threshold == pytest.approx(expected=0.37037, abs=1e-5)

        issue_manager_with_threshold.find_issues(features=embeddings["embedding"])

    def test_find_issues_with_pred_probs(self, lab):
        issue_manager = OutlierIssueManager(datalab=lab, threshold=0.3)
        pred_probs = np.array(
            [
                [0.25, 0.725, 0.025],
                [0.37, 0.42, 0.21],
                [0.05, 0.05, 0.9],
                [0.1, 0.05, 0.85],
                [0.1125, 0.65, 0.2375],
            ]
        )
        issue_manager.find_issues(pred_probs=pred_probs)
        issues, summary, info = issue_manager.issues, issue_manager.summary, issue_manager.info
        expected_issue_mask = np.array([False] * 4 + [True])
        assert np.all(
            issues["is_outlier_issue"] == expected_issue_mask
        ), "Issue mask should be correct"
        assert summary["issue_type"][0] == "outlier"
        assert summary["score"][0] == pytest.approx(expected=0.210, abs=1e-3)

        assert issue_manager.threshold == 0.3

        assert np.all(
            info.get("confident_thresholds", None) == [0.1, 0.5725, 0.56875]
        ), "Should have confident_joint info"

    def test_find_issues_with_different_thresholds(self, lab, embeddings):
        issue_manager = OutlierIssueManager(datalab=lab, k=3, threshold=0.66666)
        issue_manager.find_issues(features=embeddings["embedding"])
        issues, summary, info = issue_manager.issues, issue_manager.summary, issue_manager.info
        expected_issue_mask = np.array([False] * 4 + [True])
        assert np.all(
            issues["is_outlier_issue"] == expected_issue_mask
        ), "Issue mask should be correct"
        assert summary["issue_type"][0] == "outlier"
        assert summary["score"][0] == pytest.approx(expected=0.7732146, abs=1e-7)

        assert issue_manager.threshold == 0.66666

    def test_report(self, issue_manager):
        pred_probs = np.array(
            [
                [0.1, 0.85, 0.05],
                [0.15, 0.8, 0.05],
                [0.05, 0.05, 0.9],
                [0.1, 0.05, 0.85],
                [0.1, 0.65, 0.25],
            ]
        )
        issue_manager.find_issues(pred_probs=pred_probs)
        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info=issue_manager.info,
        )
        assert isinstance(report, str)
        assert (
            "---------------------- outlier issues ----------------------\n\n"
            "Number of examples with this issue:"
        ) in report

        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info=issue_manager.info,
            verbosity=3,
        )
        assert "Additional Information: " in report

        # Mock some vector and matrix values in the info dict
        mock_info = issue_manager.info
        vector = np.array([1, 2, 3, 4, 5, 6])
        matrix = np.array([[i for i in range(20)] for _ in range(10)])
        df = pd.DataFrame(matrix)
        mock_list = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        mock_dict = {"a": 1, "b": 2, "c": 3}
        mock_info["vector"] = vector
        mock_info["matrix"] = matrix
        mock_info["list"] = mock_list
        mock_info["dict"] = mock_dict
        mock_info["df"] = df

        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info={**issue_manager.info, **mock_info},
            verbosity=4,
        )
        assert "Additional Information: " in report
        assert "vector: [1, 2, 3, 4, '...']" in report
        assert f"matrix: array of shape {matrix.shape}\n[[ 0 " in report
        assert "list: [9, 8, 7, 6, '...']" in report
        assert 'dict:\n{\n    "a": 1,\n    "b": 2,\n    "c": 3\n}' in report
        assert "df:" in report

        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info={**issue_manager.info, **mock_info},
            verbosity=2,
        )
        assert "Additional Information: " in report
        assert "vector: [1, 2, 3, 4, '...']" not in report
        assert f"matrix: array of shape {matrix.shape}\n[[ 0 " not in report
        assert "list: [9, 8, 7, 6, '...']" not in report
        assert 'dict:\n{\n    "a": 1,\n    "b": 2,\n    "c": 3\n}' not in report
        assert "df:" not in report

    def test_collect_info(self, issue_manager, embeddings):
        """Test some values in the info dict.

        Mainly focused on the nearest neighbor info.
        """

        issue_manager.find_issues(features=embeddings["embedding"])
        info = issue_manager.info

        nearest_neighbors = info["nearest_neighbor"]
        distances_to_nearest_neighbor = info["distance_to_nearest_neighbor"]

        assert nearest_neighbors == [3, 0, 3, 0, 2], "Nearest neighbors should be correct"

        assert pytest.approx(distances_to_nearest_neighbor, abs=1e-3) == [
            0.033,
            0.05,
            0.072,
            0.033,
            2.143,
        ], "Distances to nearest neighbor should be correct"
