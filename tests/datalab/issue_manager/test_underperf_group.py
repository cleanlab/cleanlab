import numpy as np
import pytest

from cleanlab.datalab.internal.issue_manager.underperf_group import UnderperformingGroupIssueManager
from sklearn.datasets import make_blobs

SEED = 42


class TestUnderperformingGroupIssueManager:
    @pytest.fixture
    def make_data(self, lab, noisy=False):
        def data(noisy=noisy):
            N = lab.get_info("statistics")["num_examples"] * 40
            K = lab.get_info("statistics")["num_classes"] + 1  # To obtain even K
            features, labels = make_blobs(n_samples=N, centers=K, n_features=2, random_state=SEED)
            pred_probs = np.full((N, K), 0.1)
            pred_probs[np.arange(N), labels] = 0.9
            pred_probs = pred_probs / np.sum(pred_probs, axis=-1, keepdims=True)
            if noisy:  # Swap columns of a class to generate incorrect predictions
                pred_probs_slice = pred_probs[labels == 0]
                pred_probs_slice[:, [0, 1]] = pred_probs_slice[:, [1, 0]]
                pred_probs[labels == 0] = pred_probs_slice
            data = {"features": features, "pred_probs": pred_probs, "labels": labels}
            return data

        return data

    @pytest.fixture
    def issue_manager(self, lab, make_data, monkeypatch):
        data = make_data()
        monkeypatch.setattr(lab._labels, "labels", data["labels"])
        return UnderperformingGroupIssueManager(datalab=lab, threshold=0.2)

    def test_find_issues_no_underperf_group(self, issue_manager, make_data):
        data = make_data()
        features, labels, pred_probs = data["features"], data["labels"], data["pred_probs"]
        N = len(labels)
        issue_manager.find_issues(features=features, pred_probs=pred_probs)
        issues, summary = issue_manager.issues, issue_manager.summary
        assert np.sum(issues["is_underperforming_group_issue"]) == 0
        expected_issue_mask = np.full(N, False, bool)
        assert np.all(
            issues["is_underperforming_group_issue"] == expected_issue_mask
        ), "Issue mask should be correct"
        expected_scores = np.ones(N)
        np.testing.assert_allclose(
            issues["underperforming_group_score"],
            expected_scores,
            err_msg="Scores should be correct",
        )
        assert summary["issue_type"][0] == "underperforming_group"
        assert summary["score"][0] == 1.0

    def test_find_issues(self, issue_manager, make_data):
        data = make_data(noisy=True)
        features, labels, pred_probs = data["features"], data["labels"], data["pred_probs"]
        N = len(labels)
        issue_manager.find_issues(features=features, pred_probs=pred_probs)
        issues, summary = issue_manager.issues, issue_manager.summary
        assert np.sum(issues["is_underperforming_group_issue"]) == 50
        expected_issue_mask = np.zeros(N, bool)
        expected_issue_mask[labels == 0] = True
        assert np.all(
            issues["is_underperforming_group_issue"] == expected_issue_mask
        ), "Issue mask should be correct"
        expected_loss_ratio = 0.1428
        expected_scores = np.ones(N)
        expected_scores[labels == 0] = expected_loss_ratio
        np.testing.assert_allclose(
            issues["underperforming_group_score"],
            expected_scores,
            err_msg="Scores should be correct",
            rtol=1e-3,
        )
        assert summary["issue_type"][0] == "underperforming_group"
        assert summary["score"][0] == pytest.approx(expected_loss_ratio, rel=1e-3)

    def test_collect_info(self, issue_manager, make_data):
        data = make_data()
        features, pred_probs = data["features"], data["pred_probs"]
        issue_manager.find_issues(features=features, pred_probs=pred_probs)
        info = issue_manager.info
        assert "clustering" in info
        clustering_info = info["clustering"]
        assert clustering_info["algorithm"] == "DBSCAN"
        assert clustering_info["params"]["metric"] == "precomputed"
        assert clustering_info["stats"]["n_clusters"] == 4
        assert clustering_info["stats"]["silhouette_score"] == pytest.approx(0.7918, rel=1e-3)

    def test_no_meaningful_clusters(self, issue_manager, make_data, lab):
        np.random.seed(SEED)
        N = 200  # Assign value greater than k
        features = np.random.uniform(0, 20, (N, 10))
        data = make_data()
        pred_probs = data["pred_probs"]
        try:
            issue_manager.find_issues(features=features, pred_probs=pred_probs[:N])
        except ValueError as err:
            assert "No meaningful clusters" in str(err)

    def test_report(self, issue_manager, make_data):
        data = make_data()
        features, pred_probs = data["features"], data["pred_probs"]
        issue_manager.find_issues(features=features, pred_probs=pred_probs)
        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info=issue_manager.info,
        )
        assert isinstance(report, str)
        assert (
            "--------------- underperforming_group issues ---------------\n\nNumber of examples with this issue"
        ) in report
