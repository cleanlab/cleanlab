import numpy as np
import pytest
import pandas as pd

from cleanlab.datalab.internal.issue_manager.underperf_group import UnderperformingGroupIssueManager
from sklearn.datasets import make_blobs, load_iris

SEED = 42


class TestUnderperformingGroupIssueManager:
    def generate_pred_probs(self, N, K, labels, noisy=False):
        pred_probs = np.full((N, K), 0.1)
        pred_probs[np.arange(N), labels] = 0.9
        pred_probs = pred_probs / np.sum(pred_probs, axis=-1, keepdims=True)
        if noisy:  # Swap columns of a class to generate incorrect predictions
            pred_probs_slice = pred_probs[labels == 0]
            pred_probs_slice[:, [0, 1]] = pred_probs_slice[:, [1, 0]]
            pred_probs[labels == 0] = pred_probs_slice
        return pred_probs

    @pytest.fixture
    def make_data(self, noisy=False):
        def data(noisy=noisy):
            N = 400
            K = 4
            features, labels = make_blobs(n_samples=N, centers=K, n_features=2, random_state=SEED)
            pred_probs = self.generate_pred_probs(N, K, labels, noisy)
            data = {"features": features, "pred_probs": pred_probs, "labels": labels}
            return data

        return data

    @pytest.fixture
    def iris_data(self):
        iris_dataset = load_iris()
        features, labels = iris_dataset.data, iris_dataset.target
        K = len(iris_dataset.target_names)
        N = features.shape[0]
        pred_probs = self.generate_pred_probs(N, K, labels, noisy=True)
        data = {"features": features, "pred_probs": pred_probs, "labels": labels}
        return data

    @pytest.fixture
    def issue_manager(self, lab, make_data, monkeypatch):
        data = make_data()
        monkeypatch.setattr(lab._labels, "labels", data["labels"])
        clustering_kwargs = {"eps": 2}
        return UnderperformingGroupIssueManager(
            datalab=lab, threshold=0.2, clustering_kwargs=clustering_kwargs
        )

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
        # Check with cluster_labels param
        issue_manager.find_issues(features=features, pred_probs=pred_probs, cluster_labels=labels)
        issues_with_clabels, summary_with_clabels = issue_manager.issues, issue_manager.summary
        pd.testing.assert_frame_equal(issues_with_clabels, issues)
        pd.testing.assert_frame_equal(summary_with_clabels, summary)

    def test_find_issues(self, issue_manager, make_data):
        data = make_data(noisy=True)
        features, labels, pred_probs = data["features"], data["labels"], data["pred_probs"]
        N = len(labels)
        issue_manager.find_issues(features=features, pred_probs=pred_probs)
        issues, summary = issue_manager.issues, issue_manager.summary
        assert np.sum(issues["is_underperforming_group_issue"]) == 100
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
        # Check with cluster_labels param
        issue_manager.find_issues(features=features, pred_probs=pred_probs, cluster_labels=labels)
        issues_with_clabels, summary_with_clabels = issue_manager.issues, issue_manager.summary
        pd.testing.assert_frame_equal(issues_with_clabels, issues)
        pd.testing.assert_frame_equal(summary_with_clabels, summary)
        # With shifted cluster IDs
        issue_manager.find_issues(
            features=features, pred_probs=pred_probs, cluster_labels=labels + 10
        )
        issues_with_clabels, summary_with_clabels = issue_manager.issues, issue_manager.summary
        pd.testing.assert_frame_equal(issues_with_clabels, issues)
        pd.testing.assert_frame_equal(summary_with_clabels, summary)

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

    def test_no_meaningful_clusters(self, issue_manager, make_data, monkeypatch):
        np.random.seed(SEED)
        data = make_data()
        k = 10
        N = 20
        # Generate sparse data that cannot be clustered by DBSCAN
        features = np.random.uniform(-100, 100, (N, 2))
        # Dummy pred_probs and labels for running issue manager
        pred_probs = data["pred_probs"][:N]
        monkeypatch.setattr(issue_manager.datalab._labels, "labels", data["labels"][:N])
        monkeypatch.setattr(issue_manager, "k", k)  # Ensure that k is smaller than N
        exception_pattern = "No meaningful clusters"
        with pytest.raises(ValueError, match=exception_pattern):
            issue_manager.find_issues(features=features, pred_probs=pred_probs)
        # Cluster labels passed containing all outliers
        cluster_labels = np.full(N, -1)  # -1 is the outlier label for DBSCAN
        with pytest.raises(ValueError, match=exception_pattern):
            issue_manager.find_issues(
                features=features, pred_probs=pred_probs, cluster_labels=cluster_labels
            )

    def test_find_issues_feature_subset(self, issue_manager, iris_data, monkeypatch):
        features, pred_probs, labels = (
            iris_data["features"],
            iris_data["pred_probs"],
            iris_data["labels"],
        )
        # TODO: Better asserts required. Ideally -> 3 clusters, 50 samples with issue.
        monkeypatch.setattr(issue_manager.datalab._labels, "labels", labels)
        monkeypatch.setattr(issue_manager, "clustering_kwargs", {"eps": 0.5})
        # Find underperforming group based on one feature
        single_feature = features[:, 0].reshape(-1, 1)
        issue_manager.find_issues(features=single_feature, pred_probs=pred_probs)
        assert np.sum(issue_manager.issues["is_underperforming_group_issue"]) == 16
        # Find underperforming group based on two features
        issue_manager.find_issues(features=features[:, [1, 3]], pred_probs=pred_probs)
        assert np.sum(issue_manager.issues["is_underperforming_group_issue"]) == 49
        # Find underperforming group based on all features
        issue_manager.find_issues(features=features, pred_probs=pred_probs)
        assert np.sum(issue_manager.issues["is_underperforming_group_issue"]) == 48

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
