import numpy as np
import pytest
import pandas as pd
import scipy.sparse as sp

from cleanlab.datalab.internal.issue_manager.underperforming_group import (
    UnderperformingGroupIssueManager,
)
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

    def test_find_issues_no_underperforming_group(self, issue_manager, make_data):
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
        # Check with cluster_ids param
        issue_manager.find_issues(features=features, pred_probs=pred_probs, cluster_ids=labels)
        issues_with_clabels, summary_with_clabels = issue_manager.issues, issue_manager.summary
        pd.testing.assert_frame_equal(issues_with_clabels, issues)
        pd.testing.assert_frame_equal(summary_with_clabels, summary)

    def test_find_issues(self, issue_manager, make_data):
        RELATIVE_TOLERANCE = 1e-3
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
        expected_loss_ratio = 0.1429
        expected_scores = np.ones(N)
        expected_scores[labels == 0] = expected_loss_ratio
        np.testing.assert_allclose(
            issues["underperforming_group_score"],
            expected_scores,
            err_msg="Scores should be correct",
            rtol=1e-3,
        )
        assert summary["issue_type"][0] == "underperforming_group"
        assert summary["score"][0] == pytest.approx(expected_loss_ratio, rel=RELATIVE_TOLERANCE)
        # Check with cluster_ids param
        issue_manager.find_issues(features=features, pred_probs=pred_probs, cluster_ids=labels)
        issues_with_clabels, summary_with_clabels = issue_manager.issues, issue_manager.summary
        pd.testing.assert_frame_equal(issues_with_clabels, issues, rtol=RELATIVE_TOLERANCE)
        pd.testing.assert_frame_equal(summary_with_clabels, summary, rtol=RELATIVE_TOLERANCE)
        # With shifted cluster_ids
        issue_manager.find_issues(features=features, pred_probs=pred_probs, cluster_ids=labels + 10)
        issues_with_clabels, summary_with_clabels = issue_manager.issues, issue_manager.summary
        pd.testing.assert_frame_equal(issues_with_clabels, issues, rtol=RELATIVE_TOLERANCE)
        pd.testing.assert_frame_equal(summary_with_clabels, summary, rtol=RELATIVE_TOLERANCE)

    def test_collect_info(self, issue_manager, make_data):
        """Test some values in the info dict.

        Mainly focused on the clustering info.
        """
        UNDERPERFORMING_CLUSTER_ID = 0
        data = make_data(noisy=True)
        features, pred_probs, labels = data["features"], data["pred_probs"], data["labels"]
        issue_manager.find_issues(features=features, pred_probs=pred_probs)
        info = issue_manager.info
        assert "weighted_knn_graph" in info["statistics"]
        assert "clustering" in info
        # Check clustering info
        clustering_info = info["clustering"]
        assert clustering_info["algorithm"] == "DBSCAN"
        assert clustering_info["params"]["metric"] == "precomputed"
        assert clustering_info["stats"]["n_clusters"] == 4
        # Test collect_info() with cluster_ids
        issue_manager.find_issues(features=features, pred_probs=pred_probs, cluster_ids=labels)
        info = issue_manager.info
        assert "nearest_neighbor" not in info
        assert "distance_to_nearest_neighbor" not in info
        assert info["statistics"] == {}
        # Check clustering info
        clustering_info = info["clustering"]
        assert clustering_info["algorithm"] is None
        assert clustering_info["params"] == {}
        assert clustering_info["stats"]["underperforming_cluster_id"] == UNDERPERFORMING_CLUSTER_ID
        cluster_labels = clustering_info["stats"]["cluster_ids"]
        issues = issue_manager.issues
        issue_indices = issues.index[issues["is_underperforming_group_issue"]].values
        assert np.all(
            cluster_labels[issue_indices] == UNDERPERFORMING_CLUSTER_ID
        ), "All samples with issue should belong to underperforming cluster"

        np.testing.assert_equal(clustering_info["stats"]["cluster_ids"], labels)

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
        cluster_ids = np.full(N, -1)  # -1 is the outlier label for DBSCAN
        with pytest.raises(ValueError, match=exception_pattern):
            issue_manager.find_issues(
                features=features, pred_probs=pred_probs, cluster_ids=cluster_ids
            )
        # Empty cluster ids
        with pytest.raises(ValueError, match=exception_pattern):
            issue_manager.find_issues(
                features=features, pred_probs=pred_probs, cluster_ids=np.array([], dtype=int)
            )

    def test_min_cluster_samples(self, lab, issue_manager, make_data):
        data = make_data()
        features, pred_probs, labels = data["features"], data["pred_probs"], data["labels"]
        labels[:3] = max(labels) + 1  # New cluster with very few samples
        n_clusters = len(set(labels))
        # Check if small cluster is filtered
        issue_manager.find_issues(features=features, pred_probs=pred_probs, cluster_ids=labels)
        clustering_info = issue_manager.info["clustering"]
        assert clustering_info["stats"]["n_clusters"] == n_clusters - 1

        # New issue manager to consider small cluster as well
        issue_manager = UnderperformingGroupIssueManager(
            datalab=lab, threshold=0.2, min_cluster_samples=3
        )
        issue_manager.find_issues(features=features, pred_probs=pred_probs, cluster_ids=labels)
        clustering_info = issue_manager.info["clustering"]
        assert clustering_info["stats"]["n_clusters"] == n_clusters

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
        monkeypatch.setattr(issue_manager, "info", {})
        issue_manager.find_issues(features=features[:, [1, 3]], pred_probs=pred_probs)
        assert np.sum(issue_manager.issues["is_underperforming_group_issue"]) == 49
        # Find underperforming group based on all features
        monkeypatch.setattr(issue_manager, "info", {})
        issue_manager.find_issues(features=features, pred_probs=pred_probs)
        assert np.sum(issue_manager.issues["is_underperforming_group_issue"]) == 48

    def test_knn_graph_change(self, issue_manager):
        dist_matrix = np.random.randint(1, 5, size=(10, 10))
        np.fill_diagonal(dist_matrix, 0)  # Make diagonal 0 to mimic distance matrix
        knn_graph = sp.csr_matrix(dist_matrix)
        nnz_before_clustering = knn_graph.nnz
        issue_manager.perform_clustering(knn_graph)
        nnz_after_clustering = knn_graph.nnz
        assert nnz_before_clustering == nnz_after_clustering

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
