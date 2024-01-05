import numpy as np
import pytest

from cleanlab.datalab.internal.issue_manager.imbalance import ClassImbalanceIssueManager

SEED = 42


class TestClassImbalanceIssueManager:
    @pytest.fixture
    def labels(self, lab):
        np.random.seed(SEED)
        K = lab.get_info("statistics")["num_classes"]
        N = lab.get_info("statistics")["num_examples"] * 20
        labels = np.random.choice(np.arange(K - 1), size=N, p=[0.5] * (K - 1))
        labels[0] = K - 1  # Rare class
        return labels

    @pytest.fixture
    def create_issue_manager(self, lab, labels, monkeypatch):
        def manager(labels=labels):
            monkeypatch.setattr(lab._labels, "labels", labels)
            return ClassImbalanceIssueManager(datalab=lab, threshold=0.1)

        return manager

    def test_find_issues(self, create_issue_manager, labels):
        N = len(labels)
        issue_manager = create_issue_manager()
        issue_manager.find_issues()
        issues, summary = issue_manager.issues, issue_manager.summary
        assert np.sum(issues["is_class_imbalance_issue"]) == 1
        expected_issue_mask = np.array([True] + [False] * (N - 1))
        assert np.all(
            issues["is_class_imbalance_issue"] == expected_issue_mask
        ), "Issue mask should be correct"
        expected_scores = np.array([0.01] + [1.0] * (N - 1))
        np.testing.assert_allclose(
            issues["class_imbalance_score"], expected_scores, err_msg="Scores should be correct"
        )
        assert summary["issue_type"][0] == "class_imbalance"
        assert summary["score"][0] == 0.01

    def test_find_issues_no_imbalance(self, labels, create_issue_manager):
        N = len(labels)
        labels[0] = 0
        issue_manager = create_issue_manager(labels)
        issue_manager.find_issues()
        issues, summary = issue_manager.issues, issue_manager.summary
        assert np.sum(issues["is_class_imbalance_issue"]) == 0
        assert np.all(
            issues["is_class_imbalance_issue"] == np.full(N, False)
        ), "Issue mask should be correct"
        scores = issues["class_imbalance_score"]
        expected_scores = np.ones_like(scores)
        expected_scores[labels == 1] = 0.47  # Rare class proportion
        np.testing.assert_allclose(scores, expected_scores, err_msg="Scores should be correct")
        assert summary["issue_type"][0] == "class_imbalance"
        assert summary["score"][0] == 0.47

    def test_find_issues_more_imbalance(self, lab, labels, create_issue_manager):
        K = lab.get_info("statistics")["num_classes"]
        N = len(labels)
        labels[labels == K - 2] = 0
        labels[1:3] = K - 2
        issue_manager = create_issue_manager(labels)
        issue_manager.find_issues()
        issues, summary = issue_manager.issues, issue_manager.summary
        assert np.sum(issues["is_class_imbalance_issue"]) == 1
        expected_issue_mask = np.array([True] + [False] * (N - 1))
        assert np.all(
            issues["is_class_imbalance_issue"] == expected_issue_mask
        ), "Issue mask should be correct"
        expected_scores = np.array([0.01] + [1.0] * (N - 1))
        np.testing.assert_allclose(
            issues["class_imbalance_score"], expected_scores, err_msg="Scores should be correct"
        )
        assert summary["issue_type"][0] == "class_imbalance"
        assert summary["score"][0] == 0.01

    def test_report(self, create_issue_manager):
        issue_manager = create_issue_manager()
        issue_manager.find_issues()
        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info=issue_manager.info,
        )
        assert isinstance(report, str)
        assert (
            "------------------ class_imbalance issues ------------------\n\n"
            "Number of examples with this issue:"
        ) in report
        assert ("Additional Information: \n" "Rarest Class:") in report

    def test_collect_info(self, labels, create_issue_manager):
        # With Imbalance
        issue_manager = create_issue_manager(labels)
        issue_manager.find_issues()
        assert issue_manager.info["Rarest Class"] == 5

        # Without Imbalance
        labels[0] = 0
        issue_manager = create_issue_manager(labels)
        issue_manager.find_issues()
        assert issue_manager.info["Rarest Class"] == "NA"
