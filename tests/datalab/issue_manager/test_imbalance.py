import numpy as np
import pytest

from cleanlab.datalab.issue_manager.imbalance import ClassImbalanceIssueManager

SEED = 42


class TestClassImbalanceIssueManager:
    @pytest.fixture
    def embeddings(self, lab):
        embeddings_array = np.arange(lab.get_info("statistics")["num_examples"] * 10).reshape(-1, 1)
        return embeddings_array

    @pytest.fixture
    def labels(self, lab):
        K = lab.get_info("statistics")["num_classes"]
        N = lab.get_info("statistics")["num_examples"] * 10
        labels = np.random.choice(np.arange(K - 1), size=N, p=[0.5] * (K - 1))
        labels[0] = K - 1  # Rare class
        return labels

    @pytest.fixture
    def issue_manager(self, lab, labels, monkeypatch):
        monkeypatch.setattr(lab._labels, "labels", labels)
        return ClassImbalanceIssueManager(datalab=lab, fraction=0.1)

    def test_find_issues(self, embeddings, issue_manager):
        issue_manager.find_issues(features=embeddings)
        issues, summary = issue_manager.issues, issue_manager.summary
        expected_issue_mask = np.array([True] * 1 + [False] * 49)
        assert np.all(
            issues["is_class_imbalance_issue"] == expected_issue_mask
        ), "Issue mask should be correct"
        assert summary["issue_type"][0] == "class_imbalance"
        assert summary["score"][0] == pytest.approx(expected=0.9804, abs=1e-7)

    def test_report(self, embeddings, issue_manager):
        issue_manager.find_issues(features=embeddings)
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
