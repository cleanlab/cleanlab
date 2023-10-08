import numpy as np
import pytest

from cleanlab.datalab.internal.issue_manager.null import (
    NullIssueManager,
)

SEED = 42


class TestNullIssueManager:
    @pytest.fixture
    def embeddings(self, lab):
        np.random.seed(SEED)
        embeddings_array = np.random.random((4, 3))
        embeddings_array[0][0] = np.NaN
        embeddings_array[2][1] = np.NaN
        embeddings_array[2][2] = np.NaN
        return embeddings_array

    @pytest.fixture
    def issue_manager(self, lab):
        return NullIssueManager(datalab=lab)

    def test_init(self, lab, issue_manager):
        assert issue_manager.datalab == lab

    def test_find_issues(self, issue_manager, embeddings):
        np.random.seed(SEED)
        issue_manager.find_issues(features=embeddings)
        issues_sort, summary_sort, info_sort = (
            issue_manager.issues,
            issue_manager.summary,
            issue_manager.info,
        )
        expected_sorted_issue_mask = np.array([True, False, True, False])
        assert np.all(
            issues_sort["is_null_issue"] == expected_sorted_issue_mask
        ), "Issue mask should be correct"
        assert summary_sort["issue_type"][0] == "null"
        assert summary_sort["score"][0] == pytest.approx(expected=0.25, abs=1e-7)
        assert (
            info_sort.get("average_null_score", None) is not None
        ), "Should have average null score"
        assert summary_sort["score"][0] == pytest.approx(
            expected=info_sort["average_null_score"], abs=1e-7
        )

    def test_report(self, issue_manager, embeddings):
        np.random.seed(SEED)
        issue_manager.find_issues(features=embeddings)
        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info=issue_manager.info,
        )

        assert isinstance(report, str)
        assert (
            "----------------------- null issues ------------------------\n\n"
            "Number of examples with this issue:"
        ) in report

        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info=issue_manager.info,
            verbosity=3,
        )
        assert "Additional Information: " in report

    def test_collect_info(self, issue_manager, embeddings):
        """Test some values in the info dict."""

        issue_manager.find_issues(features=embeddings)
        info = issue_manager.info

        assert info["average_null_score"] == 0.25
