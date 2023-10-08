import numpy as np
import pytest

from cleanlab.datalab.internal.issue_manager.null import (
    NullIssueManager,
)

SEED = 42


class TestNullIssueManager:
    @pytest.fixture
    def embeddings(self, request):
        no_null = request.param
        np.random.seed(SEED)
        embeddings_array = np.random.random((4, 3))
        if not no_null:
            embeddings_array[0][0] = np.NaN
            embeddings_array[2][1] = np.NaN
            embeddings_array[2][2] = np.NaN
        return embeddings_array, no_null

    @pytest.fixture
    def issue_manager(self, lab):
        return NullIssueManager(datalab=lab)

    def test_init(self, lab, issue_manager):
        assert issue_manager.datalab == lab

    @pytest.mark.parametrize("embeddings", [True, False], indirect=["embeddings"])
    def test_find_issues(self, issue_manager, embeddings):
        np.random.seed(SEED)
        embeddings, no_null_flag = embeddings
        issue_manager.find_issues(features=embeddings)
        issues_sort, summary_sort, info_sort = (
            issue_manager.issues,
            issue_manager.summary,
            issue_manager.info,
        )
        if no_null_flag:
            expected_sorted_issue_mask = np.array([False, False, False, False])
            assert np.all(
                issues_sort["is_null_issue"] == expected_sorted_issue_mask
            ), "Issue mask should be correct"
            assert summary_sort["issue_type"][0] == "null"
            assert summary_sort["score"][0] == pytest.approx(expected=0.0, abs=1e-7)
            assert (
                info_sort.get("average_null_score", None) is not None
            ), "Should have average null score"
            assert summary_sort["score"][0] == pytest.approx(
                expected=info_sort["average_null_score"], abs=1e-7
            )
        else:
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

    @pytest.mark.parametrize("embeddings", [True, False], indirect=["embeddings"])
    def test_report(self, issue_manager, embeddings):
        np.random.seed(SEED)
        embeddings, no_null_flag = embeddings
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

    @pytest.mark.parametrize("embeddings", [True, False], indirect=["embeddings"])
    def test_collect_info(self, issue_manager, embeddings):
        """Test some values in the info dict."""
        embeddings, no_null_flag = embeddings
        issue_manager.find_issues(features=embeddings)
        info = issue_manager.info
        if no_null_flag:
            assert info["average_null_score"] == 0.0
        else:
            assert info["average_null_score"] == 0.25
