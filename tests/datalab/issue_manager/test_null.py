import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import floats, just

from cleanlab.datalab.internal.issue_manager.null import NullIssueManager

SEED = 42


class TestNullIssueManager:
    @pytest.fixture
    def embeddings(self):
        np.random.seed(SEED)
        embeddings_array = np.random.random((4, 3))
        return embeddings_array

    @pytest.fixture
    def embeddings_with_null(self):
        np.random.seed(SEED)
        embeddings_array = np.random.random((4, 3))
        embeddings_array[[0, 3], 0] = np.NaN
        embeddings_array[1] = np.NaN
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
        expected_sorted_issue_mask = np.array([False, False, False, False])
        assert np.all(
            issues_sort["is_null_issue"] == expected_sorted_issue_mask
        ), "Issue mask should be correct"
        assert summary_sort["issue_type"][0] == "null"
        assert summary_sort["score"][0] == pytest.approx(expected=1.0, abs=1e-7)
        assert (
            info_sort.get("average_null_score", None) is not None
        ), "Should have average null score"
        assert summary_sort["score"][0] == pytest.approx(
            expected=info_sort["average_null_score"], abs=1e-7
        )

    def test_find_issues_with_null(self, issue_manager, embeddings_with_null):
        np.random.seed(SEED)
        issue_manager.find_issues(features=embeddings_with_null)
        issues_sort, summary_sort, info_sort = (
            issue_manager.issues,
            issue_manager.summary,
            issue_manager.info,
        )
        expected_sorted_issue_mask = np.array([False, True, False, False])
        assert np.all(
            issues_sort["is_null_issue"] == expected_sorted_issue_mask
        ), "Issue mask should be correct"
        assert summary_sort["issue_type"][0] == "null"
        assert summary_sort["score"][0] == pytest.approx(expected=7 / 12, abs=1e-7)
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

    def test_report_with_null(self, issue_manager, embeddings_with_null):
        np.random.seed(SEED)
        issue_manager.find_issues(features=embeddings_with_null)
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

        assert "Additional Information: " not in report
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
        assert info["average_null_score"] == 1.0
        assert info["most_common_issue"]["pattern"] == "no_null"
        assert info["most_common_issue"]["count"] == 0
        assert info["most_common_issue"]["rows_affected"] == []
        assert info["column_impact"] == [0, 0, 0]

    def test_collect_info_with_nulls(self, issue_manager, embeddings_with_null):
        """Test some values in the info dict."""
        issue_manager.find_issues(features=embeddings_with_null)
        info = issue_manager.info
        assert info["average_null_score"] == pytest.approx(expected=7 / 12, abs=1e-7)
        assert info["most_common_issue"]["pattern"] == "100"
        assert info["most_common_issue"]["count"] == 2
        assert info["most_common_issue"]["rows_affected"] == [0, 3]
        assert info["column_impact"] == [0.75, 0.25, 0.25]

    def test_can_work_with_different_dtypes(self, issue_manager):
        features = pd.DataFrame(
            {
                "bool": [True, False, True, False],
                "object": [True, False, True, np.nan],
                "uint8": np.array([0, 1, 3, 4], dtype=np.uint8),
                "int8": np.array([0, -1, 3, -4], dtype=np.int8),
                "float": [0.1, np.nan, 0.3, 0.4],
            }
        )
        issue_manager.find_issues(features=features)
        info = issue_manager.info
        assert info["average_null_score"] == pytest.approx(expected=18 / 20, abs=1e-7)
        assert info["column_impact"] == [0, 0.25, 0, 0, 0.25]

    # Strategy for generating NaN values
    nan_strategy = just(np.nan)

    # Strategy for generating regular float values, including NaNs
    float_with_nan = floats(allow_nan=True)

    # Strategy for generating NumPy arrays with some NaN values
    features_with_nan_strategy = arrays(
        dtype=np.float64,
        shape=array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=5),
        elements=float_with_nan,
        fill=nan_strategy,
    )

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )  # No need to reset state of issue_manager fixture
    @given(embeddings=features_with_nan_strategy)
    def test_quality_scores_and_full_null_row_identification(self, issue_manager, embeddings):
        # Run the find_issues method
        issue_manager.find_issues(features=embeddings)
        issues_sort, _, _ = (
            issue_manager.issues,
            issue_manager.summary,
            issue_manager.info,
        )

        # Check for the two main properties:

        # 1. The quality score for each row should be the fraction of features which are not null in that row.
        non_null_fractions = [np.count_nonzero(~np.isnan(row)) / len(row) for row in embeddings]
        scores = issues_sort[issue_manager.issue_score_key]
        assert np.allclose(scores, non_null_fractions, atol=1e-7)

        # 2. The rows that are marked as is_null_issue should ONLY be those rows which are 100% null values.
        all_rows_are_null = np.all(np.isnan(embeddings), axis=1)
        assert np.all(issues_sort["is_null_issue"] == all_rows_are_null)
