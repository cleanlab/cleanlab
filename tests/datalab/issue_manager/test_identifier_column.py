import numpy as np
import pandas as pd
import pytest
from cleanlab.datalab.internal.issue_manager.identifier_column import IdentifierColumnIssueManager


class TestIdentifierColumnIssueManager:
    @pytest.fixture
    def issue_manager(self, lab):
        return IdentifierColumnIssueManager(datalab=lab)

    @pytest.mark.parametrize(
        "arr, expected_output",
        [
            (np.array([1, 2, 3, 4, 5]), True),
            (np.array([1, 1, 2, 2, 3, 3, 5]), False),
            (np.array([1, 1, 3, 4, 5, 8, 10]), False),
            (np.array([0, 0, 0, 0, 0, 0, 0]), False),
            (np.array([4, 5, 5, 6, 7, 8, 9, 10]), True),
            (np.array([1, 3, 4, 4, 5, 6, 7, -1]), False),
            (np.array([2, 1, 3, 5, 6, 4]), True),
            (np.array([-1, -3, -2, -4, 0]), True),
            (np.array([]), False),
            (np.array([0, 0, 0]), False),
        ],
    )
    def test_is_sequential(self, issue_manager, arr, expected_output):
        assert issue_manager._is_sequential(arr) == expected_output

    @pytest.mark.parametrize(
        "features, expected_prepared_features",
        [
            (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 4], [2, 5], [3, 6]])),
            (
                pd.DataFrame({"A": [1, 4], "B": [2, 5], "C": [3, 6]}),
                [np.array([1, 4]), np.array([2, 5]), np.array([3, 6])],
            ),
            (
                pd.DataFrame({"A": [1, 4], "B": [2.0, 5.0], "C": [3, 6]}),
                [np.array([1, 4]), np.array([2.0, 5.0]), np.array([3, 6])],
            ),
            (
                pd.DataFrame({"A": [1, 4], "B": [2, 5], "C": ["3", "6"]}),
                [np.array([1, 4]), np.array([2, 5]), np.array(["3", "6"], dtype=str)],
            ),
            ([[1, 4], [2, 5], [3, 6]], [np.array([1, 4]), np.array([2, 5]), np.array([3, 6])]),
            (
                {"A": [1, 4], "B": [2, 5], "C": [3, 6]},
                [np.array([1, 4]), np.array([2, 5]), np.array([3, 6])],
            ),
            (
                {"A": [1, 4], "B": [2.0, 5.0], "C": [3, 6], "D": ["a", "b"]},
                [
                    np.array([1, 4]),
                    np.array([2.0, 5.0]),
                    np.array([3, 6]),
                    np.array(["a", "b"], dtype=str),
                ],
            ),
        ],
    )
    def test_prepare_features(self, issue_manager, features, expected_prepared_features):
        prepared_features = issue_manager._prepare_features(features)
        assert np.array_equal(prepared_features, expected_prepared_features)

    @pytest.mark.parametrize(
        "features, expected_indices, expected_is_identifier_column",
        [
            (np.array([[1, 2, 3], [4, 5, 2]]), [2], 0.0),
            (np.array([[1, 2, 3], [1, 3, 2]]), [1, 2], 0.0),
            (
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                [],
                1.0,
            ),
            (
                np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
                [],
                1.0,
            ),
            (
                np.array([[0, 2, 3], [-1, 5, 6], [-2, 2, 3]]),
                [0],
                0.0,
            ),
            (
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
                [],
                1.0,
            ),
            (np.array([[1, 2, 7], [4, 3, 8], [7, 4, 9], [10, 5, 10]]), [1, 2], 0.0),
        ],
    )
    def test_find_issues(
        self, issue_manager, features, expected_indices, expected_is_identifier_column
    ):
        issue_manager.find_issues(features)
        print(f"summary: {issue_manager.summary['score'].values[0]}")
        # print type of score
        score = issue_manager.summary["score"].values[0]
        print(f"score type: {type(score)}")
        print(f"num_identifier_columns: {issue_manager.info['num_identifier_columns']}")
        assert issue_manager.summary["score"].values[0] == expected_is_identifier_column
        assert issue_manager.info["num_identifier_columns"] == len(expected_indices)
        assert np.array_equal(issue_manager.info["identifier_columns"], expected_indices)

    @pytest.mark.parametrize(
        "features, expected_is_identifier_column_issue, expected_is_identifier_column",
        [
            (np.array([[1, 2, 3], [4, 5, 2]]), np.array([False, False]), [1.0, 1.0]),
            (np.array([[1, 2, 3], [1, 3, 2]]), np.array([False, False]), [1.0, 1.0]),
            (
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                np.array([False, False, False]),
                [1.0, 1.0, 1.0],
            ),
            (
                np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
                np.array([False, False, False]),
                [1.0, 1.0, 1.0],
            ),
        ],
    )
    def test_issue_attribute(
        self,
        issue_manager,
        features,
        expected_is_identifier_column_issue,
        expected_is_identifier_column,
    ):
        issue_manager.find_issues(features)
        assert np.array_equal(
            issue_manager.issues[f"is_{issue_manager.issue_name}_issue"],
            expected_is_identifier_column_issue,
        )
        assert np.array_equal(
            issue_manager.issues[issue_manager.issue_score_key], expected_is_identifier_column
        )
