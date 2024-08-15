import numpy as np
import pandas as pd
import pytest

from cleanlab import Datalab
from cleanlab.datalab.internal.issue_manager.identifier_column import (
    IdentifierColumnIssueManager,
)


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
def test_is_sequential(arr, expected_output):
    lab = Datalab(pd.DataFrame(arr))
    manager = IdentifierColumnIssueManager(datalab=lab)
    assert manager._is_sequential(arr) == expected_output


@pytest.mark.parametrize(
    "features, expected_prepared_features",
    [
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 4], [2, 5], [3, 6]])),
        (pd.DataFrame({"A": [1, 4], "B": [2, 5], "C": [3, 6]}), np.array([[1, 4], [2, 5], [3, 6]])),
        ([[1, 4], [2, 5], [3, 6]], np.array([[1, 4], [2, 5], [3, 6]])),
        ({"A": [1, 4], "B": [2, 5], "C": [3, 6]}, np.array([[1, 4], [2, 5], [3, 6]])),
    ],
)
def test_prepare_features(features, expected_prepared_features):
    lab = Datalab(pd.DataFrame(features))
    manager = IdentifierColumnIssueManager(datalab=lab)

    prepared_features = manager._prepare_features(features)
    assert np.array_equal(prepared_features, expected_prepared_features)

    if isinstance(features, str):
        with pytest.raises(ValueError):
            manager._prepare_features(features)


@pytest.mark.parametrize(
    "features, expected_num_identifier_columns, expected_indices, expected_total_res",
    [
        (np.array([[1, 2, 3], [4, 5, 2]]), 1, np.array([2]), 0.0),
        (np.array([[1, 2, 3], [1, 3, 2]]), 2, np.array([1, 2]), 0.0),
        (
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            0,
            np.array([]),
            1.0,
        ),
        (
            np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
            0,
            np.array([]),
            1.0,
        ),
        (
            np.array([[0, 2, 3], [-1, 5, 6], [-2, 2, 3]]),
            1,
            np.array([0]),
            0.0,
        ),
        (
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            0,
            np.array([]),
            1.0,
        ),
        (np.array([[1, 2, 7], [4, 3, 8], [7, 4, 9], [10, 5, 10]]),
        2,
        np.array([1, 2]),
        0.0),
    ],
)
def test_find_issues(
    features, expected_num_identifier_columns, expected_indices, expected_total_res
):
    lab = Datalab(pd.DataFrame(features))
    manager = IdentifierColumnIssueManager(datalab=lab)
    manager.find_issues(features)
    assert manager.summary["score"][0] == expected_total_res
    assert manager.info["num_identifier_columns"] == expected_num_identifier_columns
    if (manager.info["identifier_columns"] == expected_indices).all():
        pass
    else:
        raise AssertionError()

@pytest.mark.parametrize(
    "features, expected_is_identifer_column, expected_score",
    [
        (np.array([[1, 2, 3], [4, 5, 2]]), False, [1.0, 1.0]),
        (np.array([[1, 2, 3], [1, 3, 2]]), False, [1.0, 1.0]),
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), False, [1.0, 1.0, 1.0]),
        (np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), False, [1.0, 1.0, 1.0]),
    ],
)
def test_issue_attribute(features, expected_is_identifer_column, expected_score):
    lab = Datalab(pd.DataFrame(features))
    manager = IdentifierColumnIssueManager(datalab=lab)
    manager.find_issues(features)
    if (manager.issues[f"is_{manager.issue_name}_issue"] == expected_is_identifer_column).all():
        pass
    else:
        raise AssertionError()

    if (manager.issues[manager.issue_score_key] == expected_score).all():
        pass
    else:
        raise AssertionError()
