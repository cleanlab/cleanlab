import tracemalloc

import numpy as np
import pandas as pd
import pytest

from cleanlab import Datalab
from cleanlab.datalab.internal.issue_manager.null import NullIssueManager


@pytest.mark.parametrize("width", [1, 7, 8, 9, 65])
def test_most_common_null_pattern_preserves_width_and_original_rows(width):
    pattern = np.arange(width) % 3 == 0
    nulls = np.zeros((7, width), dtype=bool)
    nulls[[1, 3, 6]] = pattern
    nulls[4] = True
    nulls.setflags(write=False)
    expected_rows = [1, 3, 4, 6] if width == 1 else [1, 3, 6]

    result = NullIssueManager._most_common_issue(nulls)["most_common_issue"]

    assert result == {
        "pattern": "".join("1" if missing else "0" for missing in pattern),
        "rows_affected": expected_rows,
        "count": len(expected_rows),
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_equal_frequency_patterns_prefer_the_first_occurrence(reverse):
    nulls = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]], dtype=bool)
    if reverse:
        nulls = nulls[:, ::-1]
    result = NullIssueManager._most_common_issue(nulls)["most_common_issue"]

    assert result == {
        "pattern": "001" if reverse else "100",
        "rows_affected": [1, 4],
        "count": 2,
    }


def test_patterns_that_differ_after_a_byte_boundary_remain_distinct():
    nulls = np.zeros((5, 9), dtype=bool)
    nulls[:, 0] = True
    nulls[[1, 2, 4], 8] = True

    assert NullIssueManager._most_common_issue(nulls)["most_common_issue"] == {
        "pattern": "100000001",
        "rows_affected": [1, 2, 4],
        "count": 3,
    }


def test_null_pattern_without_missing_values():
    assert NullIssueManager._most_common_issue(np.zeros((3, 9), dtype=bool)) == {
        "most_common_issue": {"pattern": "no_null", "rows_affected": [], "count": 0}
    }


def test_public_null_audit_preserves_report_information():
    features = pd.DataFrame(
        {"a": [1.0, np.nan, 2.0, np.nan, np.nan], "b": ["a", "b", None, "d", "e"]}
    )
    original = features.copy(deep=True)
    lab = Datalab(data={"row": np.arange(len(features))})

    lab.find_issues(features=features, issue_types={"null": {}})

    assert lab.get_info("null")["most_common_issue"] == {
        "pattern": "10",
        "rows_affected": [1, 3, 4],
        "count": 3,
    }
    np.testing.assert_array_equal(lab.get_issues("null")["null_score"], [1, 0.5, 0.5, 0.5, 0.5])
    pd.testing.assert_frame_equal(features, original)


def test_one_missing_row_uses_bounded_temporary_memory():
    nulls = np.zeros((20000, 256), dtype=bool)
    nulls[15001, -1] = True

    tracemalloc.start()
    try:
        result = NullIssueManager._most_common_issue(nulls)["most_common_issue"]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result == {"pattern": "0" * 255 + "1", "rows_affected": [15001], "count": 1}
    # A single affected row should not require storage for packing every feature row.
    # This leaves ample room above the row-index and single-pattern allocations.
    assert peak < nulls.nbytes // 16
