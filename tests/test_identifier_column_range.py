import tracemalloc

import numpy as np
import pytest

from cleanlab import Datalab
from cleanlab.datalab.internal.issue_manager.identifier_column import IdentifierColumnIssueManager
from cleanlab.datalab.internal.issue_manager_factory import REGISTRY, register
from cleanlab.datalab.internal.task import Task


def make_manager():
    lab = Datalab({"label": [0, 1, 0, 1]}, label_name="label", verbosity=0)
    return IdentifierColumnIssueManager(lab)


@pytest.mark.parametrize(
    "dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]
)
def test_identifier_detection_includes_the_maximum_representable_integer(dtype):
    maximum = int(np.iinfo(dtype).max)
    feature = np.array([maximum - 2, maximum, maximum - 1, maximum], dtype=dtype)
    manager = make_manager()

    with np.errstate(over="raise", invalid="raise"):
        manager.find_issues(feature[:, None])

    assert manager.info["identifier_columns"] == [0]
    assert manager.summary["score"].item() == 0.0
    np.testing.assert_array_equal(manager.issues["identifier_column_score"], np.ones(4))


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint64])
def test_widely_separated_integer_endpoints_are_not_sequential(dtype):
    info = np.iinfo(dtype)
    feature = np.array([info.min, info.max], dtype=dtype)

    with np.errstate(over="raise", invalid="raise"):
        assert not make_manager()._is_sequential(feature)


def test_identifier_check_allocates_by_sample_count_instead_of_integer_span():
    feature = np.array([0, 1000000], dtype=np.int64)
    manager = make_manager()
    tracemalloc.start()
    try:
        sequential = manager._is_sequential(feature)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert not sequential
    assert peak < 256 * 1024


@pytest.mark.parametrize(
    "values,expected",
    [([], False), ([7, 7], False), ([5, 3, 4, 3], True), ([-3, -1, -2], True), ([1, 3, 4], False)],
)
def test_bounded_identifier_check_keeps_existing_set_semantics(values, expected):
    assert make_manager()._is_sequential(np.array(values, dtype=np.int64)) is expected


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_datalab_audits_identifier_columns_at_the_integer_boundary(dtype, monkeypatch):
    # This optional issue type uses the public registration API. Restore the registry after testing.
    monkeypatch.setitem(REGISTRY, Task.CLASSIFICATION, REGISTRY[Task.CLASSIFICATION].copy())
    register(IdentifierColumnIssueManager)
    maximum = int(np.iinfo(dtype).max)
    features = np.array(
        [[maximum - 2, 0], [maximum, 20], [maximum - 1, 40], [maximum, 60]], dtype=dtype
    )
    lab = Datalab({"label": [0, 1, 0, 1]}, label_name="label", verbosity=0)

    lab.find_issues(issue_types={"identifier_column": {"features": features}})

    assert lab.get_info("identifier_column")["identifier_columns"] == [0]
    assert lab.get_issue_summary("identifier_column")["score"].item() == 0.0
