import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cleanlab import Datalab
from cleanlab.datalab.internal.issue_manager.duplicate import NearDuplicateIssueManager


def make_hub_graph(n=80, k=6):
    indices = np.tile(np.arange(k), (n, 1))
    for row in range(k):
        indices[row] = np.delete(np.arange(k + 1), row)
    return csr_matrix((np.zeros(n * k), indices.ravel(), np.arange(0, n * k + 1, k)), shape=(n, n))


def reference_sets(graph, threshold, median):
    forward = []
    for row in range(graph.shape[0]):
        start, stop = graph.indptr[row : row + 2]
        close = graph.data[start:stop] < threshold * median
        forward.append(graph.indices[start:stop][close])
    result = []
    for row, neighbors in enumerate(forward):
        incoming = [
            source
            for source, source_neighbors in enumerate(forward)
            if row in source_neighbors and source not in neighbors
        ]
        result.append(np.append(neighbors, incoming) if incoming else neighbors)
    return result


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("threshold", [0.0, 0.13, 1.0])
def test_reciprocal_sets_preserve_neighbors_order_and_dtype(dtype, threshold):
    rng = np.random.default_rng(191)
    n, k = 30, 5
    indices = np.array([rng.choice(np.delete(np.arange(n), i), k, replace=False) for i in range(n)])
    distances = np.sort(rng.random((n, k)), axis=1)
    graph = csr_matrix(
        (distances.ravel(), indices.ravel(), np.arange(0, n * k + 1, k)), shape=(n, n)
    )
    graph.indices = graph.indices.astype(dtype)
    graph.indptr = graph.indptr.astype(dtype)
    original_arrays = [array.copy() for array in (graph.data, graph.indices, graph.indptr)]
    for array in (graph.data, graph.indices, graph.indptr):
        array.setflags(write=False)

    expected = reference_sets(graph, threshold, median=0.8)
    actual = NearDuplicateIssueManager._neighbors_within_radius(graph, threshold, median=0.8)
    for actual_row, expected_row in zip(actual, expected):
        np.testing.assert_array_equal(actual_row, expected_row)
        assert actual_row.dtype == expected_row.dtype
    for array, original in zip((graph.data, graph.indices, graph.indptr), original_arrays):
        np.testing.assert_array_equal(array, original)


def test_many_incoming_edges_are_appended_together(monkeypatch):
    graph = make_hub_graph(n=160, k=10)
    expected = reference_sets(graph, 0.13, 1.0)
    original_append = np.append
    calls = []

    def append(*args, **kwargs):
        calls.append(1)
        return original_append(*args, **kwargs)

    monkeypatch.setattr(np, "append", append)
    actual = NearDuplicateIssueManager._neighbors_within_radius(graph, 0.13, 1.0)
    for actual_row, expected_row in zip(actual, expected):
        np.testing.assert_array_equal(actual_row, expected_row)
    # Repeatedly growing the same hub array copies an increasing prefix for each edge.
    assert len(calls) <= graph.shape[0]


def test_repeated_forward_edges_do_not_add_repeated_reciprocals():
    graph = csr_matrix(
        (np.zeros(8), np.array([1, 1, 2, 2, 3, 3, 0, 0]), np.arange(0, 9, 2)), shape=(4, 4)
    )
    actual = NearDuplicateIssueManager._neighbors_within_radius(graph, 0.13, 1.0)
    expected = reference_sets(graph, 0.13, 1.0)
    for actual_row, expected_row in zip(actual, expected):
        np.testing.assert_array_equal(actual_row, expected_row)


def test_public_near_duplicate_audit_preserves_graph_and_reports():
    graph = make_hub_graph()
    before = graph.copy()
    lab = Datalab({"id": np.arange(graph.shape[0])})
    lab.find_issues(knn_graph=graph, issue_types={"near_duplicate": {"k": 6}})
    issues = lab.get_issues("near_duplicate")
    assert issues["is_near_duplicate_issue"].all()
    np.testing.assert_array_equal(issues["near_duplicate_score"], np.zeros(graph.shape[0]))
    expected = reference_sets(graph, 0.13, 1.0)
    for actual_row, expected_row in zip(
        lab.get_info("near_duplicate")["near_duplicate_sets"], expected
    ):
        np.testing.assert_array_equal(actual_row, expected_row)
    np.testing.assert_array_equal(graph.data, before.data)
    np.testing.assert_array_equal(graph.indices, before.indices)
    np.testing.assert_array_equal(graph.indptr, before.indptr)
