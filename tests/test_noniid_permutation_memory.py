import tracemalloc

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.stats import gaussian_kde

from cleanlab import Datalab
from cleanlab.datalab.internal.issue_manager.noniid import NonIIDIssueManager


@pytest.fixture(autouse=True)
def preserve_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def make_graph(n, k):
    rng = np.random.RandomState(21)
    neighbors = np.array(
        [rng.choice(np.delete(np.arange(n), i), size=k, replace=False) for i in range(n)]
    )
    return csr_matrix(
        (
            np.tile(np.arange(1, k + 1, dtype=float), n),
            neighbors.ravel(),
            np.arange(0, n * k + 1, k),
        ),
        shape=(n, n),
    )


def make_manager(n, k, seed, permutations):
    lab = Datalab({"label": np.arange(n) % 2}, label_name="label", verbosity=0)
    manager = NonIIDIssueManager(lab, k=k, seed=seed, num_permutations=permutations)
    graph = make_graph(n, k)
    manager.neighbor_index_choices = manager._get_neighbors(graph)
    manager.statistics = manager._get_statistics(
        np.abs(np.arange(n)[:, None] - manager.neighbor_index_choices)
    )
    return manager, graph


def materialized_reference(manager, permutations, rng):
    n = manager.N
    orders = np.array([rng.permutation(n) for _ in range(permutations)])
    distances = np.abs(orders[:, :, None] - orders[:, manager.neighbor_index_choices])
    background_cdf = np.cumsum((n - np.arange(1, n)) / (n * (n - 1) / 2))
    statistics = []
    for sample in distances:
        sorted_distances = np.r_[np.sort(sample.ravel()), n - 1]
        foreground_cdf = np.arange(len(sorted_distances)) / sample.size
        statistics.append(np.max(np.abs(foreground_cdf - background_cdf[sorted_distances - 1])))
    return gaussian_kde(statistics).integrate_box(manager.statistics["ks"], 100)


@pytest.mark.parametrize("seed", [0, 13, None])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("permutations", [10, 25])
def test_streaming_permutations_preserve_statistics_and_random_draw_order(seed, k, permutations):
    manager, _ = make_manager(80, k, seed, permutations)
    rng = np.random.RandomState(27 if seed is None else seed)
    expected = materialized_reference(manager, permutations, rng)
    expected_next = rng.random_sample()
    np.random.seed(27)

    actual = manager._permutation_test(permutations)

    assert actual == expected
    assert np.random.random_sample() == expected_next


def test_permutation_memory_does_not_store_every_relabelled_neighbor_graph():
    manager, _ = make_manager(1024, 8, 0, 60)
    manager._permutation_test(2)
    tracemalloc.start()
    try:
        result = manager._permutation_test(60)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert 0 <= result <= 1
    assert peak < 2 * 1024 * 1024


@pytest.mark.parametrize("permutations", [10, 25])
def test_public_datalab_non_iid_check_matches_the_materialized_permutation_reference(permutations):
    manager, graph = make_manager(80, 4, 11, permutations)
    expected = materialized_reference(manager, permutations, np.random.RandomState(11))
    lab = manager.datalab

    lab.find_issues(
        knn_graph=graph,
        issue_types={"non_iid": {"k": 4, "seed": 11, "num_permutations": permutations}},
    )

    assert lab.get_info("non_iid")["p-value"] == expected
    assert lab.get_issue_summary("non_iid")["score"].item() == expected
