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


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("use_graph", [False, True])
def test_all_neighbors_have_a_point_mass_permutation_null(n, use_graph):
    lab = Datalab({"label": np.arange(n) % 2}, label_name="label", verbosity=0)
    if use_graph:
        neighbors = np.array([np.delete(np.arange(n), i) for i in range(n)])
        graph = csr_matrix(
            (np.ones(n * (n - 1)), neighbors.ravel(), np.arange(0, n * (n - 1) + 1, n - 1)),
            shape=(n, n),
        )
        kwargs = {"knn_graph": graph}
    else:
        kwargs = {"features": np.arange(n, dtype=float)[:, None]}

    lab.find_issues(issue_types={"non_iid": {"k": n - 1}}, **kwargs)

    assert lab.get_info("non_iid")["p-value"] == 1.0
    assert lab.get_issue_summary("non_iid")["score"].item() == 1.0
    assert not lab.get_issues("non_iid")["is_non_iid_issue"].any()


@pytest.mark.parametrize(
    "neighbors,seed,expected",
    [([1, 0, 0, 0], 2, 1.0), ([1, 0, 0, 0], 6, 1.0), ([1, 3, 0, 0], 0, 0.0)],
    ids=["at-point-mass", "below-point-mass", "above-point-mass"],
)
def test_constant_null_uses_its_inclusive_tail_probability(neighbors, seed, expected):
    graph = csr_matrix((np.ones(4), neighbors, np.arange(5)), shape=(4, 4))
    lab = Datalab({"label": [0, 1, 0, 1]}, label_name="label", verbosity=0)
    manager = NonIIDIssueManager(lab, k=1, seed=seed, num_permutations=2)

    manager.find_issues(knn_graph=graph)

    assert manager.p_value == expected


def test_nonconstant_permutation_null_keeps_the_existing_gaussian_kde_result():
    n = 12
    neighbors = (np.arange(n) + 1) % n
    graph = csr_matrix((np.ones(n), neighbors, np.arange(n + 1)), shape=(n, n))
    lab = Datalab({"label": np.arange(n) % 2}, label_name="label", verbosity=0)
    manager = NonIIDIssueManager(lab, k=1, seed=7, num_permutations=25)
    manager.find_issues(knn_graph=graph)
    rng = np.random.RandomState(7)
    statistics = []
    for _ in range(25):
        order = rng.permutation(n)
        distances = np.abs(order - order[neighbors])
        sorted_distances = np.r_[np.sort(distances), n - 1]
        background_cdf = np.cumsum((n - np.arange(1, n)) / (n * (n - 1) / 2))
        statistics.append(
            np.max(np.abs(np.arange(n + 1) / n - background_cdf[sorted_distances - 1]))
        )

    assert np.ptp(statistics) > 0
    expected = gaussian_kde(statistics).integrate_box(manager.statistics["ks"], 100)
    assert manager.p_value == expected
