import numpy as np
import pytest

from cleanlab.datalab.issue_manager.noniid import (
    NonIIDIssueManager,
    simplified_kolmogorov_smirnov_test,
)

SEED = 42


@pytest.mark.parametrize(
    "neighbor_histogram, non_neighbor_histogram, expected_statistic",
    [
        # Test with equal histograms
        (
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            0.0,
        ),
        # Test with maximum difference in the first bin
        (
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.25, 0.5],
            1.0,
        ),
        # Test with maximum difference in the last bin
        (
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.25, 0.25, 0.0],
            0.25,
        ),
        # Test with arbitrary histograms
        (
            [0.2, 0.3, 0.4, 0.1],
            [0.1, 0.4, 0.25, 0.3],
            0.15,  # (0.2 -> 0.5 -> *0.9* -> 1.0) vs (0.1 -> 0.5 -> *0.75* -> 1.05
        ),
    ],
    ids=[
        "equal_histograms",
        "maximum_difference_in_first_bin",
        "maximum_difference_in_last_bin",
        "arbitrary_histograms",
    ],
)
def test_simplified_kolmogorov_smirnov_test(
    neighbor_histogram, non_neighbor_histogram, expected_statistic
):
    nh = np.array(neighbor_histogram)
    nnh = np.array(non_neighbor_histogram)
    statistic = simplified_kolmogorov_smirnov_test(nh, nnh)
    np.testing.assert_almost_equal(statistic, expected_statistic)


class TestNonIIDIssueManager:
    @pytest.fixture
    def embeddings(self, lab):
        np.random.seed(SEED)
        embeddings_array = np.arange(lab.get_info("statistics")["num_examples"] * 10).reshape(-1, 1)
        return embeddings_array

    @pytest.fixture
    def issue_manager(self, lab):
        return NonIIDIssueManager(
            datalab=lab,
            metric="euclidean",
            k=10,
        )

    def test_init(self, lab, issue_manager):
        assert issue_manager.datalab == lab
        assert issue_manager.metric == "euclidean"
        assert issue_manager.k == 10
        assert issue_manager.num_permutations == 25
        assert issue_manager.significance_threshold == 0.05

        issue_manager = NonIIDIssueManager(
            datalab=lab,
            num_permutations=15,
        )

        assert issue_manager.num_permutations == 15

    def test_find_issues(self, issue_manager, embeddings):
        np.random.seed(SEED)
        issue_manager.find_issues(features=embeddings)
        issues_sort, summary_sort, info_sort = (
            issue_manager.issues,
            issue_manager.summary,
            issue_manager.info,
        )
        expected_sorted_issue_mask = np.array([False] * 46 + [True] + [False] * 3)
        assert np.all(
            issues_sort["is_non_iid_issue"] == expected_sorted_issue_mask
        ), "Issue mask should be correct"
        assert summary_sort["issue_type"][0] == "non_iid"
        assert summary_sort["score"][0] == pytest.approx(expected=0.0, abs=1e-7)
        assert info_sort.get("p-value", None) is not None, "Should have p-value"
        assert summary_sort["score"][0] == pytest.approx(expected=info_sort["p-value"], abs=1e-7)

        permutation = np.random.permutation(len(embeddings))
        new_issue_manager = NonIIDIssueManager(
            datalab=issue_manager.datalab,
            metric="euclidean",
            k=10,
        )
        new_issue_manager.find_issues(features=embeddings[permutation])
        issues_perm, summary_perm, info_perm = (
            new_issue_manager.issues,
            new_issue_manager.summary,
            new_issue_manager.info,
        )
        expected_permuted_issue_mask = np.array([False] * len(embeddings))
        assert np.all(
            issues_perm["is_non_iid_issue"] == expected_permuted_issue_mask
        ), "Issue mask should be correct"
        assert summary_perm["issue_type"][0] == "non_iid"
        # ensure score is large, cannot easily ensure precise value because random seed has different effects on different OS:
        assert summary_perm["score"][0] > 0.05
        assert info_perm.get("p-value", None) is not None, "Should have p-value"
        assert summary_perm["score"][0] == pytest.approx(expected=info_perm["p-value"], abs=1e-7)

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
            "---------------------- non_iid issues ----------------------\n\n"
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
        """Test some values in the info dict.

        Mainly focused on the nearest neighbor info.
        """

        issue_manager.find_issues(features=embeddings)
        info = issue_manager.info

        assert info["p-value"] == 0
        assert info["metric"] == "euclidean"
        assert info["k"] == 10

    @pytest.mark.parametrize(
        "seed",
        [
            "default",
            SEED,
            None,
        ],
        ids=["default", "seed", "no_seed"],
    )
    def test_seed(self, lab, seed):
        num_classes = 10
        means = [
            np.array([np.random.uniform(high=10), np.random.uniform(high=10)])
            for _ in range(num_classes)
        ]
        sigmas = [np.random.uniform(high=1) for _ in range(num_classes)]
        class_stats = list(zip(means, sigmas))
        num_samples = 2000

        def generate_data_iid():
            # This should be IID, resulting in a larger p-value
            samples = []
            labels = []

            for _ in range(num_samples):
                label = np.random.choice(num_classes)
                mean, sigma = class_stats[label]
                sample = np.random.normal(mean, sigma)
                samples.append(sample)
                labels.append(label)
            samples = np.array(samples)
            labels = np.array(labels)
            dataset = {"features": samples, "labels": labels}
            return dataset

        dataset = generate_data_iid()
        embeddings = dataset["features"]

        # Create new issue manager, ignore the lab assigned for this test
        if seed == "default":
            issue_manager = NonIIDIssueManager(
                datalab=lab,
                metric="euclidean",
                k=10,
            )
        else:
            issue_manager = NonIIDIssueManager(
                datalab=lab,
                metric="euclidean",
                k=10,
                seed=seed,
            )
        issue_manager.find_issues(features=embeddings)
        p_value = issue_manager.info["p-value"]

        # Run again with the same seed
        issue_manager.find_issues(features=embeddings)
        p_value2 = issue_manager.info["p-value"]

        assert p_value > 0.0
        if seed is not None or seed == "default":
            assert p_value == p_value2
        else:
            assert p_value != p_value2
