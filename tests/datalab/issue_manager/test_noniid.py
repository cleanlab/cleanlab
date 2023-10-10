import numpy as np
import pytest

from cleanlab.datalab.internal.issue_manager.noniid import (
    NonIIDIssueManager,
    simplified_kolmogorov_smirnov_test,
)

from cleanlab.datalab.internal.helper_factory import report_factory
from cleanlab.datalab.datalab import Datalab

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
    def pred_probs(self, lab):
        pred_probs_array = (
                               np.arange(lab.get_info("statistics")["num_examples"] * 10).reshape(-1, 1)
                           ) / len(np.arange(lab.get_info("statistics")["num_examples"] * 10).reshape(-1, 1))
        return pred_probs_array

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

    @pytest.mark.parametrize("datalab_flag, features_flag",
                             [(True, True), (False, False), (False, True), (True, False)])
    def test_find_issues(self, issue_manager, embeddings, pred_probs, lab, datalab_flag, features_flag, dataset,
                         label_name):

        np.random.seed(SEED)
        if datalab_flag:
            if features_flag:
                lab.find_issues(features=embeddings)
            else:
                lab.find_issues(pred_probs=pred_probs)

            issues_sort, summary_sort, info_sort = (
                lab.get_issues("non_iid"),
                lab.get_issue_summary("non_iid"),
                lab.get_info("non_iid"),
            )
        else:
            if features_flag:
                issue_manager.find_issues(features=embeddings)
            else:
                issue_manager.find_issues(pred_probs=pred_probs)
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
        new_lab = Datalab(data=dataset, label_name=label_name)
        if datalab_flag:
            if features_flag:
                new_lab.find_issues(features=embeddings[permutation])
            else:
                new_lab.find_issues(pred_probs=pred_probs[permutation])
            issues_perm, summary_perm, info_perm = (
                new_lab.get_issues("non_iid"),
                new_lab.get_issue_summary("non_iid"),
                new_lab.get_info("non_iid"),
            )
        else:
            if features_flag:
                new_issue_manager.find_issues(features=embeddings[permutation])
            else:
                new_issue_manager.find_issues(pred_probs=pred_probs[permutation])
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

    @pytest.mark.parametrize("datalab_flag, features_flag",
                             [(True, True), (False, False), (False, True), (True, False)])
    def test_report(self, issue_manager, embeddings, lab, pred_probs, datalab_flag, features_flag):
        np.random.seed(SEED)
        if datalab_flag:
            if features_flag:
                lab.find_issues(features=embeddings)
            else:
                lab.find_issues(pred_probs=pred_probs)
            lab.data_issues.issue_summary = lab.data_issues.issue_summary[
                lab.data_issues.issue_summary["issue_type"] == "non_iid"]
            reporter = report_factory(lab._imagelab)(
                data_issues=lab.data_issues,
                verbosity=0,
                imagelab=lab._imagelab,
            )
            report = reporter.get_report(num_examples=len(embeddings) if features_flag else len(pred_probs))
        else:
            if features_flag:
                issue_manager.find_issues(features=embeddings)
            else:
                issue_manager.find_issues(pred_probs=pred_probs)
            report = issue_manager.report(
                issues=issue_manager.issues,
                summary=issue_manager.summary,
                info=issue_manager.info,
            )

        assert isinstance(report, str)
        if datalab_flag:
            assert ('Here is a summary of the different kinds of issues found in the data:\n'
                    '\n'
                    'issue_type  num_issues\n'
                    '   non_iid') in report
        else:
            assert (
                       "---------------------- non_iid issues ----------------------\n\n"
                       "Number of examples with this issue:"
                   ) in report

        if datalab_flag:
            if features_flag:
                lab.find_issues(features=embeddings)
            else:
                lab.find_issues(pred_probs=pred_probs)
            lab.data_issues.issue_summary = lab.data_issues.issue_summary[
                lab.data_issues.issue_summary["issue_type"] == "non_iid"]
            reporter = report_factory(lab._imagelab)(
                data_issues=lab.data_issues,
                verbosity=3,
                imagelab=lab._imagelab,
            )
            report = reporter.get_report(num_examples=len(embeddings) if features_flag else len(pred_probs))
        else:
            if features_flag:
                issue_manager.find_issues(features=embeddings)
            else:
                issue_manager.find_issues(pred_probs=pred_probs)
            report = issue_manager.report(
                issues=issue_manager.issues,
                summary=issue_manager.summary,
                info=issue_manager.info,
                verbosity=3,
            )

        assert "Additional Information: " in report

    @pytest.mark.parametrize("datalab_flag, features_flag",
                             [(True, True), (False, False), (False, True), (True, False)])
    def test_collect_info(self, issue_manager, embeddings, pred_probs, lab, datalab_flag, features_flag):
        """Test some values in the info dict.

        Mainly focused on the nearest neighbor info.
        """

        if datalab_flag:
            if features_flag:
                lab.find_issues(features=embeddings)
            else:
                lab.find_issues(pred_probs=pred_probs)

            info = lab.get_info("non_iid")
        else:
            if features_flag:
                issue_manager.find_issues(features=embeddings)
            else:
                issue_manager.find_issues(pred_probs=pred_probs)
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

        # using pred_probs
        # normalizing pred_probs (0 to 1)
        pred_probs = embeddings / (np.max(embeddings) - np.min(embeddings))
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
        issue_manager.find_issues(pred_probs=pred_probs)
        p_value = issue_manager.info["p-value"]

        # Run again with the same seed
        issue_manager.find_issues(pred_probs=pred_probs)
        p_value2 = issue_manager.info["p-value"]

        assert p_value > 0.0
        if seed is not None or seed == "default":
            assert p_value == p_value2
        else:
            assert p_value != p_value2
