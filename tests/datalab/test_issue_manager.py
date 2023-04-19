import numpy as np
import pandas as pd
import pytest

from cleanlab.datalab.issue_manager import (
    IssueManager,
    LabelIssueManager,
    OutlierIssueManager,
    NearDuplicateIssueManager,
    NonIIDIssueManager,
)
from cleanlab.datalab.factory import REGISTRY, register
from cleanlab.outlier import OutOfDistribution

SEED = 42


class TestLabelIssueManager:
    @pytest.fixture
    def issue_manager(self, lab):
        return LabelIssueManager(datalab=lab)

    def test_find_issues(self, pred_probs, issue_manager):
        """Test that the find_issues method works."""
        issue_manager.find_issues(pred_probs=pred_probs)
        issues, summary, info = issue_manager.issues, issue_manager.summary, issue_manager.info
        assert isinstance(issues, pd.DataFrame), "Issues should be a dataframe"

        assert isinstance(summary, pd.DataFrame), "Summary should be a dataframe"
        assert summary["issue_type"].values[0] == "label"
        assert pytest.approx(summary["score"].values[0]) == 0.6

        assert isinstance(info, dict), "Info should be a dict"
        info_keys = info.keys()
        expected_keys = [
            "num_label_issues",
            "average_label_quality",
            "confident_joint",
            "classes_by_label_quality",
            "overlapping_classes",
            "py",
            "noise_matrix",
            "inverse_noise_matrix",
        ]
        assert all(
            [key in info_keys for key in expected_keys]
        ), f"Info should have the right keys, but is missing {set(expected_keys) - set(info_keys)}"

    def test_find_issues_with_kwargs(self, pred_probs, issue_manager):
        issue_manager.find_issues(pred_probs=pred_probs, thresholds=[0.2, 0.3, 0.1])

    def test_init_with_clean_learning_kwargs(self, lab, issue_manager):
        """Test that the init method can provide kwargs to the CleanLearning constructor."""
        new_issue_manager = LabelIssueManager(
            datalab=lab,
            clean_learning_kwargs={"cv_n_folds": 10},
        )
        cv_n_folds = [im.cl.cv_n_folds for im in [issue_manager, new_issue_manager]]
        assert cv_n_folds == [5, 10], "Issue manager should have the right attributes"

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"asymmetric": True}],
        ids=["No kwargs", "asymmetric=True"],
    )
    def test_get_summary_parameters(self, issue_manager, kwargs, monkeypatch):
        mock_health_summary_parameters = {
            "labels": [1, 0, 2],
            "asymmetric": False,
            "class_names": ["a", "b", "c"],
            "num_examples": 3,
            "joint": [1 / 3, 1 / 3, 1 / 3],
            "confident_joint": [1 / 3, 1 / 3, 1 / 3],
            "multi_label": False,
        }
        monkeypatch.setattr(
            issue_manager.datalab, "_labels", mock_health_summary_parameters["labels"]
        )
        pred_probs = np.random.rand(3, 3)
        monkeypatch.setattr(
            issue_manager, "health_summary_parameters", mock_health_summary_parameters
        )
        summary_parameters = issue_manager._get_summary_parameters(pred_probs=pred_probs, **kwargs)
        expected_parameters = {
            "confident_joint": [1 / 3, 1 / 3, 1 / 3],
            "asymmetric": False,
            "class_names": ["a", "b", "c"],
        }
        expected_parameters.update(kwargs)
        assert summary_parameters == expected_parameters

        # Test missing "confident_joint" key
        mock_health_summary_parameters.pop("confident_joint")
        monkeypatch.setattr(
            issue_manager, "health_summary_parameters", mock_health_summary_parameters
        )
        summary_parameters = issue_manager._get_summary_parameters(pred_probs=pred_probs, **kwargs)
        expected_parameters = {
            "joint": [1 / 3, 1 / 3, 1 / 3],
            "num_examples": 3,
            "asymmetric": False,
            "class_names": ["a", "b", "c"],
        }
        expected_parameters.update(kwargs)
        assert summary_parameters == expected_parameters

        # Test missing "joint" key
        mock_health_summary_parameters.pop("joint")
        monkeypatch.setattr(
            issue_manager, "health_summary_parameters", mock_health_summary_parameters
        )
        summary_parameters = issue_manager._get_summary_parameters(pred_probs=pred_probs, **kwargs)
        expected_parameters = {
            "pred_probs": pred_probs,
            "labels": [1, 0, 2],
            "asymmetric": False,
            "class_names": ["a", "b", "c"],
        }
        expected_parameters.update(kwargs)
        assert np.all(summary_parameters["pred_probs"] == expected_parameters["pred_probs"])
        summary_parameters.pop("pred_probs")
        expected_parameters.pop("pred_probs")
        assert summary_parameters == expected_parameters


class TestOutlierIssueManager:
    @pytest.fixture
    def embeddings(self, lab):
        np.random.seed(SEED)
        embeddings_array = 0.5 + 0.1 * np.random.rand(lab.get_info("statistics")["num_examples"], 2)
        embeddings_array[4, :] = -1
        return {"embedding": embeddings_array}

    @pytest.fixture
    def issue_manager(self, lab):
        return OutlierIssueManager(datalab=lab, k=3)

    @pytest.fixture
    def issue_manager_with_threshold(self, lab):
        return OutlierIssueManager(datalab=lab, k=2, threshold=0.5)

    def test_init(self, issue_manager, issue_manager_with_threshold):
        assert isinstance(issue_manager.ood, OutOfDistribution)
        assert issue_manager.ood.params["k"] == 3
        assert issue_manager.threshold == None

        assert issue_manager_with_threshold.ood.params["k"] == 2
        assert issue_manager_with_threshold.threshold == 0.5

    def test_find_issues(self, issue_manager, issue_manager_with_threshold, embeddings):
        issue_manager.find_issues(features=embeddings["embedding"])
        issues, summary, info = issue_manager.issues, issue_manager.summary, issue_manager.info
        expected_issue_mask = np.array([False] * 4 + [True])
        assert np.all(
            issues["is_outlier_issue"] == expected_issue_mask
        ), "Issue mask should be correct"
        assert summary["issue_type"][0] == "outlier"
        assert summary["score"][0] == pytest.approx(expected=0.7732146, abs=1e-7)

        assert info.get("knn", None) is not None, "Should have knn info"
        assert issue_manager.threshold == pytest.approx(expected=0.1004, abs=1e-4)

        issue_manager_with_threshold.find_issues(features=embeddings["embedding"])

    def test_find_issues_with_pred_probs(self, issue_manager):
        pred_probs = np.array(
            [
                [0.1, 0.85, 0.05],
                [0.15, 0.8, 0.05],
                [0.05, 0.05, 0.9],
                [0.1, 0.05, 0.85],
                [0.1, 0.65, 0.25],
            ]
        )
        issue_manager.find_issues(pred_probs=pred_probs)
        issues, summary, info = issue_manager.issues, issue_manager.summary, issue_manager.info
        expected_issue_mask = np.array([False] * 4 + [True])
        assert np.all(
            issues["is_outlier_issue"] == expected_issue_mask
        ), "Issue mask should be correct"
        assert summary["issue_type"][0] == "outlier"
        assert summary["score"][0] == pytest.approx(expected=0.151, abs=1e-3)

        assert issue_manager.threshold == pytest.approx(expected=0.0421, abs=1e-4)

        assert np.all(
            info.get("confident_thresholds", None) == [0.1, 0.825, 0.575]
        ), "Should have confident_joint info"

    def test_find_issues_with_different_iqr_scale(self, issue_manager, embeddings):
        issue_manager.find_issues(features=embeddings["embedding"], iqr_scale=0.5)
        issues, summary, info = issue_manager.issues, issue_manager.summary, issue_manager.info
        expected_issue_mask = np.array([False] * 4 + [True])
        assert np.all(
            issues["is_outlier_issue"] == expected_issue_mask
        ), "Issue mask should be correct"
        assert summary["issue_type"][0] == "outlier"
        assert summary["score"][0] == pytest.approx(expected=0.7732146, abs=1e-7)

        # Lower iqr_scale should lower the threshold
        assert issue_manager.threshold == pytest.approx(expected=0.0839, abs=1e-4)

    def test_report(self, issue_manager):
        pred_probs = np.array(
            [
                [0.1, 0.85, 0.05],
                [0.15, 0.8, 0.05],
                [0.05, 0.05, 0.9],
                [0.1, 0.05, 0.85],
                [0.1, 0.65, 0.25],
            ]
        )
        issue_manager.find_issues(pred_probs=pred_probs)
        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info=issue_manager.info,
        )
        assert isinstance(report, str)
        assert (
            "---------------------- outlier issues ----------------------\n\n"
            "Number of examples with this issue:"
        ) in report

        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info=issue_manager.info,
            verbosity=3,
        )
        assert "Additional Information: " in report

        # Mock some vector and matrix values in the info dict
        mock_info = issue_manager.info
        vector = np.array([1, 2, 3, 4, 5, 6])
        matrix = np.array([[i for i in range(20)] for _ in range(10)])
        df = pd.DataFrame(matrix)
        mock_list = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        mock_dict = {"a": 1, "b": 2, "c": 3}
        mock_info["vector"] = vector
        mock_info["matrix"] = matrix
        mock_info["list"] = mock_list
        mock_info["dict"] = mock_dict
        mock_info["df"] = df

        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info={**issue_manager.info, **mock_info},
            verbosity=4,
        )
        assert "Additional Information: " in report
        assert "vector: [1, 2, 3, 4, '...']" in report
        assert f"matrix: array of shape {matrix.shape}\n[[ 0 " in report
        assert "list: [9, 8, 7, 6, '...']" in report
        assert 'dict:\n{\n    "a": 1,\n    "b": 2,\n    "c": 3\n}' in report
        assert "df:" in report

        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info={**issue_manager.info, **mock_info},
            verbosity=2,
        )
        assert "Additional Information: " in report
        assert "vector: [1, 2, 3, 4, '...']" not in report
        assert f"matrix: array of shape {matrix.shape}\n[[ 0 " not in report
        assert "list: [9, 8, 7, 6, '...']" not in report
        assert 'dict:\n{\n    "a": 1,\n    "b": 2,\n    "c": 3\n}' not in report
        assert "df:" not in report

    def test_collect_info(self, issue_manager, embeddings):
        """Test some values in the info dict.

        Mainly focused on the nearest neighbor info.
        """

        issue_manager.find_issues(features=embeddings["embedding"])
        info = issue_manager.info

        nearest_neighbors = info["nearest_neighbor"]
        distances_to_nearest_neighbor = info["distance_to_nearest_neighbor"]

        assert nearest_neighbors == [3, 0, 3, 0, 2], "Nearest neighbors should be correct"

        assert pytest.approx(distances_to_nearest_neighbor, abs=1e-3) == [
            0.033,
            0.05,
            0.072,
            0.033,
            2.143,
        ], "Distances to nearest neighbor should be correct"


class TestNearDuplicateIssueManager:
    @pytest.fixture
    def embeddings(self, lab):
        np.random.seed(SEED)
        embeddings_array = 0.5 + 0.1 * np.random.rand(lab.get_info("statistics")["num_examples"], 2)
        embeddings_array[4, :] = (
            embeddings_array[3, :] + np.random.rand(embeddings_array.shape[1]) * 0.001
        )
        return {"embedding": embeddings_array}

    @pytest.fixture
    def issue_manager(self, lab, embeddings, monkeypatch):
        mock_data = lab.data.from_dict({**lab.data.to_dict(), **embeddings})
        monkeypatch.setattr(lab, "data", mock_data)
        return NearDuplicateIssueManager(
            datalab=lab,
            metric="euclidean",
            k=2,
        )

    def test_init(self, lab, issue_manager):
        assert issue_manager.datalab == lab
        assert issue_manager.metric == "euclidean"
        assert issue_manager.k == 2
        assert issue_manager.threshold == 0.2

        issue_manager = NearDuplicateIssueManager(
            datalab=lab,
            threshold=0.1,
        )
        assert issue_manager.threshold == 0.1

    def test_find_issues(self, issue_manager, embeddings):
        issue_manager.find_issues(features=embeddings["embedding"])
        issues, summary, info = issue_manager.issues, issue_manager.summary, issue_manager.info
        expected_issue_mask = np.array([False] * 3 + [True] * 2)
        assert np.all(
            issues["is_near_duplicate_issue"] == expected_issue_mask
        ), "Issue mask should be correct"
        assert summary["issue_type"][0] == "near_duplicate"
        assert summary["score"][0] == pytest.approx(expected=0.03122489, abs=1e-7)

        assert (
            info.get("near_duplicate_sets", None) is not None
        ), "Should have sets of near duplicates"

        new_issue_manager = NearDuplicateIssueManager(
            datalab=issue_manager.datalab,
            metric="euclidean",
            k=2,
            threshold=0.1,
        )
        new_issue_manager.find_issues(features=embeddings["embedding"])

    def test_report(self, issue_manager, embeddings):
        issue_manager.find_issues(features=embeddings["embedding"])
        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info=issue_manager.info,
        )
        assert isinstance(report, str)
        assert (
            "------------------ near_duplicate issues -------------------\n\n"
            "Number of examples with this issue:"
        ) in report

        report = issue_manager.report(
            issues=issue_manager.issues,
            summary=issue_manager.summary,
            info=issue_manager.info,
            verbosity=3,
        )
        assert "Additional Information: " in report


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
        expected_sorted_issue_mask = np.array([False] * len(embeddings))
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
        assert summary_perm["score"][0] == pytest.approx(expected=0.310207044, abs=1e-7)
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


def test_register_custom_issue_manager(monkeypatch):
    import io
    import sys

    assert "foo" not in REGISTRY

    @register
    class Foo(IssueManager):
        issue_name = "foo"

        def find_issues(self):
            pass

    assert "foo" in REGISTRY
    assert REGISTRY["foo"] == Foo

    # Reregistering should overwrite the existing class, put print a warning

    monkeypatch.setattr("sys.stdout", io.StringIO())

    @register
    class NewFoo(IssueManager):
        issue_name = "foo"

        def find_issues(self):
            pass

    assert "foo" in REGISTRY
    assert REGISTRY["foo"] == NewFoo
    assert all(
        [
            text in sys.stdout.getvalue()
            for text in ["Warning: Overwriting existing issue manager foo with ", "NewFoo"]
        ]
    ), "Should print a warning"
