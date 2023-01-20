import numpy as np
import pandas as pd
import pytest

from cleanlab.experimental.datalab.issue_manager import (
    LabelIssueManager,
    OutOfDistributionIssueManager,
)
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

    def test_init_with_clean_learning_kwargs(self, lab, issue_manager):
        """Test that the init method can providee kwargs to the CleanLearning constructor."""
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


class TestOutOfDistributionIssueManager:
    @pytest.fixture
    def embeddings(self, lab):
        np.random.seed(SEED)
        embeddings_array = 0.5 + 0.1 * np.random.rand(lab.get_info("data", "num_examples"), 2)
        embeddings_array[4, :] = -1
        return {"embedding": embeddings_array}

    @pytest.fixture
    def issue_manager(self, lab, embeddings, monkeypatch):
        mock_data = lab.data.from_dict({**lab.data.to_dict(), **embeddings})
        monkeypatch.setattr(lab, "data", mock_data)
        return OutOfDistributionIssueManager(datalab=lab, ood_kwargs={"params": {"k": 3}})

    def test_init(self, issue_manager):
        assert isinstance(issue_manager.ood, OutOfDistribution)
        assert issue_manager.ood.params["k"] == 3

    def test_extract_embeddings(self, issue_manager, embeddings):
        extracted_embeddings = issue_manager._extract_embeddings(
            columns="embedding", format_kwargs={"dtype": np.float64}
        )
        assert isinstance(extracted_embeddings, np.ndarray), "Should be a numpy array"
        assert np.all(extracted_embeddings == embeddings["embedding"]), "Embeddings should match"

    def test_find_issues(self, issue_manager):
        issue_manager.find_issues(features="embedding")
        issues, summary, info = issue_manager.issues, issue_manager.summary, issue_manager.info
        expected_issue_mask = np.array([False] * 4 + [True])
        assert np.all(
            issues["is_outlier_issue"] == expected_issue_mask
        ), "Issue mask should be correct"
        assert summary["issue_type"][0] == "outlier"
        assert summary["score"][0] == pytest.approx(expected=0.8257756, rel=1e-7)

        assert info.get("knn", None) is not None, "Should have knn info"

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
        assert summary["score"][0] == pytest.approx(expected=0.151, rel=1e-3)

        assert np.all(
            info.get("confident_thresholds", None) == [0.1, 0.825, 0.575]
        ), "Should have confident_joint info"
