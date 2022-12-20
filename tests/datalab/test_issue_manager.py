import numpy as np
import pandas as pd
import pytest

from cleanlab.experimental.datalab.issue_manager import LabelIssueManager


@pytest.mark.parametrize(
    "issue_manager_class",
    [LabelIssueManager],
    ids=["LabelIssueManager"],
)
class TestIssueManager:
    @pytest.fixture
    def issue_manager(self, lab, issue_manager_class):
        return issue_manager_class(datalab=lab)

    def test_init(self, lab, issue_manager_class):
        """Test that the init method works."""
        issue_manager = issue_manager_class(datalab=lab)
        assert issue_manager.datalab == lab

    def test_find_issues(self, pred_probs, issue_manager):
        """Test that the find_issues method works."""
        issue_manager.find_issues(pred_probs=pred_probs)
        issues, summary, info = issue_manager.issues, issue_manager.summary, issue_manager.info
        assert isinstance(issues, pd.DataFrame), "Issues should be a dataframe"
        is_issue_key = "_".join(["is", issue_manager.issue_key, "issue"])
        assert is_issue_key in issues.columns, "Issues should have an is_<issue_key>_issue column"

        assert isinstance(summary, pd.DataFrame), "Summary should be a dataframe"
        assert all(
            [col in summary.columns for col in ["issue_type", "score"]]
        ), "Summary should have issue_type and score columns"

        assert isinstance(info, dict), "Info should be a dict"


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
