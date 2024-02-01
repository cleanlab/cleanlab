import numpy as np
import pandas as pd
import pytest

from cleanlab import Datalab
from cleanlab.datalab.internal.issue_manager.regression.label import RegressionLabelIssueManager
from cleanlab.datalab.internal.task import Task


def ground_truth_target_function(x):
    return 10 * x + 1


class TestRegressionLabelIssueManager:
    def test_manager_found_in_registry(self):
        from cleanlab.datalab.internal.issue_manager_factory import REGISTRY

        error_msg = (
            "RegressionLabelIssueManager should be registered to the regression task as 'label'"
        )
        assert REGISTRY[Task.REGRESSION].get("label") == RegressionLabelIssueManager, error_msg

    @pytest.fixture
    def features(self):
        # 1 feature, 7 points
        return np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5]).reshape(-1, 1)

    @pytest.fixture
    def regression_lab(self, features):
        y = ground_truth_target_function(features)
        # Flip the sign of the point x=0.4
        y[features == 0.4] *= -1
        y = y.ravel()
        return Datalab({"y": y}, label_name="y", task="regression")

    @pytest.fixture
    def issue_manager(self, regression_lab):
        return RegressionLabelIssueManager(datalab=regression_lab)

    def test_find_issues_with_features(self, issue_manager, features):
        issue_manager.find_issues(features=features)
        issues = issue_manager.issues
        assert isinstance(issues, pd.DataFrame), "Issues should be a dataframe"
        expected_issue_mask = features.ravel() == 0.4
        assert sum(expected_issue_mask) == 1, "There should be exactly one issue"

        np.testing.assert_array_equal(issues["is_label_issue"].values, expected_issue_mask)
        # Assert that he minimum score "label_score" is at the correct index
        index_of_error = np.where(expected_issue_mask)[0][0]
        assert issues["label_score"].values.argmin() == index_of_error

    def test_init_with_model(self, issue_manager):
        from sklearn.neighbors import KNeighborsRegressor

        model = KNeighborsRegressor(n_neighbors=2)
        assert issue_manager.cl.model != model

        # Passing in a model to the constructor should set the cl.model field
        clean_learning_kwargs = {"model": model}
        lab = issue_manager.datalab
        new_issue_manager = RegressionLabelIssueManager(
            datalab=lab, clean_learning_kwargs=clean_learning_kwargs
        )
        assert new_issue_manager.cl.model == model

    @pytest.fixture
    def predictions(self, features):
        y_ground_truth = ground_truth_target_function(features).ravel()
        noise = 0.1 * np.random.randn(len(y_ground_truth))
        return y_ground_truth + noise

    def test_raises_find_issues_error_without_valid_inputs(self, issue_manager):
        with pytest.raises(ValueError) as e:
            expected_error_msg = (
                "Regression requires numerical `features` or `predictions` "
                "to be passed in as an argument to `find_issues`."
            )
            issue_manager.find_issues()
            assert expected_error_msg in str(e)

    def test_find_issue_with_predictions(self, issue_manager, features, predictions):
        issue_manager.find_issues(predictions=predictions)
        issues = issue_manager.issues
        assert isinstance(issues, pd.DataFrame), "Issues should be a dataframe"
        expected_issue_mask = features.ravel() == 0.4
        assert sum(expected_issue_mask) == 1, "There should be exactly one issue"

        np.testing.assert_array_equal(issues["is_label_issue"].values, expected_issue_mask)
        # Assert that he minimum score "label_score" is at the correct index
        index_of_error = np.where(expected_issue_mask)[0][0]
        assert issues["label_score"].values.argmin() == index_of_error


class TestRegressionLabelIssueManagerIntegration:
    """This class contains tests for the find_issues method with a CleanLearning
    object that behaves deterministically. This is useful to run a "regression"-test on
    the results computed by the find_issues method.
    The test dataset is a random toy regression dataset with 5 features and 100 samples.
    The ground truth is a linear function of the first feature plus a bias defined in the
    class attribute BIAS.
    The ground truth is used to emulate a perfect model and compute the expected score
    for the label issue detection. The gaussian noise contributes to lower label quality
    scores.
    """

    BIAS = 1.0

    @pytest.fixture()
    def regression_dataset(self):
        """For integration tests, a simple regression dataset is simpler than
        a tiny, hand-crafted one."""
        from sklearn.datasets import make_regression

        # Return coefficients as well for testing purposes,
        # interpret as ground truth
        X, y, coef = make_regression(
            n_samples=100,
            n_features=5,
            n_informative=1,
            n_targets=1,
            bias=self.BIAS,
            noise=0.1,
            random_state=0,
            coef=True,
        )
        return X, y, coef

    @pytest.fixture()
    def issue_manager(self, regression_dataset):
        _, y, _ = regression_dataset
        lab = Datalab({"y": y}, label_name="y", task="regression")
        return RegressionLabelIssueManager(datalab=lab, clean_learning_kwargs={"seed": 0})

    def test_find_issues_with_features(self, regression_dataset, issue_manager):
        X, _, _ = regression_dataset
        issue_manager.find_issues(features=X)
        summary = issue_manager.summary
        assert np.isclose(summary["score"], 0.262423, atol=1e-5)

    def test_find_issues_with_predictions(self, regression_dataset, issue_manager):
        X, _, coef = regression_dataset
        y_pred = X @ coef + self.BIAS
        issue_manager.find_issues(predictions=y_pred)
        summary = issue_manager.summary
        assert np.isclose(summary["score"], 0.361287, atol=1e-5)
