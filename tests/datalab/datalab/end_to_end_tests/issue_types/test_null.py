import numpy as np
import pandas as pd
import pytest
from cleanlab import Datalab
from cleanlab.datalab.internal.issue_manager.null import NullIssueManager


@pytest.fixture
def setup_test_environment():
    """Setup the basic test environment with a Datalab instance and a NullIssueManager."""
    data = {
        "features": np.array([[1, 2, np.nan], [4, np.nan, 6], [np.nan, 8, 9]]),
        "label": np.array([0, 1, 0]),
    }
    lab = Datalab(data=data, label_name="label")
    issue_manager = NullIssueManager(lab)
    return {"lab": lab, "issue_manager": issue_manager}


class FeatureFactory:
    """Factory class to create features in different formats."""

    @staticmethod
    def create_features(format_type="numpy"):
        """Generate test features in the specified format."""
        strategy = {
            "numpy": FeatureFactory._create_numpy_features,
            "pandas": FeatureFactory._create_pandas_features,
        }
        feature_strategy = strategy.get(format_type)
        if feature_strategy is None:
            raise ValueError(
                f"Invalid format_type to test: {format_type}, must be one of {list(strategy.keys())}"
            )
        return feature_strategy()

    @staticmethod
    def _create_numpy_features():
        """Generate features in numpy format."""
        return np.array([[1, 2, np.nan], [4, np.nan, 6], [np.nan, 8, 9]])

    @staticmethod
    def _create_pandas_features():
        """Generate features in pandas DataFrame format."""
        features = FeatureFactory._create_numpy_features()
        return pd.DataFrame(features, columns=["a", "b", "c"])


@pytest.mark.parametrize("format_type", ["numpy", "pandas"])
def test_null_issue_manager(setup_test_environment, format_type):
    features = FeatureFactory.create_features(format_type)
    null_issue_manager = setup_test_environment["issue_manager"]
    null_issue_manager.find_issues(features=features)
    assert null_issue_manager.info["most_common_issue"] == {
        "pattern": "001",
        "rows_affected": [0],
        "count": 1,
    }


@pytest.mark.parametrize("format_type", ["numpy", "pandas"])
def test_lab_find_issues(setup_test_environment, format_type):
    features = FeatureFactory.create_features(format_type)
    lab = setup_test_environment["lab"]
    lab.find_issues(features=features, issue_types={"null": {}})

    null_issues = lab.get_issues("null")
    expected_null_issues = pd.DataFrame(
        {
            "is_null_issue": [False] * 3,
            "null_score": [2 / 3] * 3,
        }
    )
    pd.testing.assert_frame_equal(null_issues, expected_null_issues)

    most_common_issue = lab.get_info("null")["most_common_issue"]
    assert most_common_issue == {"pattern": "001", "rows_affected": [0], "count": 1}

    column_impact = lab.get_info("null")["column_impact"]
    np.testing.assert_array_equal(column_impact, [1 / 3] * 3)
