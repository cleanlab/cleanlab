import pytest
import numpy as np

from cleanlab.datalab.internal.issue_finder import IssueFinder

from cleanlab import Datalab
from cleanlab.datalab.internal.task import Task


class TestIssueFinder:
    task = Task.CLASSIFICATION

    @pytest.fixture
    def lab(self):
        N = 30
        K = 2
        y = np.random.randint(0, K, size=N)
        lab = Datalab(data={"y": y}, label_name="y")
        return lab

    @pytest.fixture
    def issue_finder(self, lab):
        return IssueFinder(datalab=lab, task=self.task)

    def test_init(self, issue_finder):
        assert issue_finder.verbosity == 1

    def test_get_available_issue_types(self, issue_finder):
        expected_issue_types = {"class_imbalance": {}}
        # Test with no kwargs, no issue type expected to be returned
        for key in ["pred_probs", "features", "knn_graph"]:
            issue_types = issue_finder.get_available_issue_types(**{key: None})
            assert (
                issue_types == expected_issue_types
            ), "Only class_imbalance issue type for classification requires no kwargs"

        # Test with only issue_types, input should be
        issue_types_dicts = [
            {"label": {}},
            {"label": {"some_arg": "some_value"}},
            {"label": {"some_arg": "some_value"}, "outlier": {}},
            {"label": {}, "outlier": {}, "some_issue_type": {"some_arg": "some_value"}},
            {},
        ]
        for issue_types in issue_types_dicts:
            available_issue_types = issue_finder.get_available_issue_types(issue_types=issue_types)
            fail_msg = f"Failed to get available issue types with issue_types={issue_types}"
            assert available_issue_types == issue_types, fail_msg

    def test_find_issues(self, issue_finder, lab):
        N = len(lab.data)
        K = lab.get_info("statistics")["num_classes"]
        X = np.random.rand(N, 2)
        pred_probs = np.random.rand(N, K)
        pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)

        data_issues = lab.data_issues
        assert data_issues.issues.empty

        issue_finder.find_issues(
            features=X,
            pred_probs=pred_probs,
        )

        assert not data_issues.issues.empty

    def test_validate_issue_types_dict(self, issue_finder, monkeypatch):
        issue_types = {
            "issue_type_1": {f"arg_{i}": f"value_{i}" for i in range(1, 3)},
            "issue_type_2": {f"arg_{i}": f"value_{i}" for i in range(1, 4)},
        }
        defaults_dict = issue_types.copy()

        issue_types["issue_type_2"][
            "arg_2"
        ] = "another_value_2"  # Should be in default, but not affect the test
        issue_types["issue_type_2"][
            "arg_4"
        ] = "value_4"  # Additional arg not in defaults should be allowed (ignored)

        with monkeypatch.context() as m:
            m.setitem(issue_types, "issue_type_1", {})
            with pytest.raises(ValueError) as e:
                issue_finder._validate_issue_types_dict(issue_types, defaults_dict)
            assert all([string in str(e.value) for string in ["issue_type_1", "arg_1", "arg_2"]])


class TestRegressionIssueFinder:
    task = "regression"

    @pytest.fixture
    def lab(self):
        N = 30
        K = 2
        y = np.random.randint(0, K, size=N)
        lab = Datalab(data={"y": y}, label_name="y", task=self.task)
        return lab

    @pytest.fixture
    def issue_finder(self, lab):
        return IssueFinder(datalab=lab, task=Task.from_str(self.task))

    def test_get_available_issue_types(self, issue_finder):
        expected_issue_types = {"label": {}}

        # Test with no kwargs
        for key in ["pred_probs", "features", "knn_graph"]:
            issue_types = issue_finder.get_available_issue_types(**{key: None})
            assert (
                issue_types == expected_issue_types
            ), "Regression should only support label issues"

        # Test with issue_types:
        issue_types_dicts = [
            {"label": {}},
            {"label": {"some_arg": "some_value"}},
            {"label": {"some_arg": "some_value"}, "outlier": {}},
            {},
        ]
        supported_issue_types = ["label"]
        for issue_types in issue_types_dicts:
            available_issue_types = issue_finder.get_available_issue_types(issue_types=issue_types)
            fail_msg = f"Failed to get available issue types with issue_types={issue_types}"
            assert available_issue_types == issue_types, fail_msg

        # Test with all kwargs
        kwargs = {k: k for k in ["pred_probs", "features", "knn_graph"]}
        kwargs["issue_types"] = {"label": {}}
        available_issue_types = issue_finder.get_available_issue_types(**kwargs)
        assert available_issue_types == {
            "label": {
                "predictions": "pred_probs",  # Expect the ModelOutput.argument class variable to replace the key
                "features": "features",
            },
        }
