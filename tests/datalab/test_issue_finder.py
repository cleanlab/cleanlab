import pytest
import numpy as np

from cleanlab.datalab.internal.issue_finder import IssueFinder

from cleanlab import Datalab


class TestIssueFinder:
    task = "classification"

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

    def test_regression(self):
        N = 30
        K = 2
        y = np.random.randint(0, K, size=N)
        lab = Datalab(data={"y": y}, label_name="y", task="regression")
        assert set(lab.list_default_issue_types()) == set(
            ["label", "outlier", "near_duplicate", "non_iid"]
        )
        assert set(lab.list_possible_issue_types()) == set(
            ["label", "outlier", "near_duplicate", "non_iid"]
        )
