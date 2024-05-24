import numpy as np
import pytest

from cleanlab import Datalab
from cleanlab.datalab.internal.issue_finder import IssueFinder
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

    @pytest.mark.parametrize("key", ["pred_probs", "features", "knn_graph"])
    def test_get_available_issue_types_no_kwargs(self, issue_finder, key):
        expected_issue_types = {"class_imbalance": {}}
        issue_types = issue_finder.get_available_issue_types(**{key: None})
        assert (
            issue_types == expected_issue_types
        ), "Only class_imbalance issue type for classification requires no kwargs"

    @pytest.mark.parametrize(
        "issue_types",
        [
            {"label": {}},
            {"label": {"some_arg": "some_value"}},
            {"label": {"some_arg": "some_value"}, "outlier": {}},
            {"label": {}, "outlier": {}, "some_issue_type": {"some_arg": "some_value"}},
            {},
        ],
    )
    def test_get_available_issue_types_with_issue_types(self, issue_finder, issue_types):
        available_issue_types = issue_finder.get_available_issue_types(issue_types=issue_types)
        assert (
            available_issue_types == issue_types
        ), f"Failed to get available issue types with issue_types={issue_types}"

    @pytest.mark.parametrize(
        "keys, should_contain_underperforming_group",
        [
            # Test cases where 'pred_probs' is not provided, should all give False
            (["features"], False),
            (["knn_graph"], False),
            (["cluster_ids"], False),
            (["features", "knn_graph"], False),
            (["features", "cluster_ids"], False),
            (["knn_graph", "cluster_ids"], False),
            (["features", "knn_graph", "cluster_ids"], False),
            # Test cases where 'pred_probs' is provided should all give True
            (["pred_probs", "features"], True),
            (["pred_probs", "knn_graph"], True),
            (["pred_probs", "cluster_ids"], True),
            (["pred_probs", "features", "knn_graph"], True),
            (["pred_probs", "features", "cluster_ids"], True),
            (["pred_probs", "knn_graph", "cluster_ids"], True),
            (["pred_probs", "features", "knn_graph", "cluster_ids"], True),
            # only if other required keys are provided
            (["pred_probs"], False),
        ],
        ids=lambda v: (
            f"keys={v} "
            if isinstance(v, list)
            else ("> available" if v is True else "> unavailable")
        ),
    )
    # Some warnings about preferring cluster_ids over knn_graph, or knn_graph over features can be ignored
    @pytest.mark.filterwarnings(r"ignore:.*will (likely )?prefer.*:UserWarning")
    # No other warnings should be allowed
    @pytest.mark.filterwarnings("error")
    def test_underperforming_group_availability_issue_1065(
        self, issue_finder, keys, should_contain_underperforming_group
    ):
        """
        Tests the availability of the 'underperforming_group' issue type based on the presence of 'pred_probs' and other required keys in the supplied arguments.

        This test addresses issue #1065, where the mapping that decides which issue types to run based on the supplied arguments is incorrect.
        Specifically, the 'underperforming_group' check should only be executed if 'pred_probs' and another required key are included in the supplied arguments.
        See: https://github.com/cleanlab/cleanlab/issues/1065.

        Parameters
        ----------
        keys : list
            A list of keys to be included in the kwargs.
        should_contain_underperforming_group : bool
            A flag indicating whether the 'underperforming_group' issue type should be present in the available issue types.

        Scenarios
        ---------
        Various combinations of 'features', 'pred_probs', 'knn_graph', and 'cluster_ids' are tested.

        Asserts
        -------
        Ensures 'underperforming_group' is in the available issue types if 'pred_probs' and another required key are provided.
        Ensures 'underperforming_group' is not in the available issue types if the required conditions are not met.
        """
        mock_value = object()  # Mock value to simulate presence of the required keys
        kwargs = {key: mock_value for key in keys}

        available_issue_types = issue_finder.get_available_issue_types(**kwargs)
        if should_contain_underperforming_group:
            assert (
                "underperforming_group" in available_issue_types
            ), "underperforming_group should be available if 'pred_probs' and another required key are provided"
        else:
            assert (
                "underperforming_group" not in available_issue_types
            ), "underperforming_group should not be available if the required conditions are not met"

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

        ## Test availability of underperforming_group issue type
        only_features_available = {"features": np.random.random((10, 2))}
        available_issue_types = issue_finder.get_available_issue_types(**only_features_available)
        fail_msg = "underperforming_group should not be available if 'pred_probs' is not provided"
        assert "underperforming_group" not in available_issue_types, fail_msg
        features_and_pred_probs_available = {
            **only_features_available,
            "pred_probs": np.random.random((10, 2)),
        }
        available_issue_types = issue_finder.get_available_issue_types(
            **features_and_pred_probs_available
        )
        fail_msg = "underperforming_group should be available if 'pred_probs' is provided"
        assert "underperforming_group" in available_issue_types, fail_msg

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
        expected_issue_types = {}

        # Test with no kwargs
        for key in ["pred_probs", "features", "knn_graph"]:
            issue_types = issue_finder.get_available_issue_types(**{key: None})
            assert (
                issue_types == expected_issue_types
            ), "No issue type for regression requires no kwargs"

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
