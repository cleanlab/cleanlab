# Copyright (C) 2017-2022  Cleanlab Inc.
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with cleanlab.  If not, see <https://www.gnu.org/licenses/>.


import itertools
import os
import pickle
from unittest.mock import MagicMock, Mock, patch
from cleanlab.experimental.datalab.datalab import Datalab

from sklearn.neighbors import NearestNeighbors
from datasets.dataset_dict import DatasetDict
import numpy as np
import pandas as pd

from pathlib import Path

import pytest
import timeit


def test_datalab_invalid_datasetdict(dataset, label_name):
    with pytest.raises(ValueError) as e:
        datadict = DatasetDict({"train": dataset, "test": dataset})
        Datalab(datadict, label_name)  # type: ignore
        assert "Please pass a single dataset, not a DatasetDict." in str(e)


class TestDatalab:
    """Tests for the Datalab class."""

    @pytest.fixture
    def lab(self, dataset, label_name):
        return Datalab(data=dataset, label_name=label_name)

    def test_print(self, lab, capsys):
        # Can print the object
        print(lab)
        captured = capsys.readouterr()
        assert "Datalab\n" == captured.out

    def tmp_path(self):
        # A path for temporarily saving the instance during tests.
        # This is a workaround for the fact that the Datalab class
        # does not have a save method.
        return Path(__file__).parent / "tmp.pkl"

    def test_attributes(self, lab):
        # Has the right attributes
        for attr in ["data", "label_name", "_labels", "info", "issues"]:
            assert hasattr(lab, attr), f"Missing attribute {attr}"

        assert all(lab._labels == np.array([1, 1, 2, 0, 2]))
        assert isinstance(lab.issues, pd.DataFrame), "Issues should by in a dataframe"
        assert isinstance(lab.issue_summary, pd.DataFrame), "Issue summary should be a dataframe"

    def test_get_info(self, lab):
        mock_info: dict = {
            "label": {
                "given_label": [1, 0, 1, 0, 2],
                "predicted_label": [1, 1, 2, 0, 2],
                # get_info("label") adds `class_names` from statistics
            },
            "outlier": {
                "nearest_neighbor": [1, 0, 0, 4, 3],
            },
        }
        mock_info = {**lab.info, **mock_info}
        lab.info = mock_info

        label_info = lab.get_info("label")
        assert label_info["given_label"].tolist() == [4, 3, 4, 3, 5]
        assert label_info["predicted_label"].tolist() == [4, 4, 5, 3, 5]
        assert label_info["class_names"] == [3, 4, 5]

        outlier_info = lab.get_info("outlier")
        assert outlier_info["nearest_neighbor"] == [1, 0, 0, 4, 3]

        assert lab.get_info() == lab.info == mock_info

    def test_get_summary(self, lab, monkeypatch):
        mock_summary: pd.DataFrame = pd.DataFrame(
            {
                "issue_type": ["label", "outlier"],
                "score": [0.5, 0.3],
                "num_issues": [1, 2],
            }
        )
        monkeypatch.setattr(lab, "issue_summary", mock_summary)

        label_summary = lab.get_summary(issue_name="label")
        pd.testing.assert_frame_equal(label_summary, mock_summary.iloc[[0]])

        outlier_summary = lab.get_summary(issue_name="outlier")
        pd.testing.assert_frame_equal(
            outlier_summary, mock_summary.iloc[[1]].reset_index(drop=True)
        )

        summary = lab.get_summary()
        pd.testing.assert_frame_equal(summary, mock_summary)

    def test_get_issues(self, lab, monkeypatch):
        mock_issues: pd.DataFrame = pd.DataFrame(
            {
                "is_label_issue": [True, False, False, True, False],
                "label_score": [0.2, 0.4, 0.6, 0.1, 0.8],
                "is_outlier_issue": [False, True, True, False, True],
                "outlier_score": [0.5, 0.3, 0.1, 0.7, 0.2],
            },
        )
        monkeypatch.setattr(lab, "issues", mock_issues)

        mock_predicted_labels = np.array([0, 1, 2, 1, 2])

        mock_nearest_neighbor = [1, 3, 5, 2, 0]
        mock_distance_to_nearest_neighbor = [0.1, 0.2, 0.3, 0.4, 0.5]

        lab.info.update(
            {
                "label": {
                    "given_label": lab.labels,
                    "predicted_label": mock_predicted_labels,
                },
                "outlier": {
                    "nearest_neighbor": mock_nearest_neighbor,
                    "distance_to_nearest_neighbor": mock_distance_to_nearest_neighbor,
                },
            }
        )

        label_issues = lab.get_issues(issue_name="label")

        expected_label_issues = pd.DataFrame(
            {
                **{key: mock_issues[key] for key in ["is_label_issue", "label_score"]},
                "given_label": [4, 4, 5, 3, 5],
                "predicted_label": [3, 4, 5, 4, 5],
            },
        )

        pd.testing.assert_frame_equal(label_issues, expected_label_issues)

        outlier_issues = lab.get_issues(issue_name="outlier")

        expected_outlier_issues = pd.DataFrame(
            {
                **{key: mock_issues[key] for key in ["is_outlier_issue", "outlier_score"]},
                "nearest_neighbor": mock_nearest_neighbor,
                "distance_to_nearest_neighbor": mock_distance_to_nearest_neighbor,
            },
        )
        pd.testing.assert_frame_equal(outlier_issues, expected_outlier_issues)

        issues = lab.get_issues()
        pd.testing.assert_frame_equal(issues, mock_issues)

    @pytest.mark.parametrize(
        "issue_types",
        [None, {"label": {}}],
        ids=["Default issues", "Only label issues"],
    )
    def test_find_issues(self, lab, pred_probs, issue_types):
        assert lab.issues.empty, "Issues should be empty before calling find_issues"
        assert lab.issue_summary.empty, "Issue summary should be empty before calling find_issues"
        assert lab.info["statistics"]["health_score"] is None
        lab.find_issues(pred_probs=pred_probs, issue_types=issue_types)
        assert not lab.issues.empty, "Issues weren't updated"
        assert not lab.issue_summary.empty, "Issue summary wasn't updated"
        assert (
            lab.info["statistics"]["health_score"] == lab.issue_summary["score"].mean()
        )  # TODO: Avoid re-implementing logic in test

        if issue_types is None:
            # Test default issue types
            columns = lab.issues.columns
            for issue_type in ["label", "outlier"]:
                assert f"is_{issue_type}_issue" in columns
                assert f"{issue_type}_score" in columns

    def test_find_issues_with_custom_hyperparams(self, lab, pred_probs):
        dataset_size = lab.get_info("statistics")["num_examples"]
        embedding_size = 2
        mock_embeddings = np.random.rand(dataset_size, embedding_size)

        ks = [2, 3]
        metrics = ["euclidean", "cosine"]
        combinations = list(itertools.product(ks, metrics))
        for k, metric in combinations:
            knn = NearestNeighbors(n_neighbors=k, metric=metric)
            issue_types = {"outlier": {"knn": knn}}
            lab.find_issues(
                pred_probs=pred_probs,
                features=mock_embeddings,
                issue_types=issue_types,
            )
            assert lab.info["outlier"]["metric"] == metric
            assert lab.info["outlier"]["k"] == k

    def test_validate_issue_types_dict(self, lab, monkeypatch):
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
                lab._validate_issue_types_dict(issue_types, defaults_dict)
            assert all([string in str(e.value) for string in ["issue_type_1", "arg_1", "arg_2"]])

    @pytest.mark.parametrize(
        "defaults_dict",
        [
            {"issue_type_1": {"arg_1": "default_value_1"}},
        ],
    )
    @pytest.mark.parametrize(
        "issue_types",
        [{"issue_type_1": {"arg_1": "value_1", "arg_2": "value_2"}}, {"issue_type_1": {}}],
    )
    def test_set_issue_types(self, lab, issue_types, defaults_dict, monkeypatch):
        """Test that the issue_types dict is set correctly."""
        with monkeypatch.context() as m:
            # Mock the validation method to do nothing
            m.setattr(lab, "_validate_issue_types_dict", lambda x, y: None)
            issue_types_copy = lab._set_issue_types(issue_types, defaults_dict)

            # For each argument in issue_types missing from defaults_dict, it should be added to the defaults dict
            for issue_type, args in issue_types.items():
                missing_args = set(args.keys()) - set(defaults_dict[issue_type].keys())
                for arg in missing_args:
                    assert issue_types_copy[issue_type][arg] == args[arg]

    # Mock the lab.issues dataframe to have some pre-existing issues
    def test_update_issues(self, lab, pred_probs, monkeypatch):
        """If there are pre-existing issues in the lab,
        find_issues should add columns to the issues dataframe for each example.
        """
        mock_issues = pd.DataFrame(
            {
                "is_foo_issue": [False, True, False, False, False],
                "foo_score": [0.6, 0.8, 0.7, 0.7, 0.8],
            }
        )
        monkeypatch.setattr(lab, "issues", mock_issues)
        mock_issue_summary = pd.DataFrame(
            {
                "issue_type": ["foo"],
                "score": [0.72],
                "num_issues": [1],
            }
        )
        monkeypatch.setattr(lab, "issue_summary", mock_issue_summary)

        lab.find_issues(pred_probs=pred_probs, issue_types={"label": {}})
        # Check that the issues dataframe has the right columns
        expected_issues_df = pd.DataFrame(
            {
                "is_foo_issue": mock_issues.is_foo_issue,
                "foo_score": mock_issues.foo_score,
                "is_label_issue": [False, False, False, False, False],
                "label_score": [0.95071431, 0.15601864, 0.60111501, 0.70807258, 0.18182497],
            }
        )
        pd.testing.assert_frame_equal(lab.issues, expected_issues_df, check_exact=False)

        expected_issue_summary_df = pd.DataFrame(
            {
                "issue_type": ["foo", "label"],
                "score": [0.72, 0.6],
                "num_issues": [1, 0],
            }
        )
        pd.testing.assert_frame_equal(
            lab.issue_summary, expected_issue_summary_df, check_exact=False
        )

    def test_save(self, lab, tmp_path, monkeypatch):
        """Test that the save and load methods work."""
        lab.save(tmp_path)
        assert tmp_path.exists(), "Save directory was not created"
        assert (tmp_path / "data").is_dir(), "Data directory was not saved"
        assert (tmp_path / "issues.csv").exists(), "Issues file was not saved"
        assert (tmp_path / "summary.csv").exists(), "Issue summary file was not saved"
        assert (tmp_path / "datalab.pkl").exists(), "Datalab file was not saved"

        # Mock the issues dataframe
        mock_issues = pd.DataFrame(
            {
                "is_foo_issue": [False, True, False, False, False],
                "foo_score": [0.6, 0.8, 0.7, 0.7, 0.8],
            }
        )
        monkeypatch.setattr(lab, "issues", mock_issues)

        # Mock the issue summary dataframe
        mock_issue_summary = pd.DataFrame(
            {
                "issue_type": ["foo"],
                "score": [0.72],
            }
        )
        monkeypatch.setattr(lab, "issue_summary", mock_issue_summary)
        lab.save(tmp_path)
        assert (tmp_path / "issues.csv").exists(), "Issues file was not saved"
        assert (tmp_path / "summary.csv").exists(), "Issue summary file was not saved"

        # Save works in an arbitrary directory, that should be created if it doesn't exist
        new_dir = tmp_path / "subdir"
        assert not new_dir.exists(), "Directory should not exist"
        lab.save(new_dir)
        assert new_dir.exists(), "Directory was not created"

    def test_pickle(self, lab, tmp_path):
        """Test that the class can be pickled."""
        pickle_file = os.path.join(tmp_path, "lab.pkl")
        with open(pickle_file, "wb") as f:
            pickle.dump(lab, f)
        with open(pickle_file, "rb") as f:
            lab2 = pickle.load(f)

        assert lab2.label_name == "star"

    def test_load(self, lab, tmp_path, dataset, monkeypatch):
        """Test that the save and load methods work."""

        # Mock the issues dataframe
        mock_issues = pd.DataFrame(
            {
                "is_foo_issue": [False, True, False, False, False],
                "foo_score": [0.6, 0.8, 0.7, 0.7, 0.8],
            }
        )
        monkeypatch.setattr(lab, "issues", mock_issues)

        # Mock the issue summary dataframe
        mock_issue_summary = pd.DataFrame(
            {
                "issue_type": ["foo"],
                "score": [0.72],
            }
        )
        monkeypatch.setattr(lab, "issue_summary", mock_issue_summary)

        lab.save(tmp_path)

        loaded_lab = Datalab.load(tmp_path)
        data = lab._data
        loaded_data = loaded_lab._data
        assert loaded_data == data
        assert loaded_lab.info == lab.info
        pd.testing.assert_frame_equal(loaded_lab.issues, mock_issues)
        pd.testing.assert_frame_equal(loaded_lab.issue_summary, mock_issue_summary)

        # Load accepts a `Dataset`.
        loaded_lab = Datalab.load(tmp_path, data=dataset)
        assert loaded_lab.data._data == dataset.data

        # Misaligned dataset raises a ValueError
        with pytest.raises(ValueError) as excinfo:
            Datalab.load(tmp_path, data=dataset.shard(2, 0))
            expected_error_msg = "Length of data (2) does not match length of labels (5)"
            assert expected_error_msg == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            Datalab.load(tmp_path, data=dataset.shuffle())
            expected_error_msg = (
                "Data has been modified since Lab was saved. Cannot load Lab with modified data."
            )
            assert expected_error_msg == str(excinfo.value)

    def test_failed_issue_managers(self, lab, monkeypatch):
        """Test that a failed issue manager will not be added to the Datalab instance after
        the call to `find_issues`."""
        mock_issue_types = {"erroneous_issue_type": {}}
        monkeypatch.setattr(lab, "_set_issue_types", lambda *args, **kwargs: mock_issue_types)

        mock_issue_manager = Mock()
        mock_issue_manager.name = "erronous"
        mock_issue_manager.find_issues.side_effect = ValueError("Some error")

        class MockIssueManagerFactory:
            @staticmethod
            def from_list(*args, **kwargs):
                return [mock_issue_manager]

        monkeypatch.setattr(
            "cleanlab.experimental.datalab.datalab._IssueManagerFactory", MockIssueManagerFactory
        )

        assert lab.issues.empty
        lab.find_issues()
        assert lab.issues.empty

    @pytest.mark.parametrize("include_description", [True, False])
    def test_get_report(self, lab, include_description, monkeypatch):
        """Test that the report method works. Assuming we have two issue managers, each should add
        their section to the report."""

        mock_issue_manager = Mock()
        mock_issue_manager.issue_name = "foo"
        mock_issue_manager.report.return_value = "foo report"

        class MockIssueManagerFactory:
            @staticmethod
            def from_str(*args, **kwargs):
                return mock_issue_manager

        monkeypatch.setattr(
            "cleanlab.experimental.datalab.datalab._IssueManagerFactory", MockIssueManagerFactory
        )
        mock_issues = pd.DataFrame(
            {
                "is_foo_issue": [False, True, False, False, False],
                "foo_score": [0.6, 0.8, 0.7, 0.7, 0.8],
            }
        )
        monkeypatch.setattr(lab, "issues", mock_issues)

        mock_issue_summary = pd.DataFrame(
            {
                "issue_type": ["foo"],
                "score": [0.72],
            }
        )

        mock_info = {"foo": {"bar": "baz"}}

        monkeypatch.setattr(lab, "issue_summary", mock_issue_summary)

        monkeypatch.setattr(
            lab, "_add_issue_summary_to_report", lambda *args, **kwargs: "Here is a lab summary\n\n"
        )
        monkeypatch.setattr(lab, "issues", mock_issues, raising=False)
        monkeypatch.setattr(lab, "info", mock_info, raising=False)

        report = lab._get_report(
            num_examples=3, verbosity=0, include_description=include_description
        )
        expected_report = "\n\n".join(["Here is a lab summary", "foo report"])
        assert report == expected_report

    def test_report(self, lab, monkeypatch):
        # lab.report simply wraps _get_report in a print statement
        mock_get_report = Mock()
        # Define a mock function that takes a verbosity argument and a k argument
        # and returns a string
        mock_get_report.side_effect = (
            lambda num_examples, verbosity, **_: f"Report with verbosity={verbosity} and k={num_examples}"
        )
        monkeypatch.setattr(lab, "_get_report", mock_get_report)

        # Call report with no arguments, test that it prints the report
        with patch("builtins.print") as mock_print:
            lab.report(verbosity=0)
            mock_print.assert_called_once_with("Report with verbosity=0 and k=5")
            mock_print.reset_mock()
            lab.report(num_examples=10, verbosity=3)
            mock_print.assert_called_once_with("Report with verbosity=3 and k=10")
            mock_print.reset_mock()
            lab.report()
            mock_print.assert_called_once_with("Report with verbosity=1 and k=5")


class TestDatalabIssueManagerInteraction:
    """The Datalab class should integrate with the IssueManager class correctly.

    Tests include:
    - Make sure a custom manager needs to be registered to work with Datalab
    - Make sure that `find_issues()` with different affects the outcome (e.g. `Datalab.issues`)
        differently depending on the issue manager.
    """

    @pytest.fixture
    def custom_issue_manager(self):
        from cleanlab.experimental.datalab.issue_manager import IssueManager

        class CustomIssueManager(IssueManager):
            issue_name = "custom_issue"

            def find_issues(self, custom_argument: int = 1, **_) -> None:
                # Flag example as an issue if the custom argument equals its index
                scores = [
                    abs(i - custom_argument) / (i + custom_argument)
                    for i in range(len(self.datalab.data))
                ]
                self.issues = pd.DataFrame(
                    {
                        f"is_{self.issue_name}_issue": [
                            i == custom_argument for i in range(len(self.datalab.data))
                        ],
                        self.issue_score_key: scores,
                    },
                )
                summary_score = np.mean(scores)
                self.summary = self.make_summary(score=summary_score)

        return CustomIssueManager

    def test_custom_issue_manager_not_registered(self, lab):
        """Test that a custom issue manager that is not registered will not be used."""
        # Mock registry dictionary
        mock_registry = MagicMock()
        mock_registry.__getitem__.side_effect = KeyError("issue type not registered")

        with patch("cleanlab.experimental.datalab.factory.REGISTRY", mock_registry):
            with pytest.raises(ValueError) as excinfo:
                lab.find_issues(issue_types={"custom_issue": {}})

                assert "issue type not registered" in str(excinfo.value)

            assert mock_registry.__getitem__.called_once_with("custom_issue")

            assert lab.issues.empty
            assert lab.issue_summary.empty

    def test_custom_issue_manager_registered(self, lab, custom_issue_manager):
        """Test that a custom issue manager that is registered will be used."""
        from cleanlab.experimental.datalab.factory import register

        register(custom_issue_manager)

        assert lab.issues.empty
        assert lab.issue_summary.empty

        lab.find_issues(issue_types={"custom_issue": {}})

        expected_is_custom_issue_issue = [False, True] + [False] * 3
        expected_custom_issue_score = [1 / 1, 0 / 2, 1 / 3, 2 / 4, 3 / 5]
        expected_issues = pd.DataFrame(
            {
                "is_custom_issue_issue": expected_is_custom_issue_issue,
                "custom_issue_score": expected_custom_issue_score,
            }
        )
        assert pd.testing.assert_frame_equal(lab.issues, expected_issues) is None

    def test_find_issues_for_custom_issue_manager_with_custom_kwarg(
        self, lab, custom_issue_manager
    ):
        """Test that a custom issue manager that is registered will be used."""
        from cleanlab.experimental.datalab.factory import register

        register(custom_issue_manager)

        assert lab.issues.empty
        assert lab.issue_summary.empty

        lab.find_issues(issue_types={"custom_issue": {"custom_argument": 3}})

        expected_is_custom_issue_issue = [False, False, False, True, False]
        expected_custom_issue_score = [3 / 3, 2 / 4, 1 / 5, 0 / 6, 1 / 7]
        expected_issues = pd.DataFrame(
            {
                "is_custom_issue_issue": expected_is_custom_issue_issue,
                "custom_issue_score": expected_custom_issue_score,
            }
        )
        assert pd.testing.assert_frame_equal(lab.issues, expected_issues) is None

        # Clean up registry
        from cleanlab.experimental.datalab.factory import REGISTRY

        REGISTRY.pop(custom_issue_manager.issue_name)


@pytest.mark.parametrize(
    "find_issues_kwargs",
    [
        ({"pred_probs": np.random.rand(3, 2)}),
        ({"features": np.random.rand(3, 2)}),
        ({"pred_probs": np.random.rand(3, 2), "features": np.random.rand(6, 2)}),
    ],
    ids=["pred_probs", "features", "pred_probs and features"],
)
def test_report_for_outlier_issues_via_pred_probs(find_issues_kwargs):
    data = {"labels": [0, 1, 0]}
    lab = Datalab(data=data, label_name="labels")
    find_issues_kwargs["issue_types"] = {"outlier": {"k": 1}}
    lab.find_issues(**find_issues_kwargs)

    report = lab._get_report(num_examples=3, verbosity=0, include_description=False)
    assert report, "Report should not be empty"


def test_near_duplicates_reuses_knn_graph():
    """'outlier' and 'near_duplicate' issues both require a KNN graph.
    This test ensures that the KNN graph is only computed once.
    E.g. if outlier is called first, and then near_duplicate can reuse the
    resulting graph.
    """
    N = 3000
    num_features = 1000
    k = 20
    find_issues_kwargs = {"issue_types": {"near_duplicate": {"k": k}}}
    data = {"labels": np.random.randint(0, 2, size=N)}
    features = np.random.rand(N, num_features)
    lab = Datalab(data=data, label_name="labels")
    time_only_near_duplicates = timeit.timeit(
        lambda: lab.find_issues(features=features, **find_issues_kwargs),
        number=1,
    )

    lab = Datalab(data=data, label_name="labels")
    # Outliers need more neighbors, so this should be slower, so the graph will be computed twice
    find_issues_kwargs = {
        "issue_types": {"near_duplicate": {"k": k}, "outlier": {"k": k}},
    }
    time_near_duplicates_and_outlier = timeit.timeit(
        lambda: lab.find_issues(features=features, **find_issues_kwargs),
        number=1,
    )
    assert time_only_near_duplicates < time_near_duplicates_and_outlier

    find_issues_kwargs = {
        "issue_types": {"outlier": {"k": k}, "near_duplicate": {"k": k}},
    }
    time_outliers_before_near_duplicates = timeit.timeit(
        lambda: lab.find_issues(features=features, **find_issues_kwargs),
        number=1,
    )
    assert (
        time_outliers_before_near_duplicates < time_near_duplicates_and_outlier
    ), "KNN graph reuse should make this run of find_issues faster."
