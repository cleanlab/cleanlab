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


import contextlib
import io
import os
import pickle
import timeit
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from datasets.dataset_dict import DatasetDict
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import cleanlab
from cleanlab.datalab.datalab import Datalab
from cleanlab.datalab.internal.issue_finder import IssueFinder
from cleanlab.datalab.internal.report import Reporter

SEED = 42


def test_datalab_invalid_datasetdict(dataset, label_name):
    with pytest.raises(ValueError) as e:
        datadict = DatasetDict({"train": dataset, "test": dataset})
        Datalab(datadict, label_name)  # type: ignore
        assert "Please pass a single dataset, not a DatasetDict." in str(e)


@pytest.fixture(scope="function")
def list_possible_issue_types(monkeypatch, request):
    monkeypatch.setattr(IssueFinder, "list_possible_issue_types", lambda *_: request.param)


class TestDatalab:
    """Tests for the Datalab class."""

    @pytest.fixture
    def lab(self, dataset, label_name):
        return Datalab(data=dataset, label_name=label_name)

    def test_print(self, lab, capsys):
        # Can print the object
        print(lab)
        captured = capsys.readouterr()
        expected_output = (
            "Datalab:\n"
            "Checks run: No\n"
            "Number of examples: 5\n"
            "Number of classes: 3\n"
            "Issues identified: Not checked\n"
        )
        assert expected_output == captured.out

    def test_class_names(self):
        y = ["a", "3", "2", "3"]
        lab = Datalab({"y": y}, label_name="y")
        assert lab.class_names == ["2", "3", "a"]

        y = [-1, 4, 0.5, 0, 4, -1]
        lab = Datalab({"y": y}, label_name="y")
        assert lab.class_names == [-1, 0, 0.5, 4]

    def test_list_default_issue_types(self):
        assert Datalab.list_default_issue_types() == [
            "label",
            "outlier",
            "near_duplicate",
            "non_iid",
        ]

    def tmp_path(self):
        # A path for temporarily saving the instance during tests.
        # This is a workaround for the fact that the Datalab class
        # does not have a save method.
        return Path(__file__).parent / "tmp.pkl"

    def test_attributes(self, lab):
        # Has the right attributes
        for attr in ["data", "label_name", "_labels", "info", "issues"]:
            assert hasattr(lab, attr), f"Missing attribute {attr}"

        assert all(lab.labels == np.array([1, 1, 2, 0, 2]))
        assert isinstance(lab.issues, pd.DataFrame), "Issues should by in a dataframe"
        assert isinstance(lab.issue_summary, pd.DataFrame), "Issue summary should be a dataframe"

    def test_get_info(self, lab):
        mock_info: dict = {
            "label": {
                "given_label": [1, 0, 1, 0, 2],
                "predicted_label": [1, 1, 2, 0, 2],
                # get_info("label") adds `class_names` from statistics
            },
            "near_duplicate": {
                "nearest_neighbor": [1, 0, 0, 4, 3],
            },
        }
        mock_info = {**lab.info, **mock_info}
        lab.info = mock_info

        label_info = lab.get_info("label")
        assert label_info["given_label"].tolist() == [4, 3, 4, 3, 5]
        assert label_info["predicted_label"].tolist() == [4, 4, 5, 3, 5]
        assert label_info["class_names"] == [3, 4, 5]

        near_duplicate_info = lab.get_info("near_duplicate")
        assert near_duplicate_info["nearest_neighbor"] == [1, 0, 0, 4, 3]

        assert lab.get_info() == lab.info == mock_info

    def test_get_issue_summary(self, lab, monkeypatch):
        mock_summary: pd.DataFrame = pd.DataFrame(
            {
                "issue_type": ["label", "outlier"],
                "score": [0.5, 0.3],
                "num_issues": [1, 2],
            }
        )
        monkeypatch.setattr(lab, "issue_summary", mock_summary)

        label_summary = lab.get_issue_summary(issue_name="label")
        pd.testing.assert_frame_equal(label_summary, mock_summary.iloc[[0]])

        outlier_summary = lab.get_issue_summary(issue_name="outlier")
        pd.testing.assert_frame_equal(
            outlier_summary, mock_summary.iloc[[1]].reset_index(drop=True)
        )

        summary = lab.get_issue_summary()
        pd.testing.assert_frame_equal(summary, mock_summary)

    def test_get_issues(self, lab, monkeypatch):
        mock_issues: pd.DataFrame = pd.DataFrame(
            {
                "is_label_issue": [True, False, False, True, False],
                "label_score": [0.2, 0.4, 0.6, 0.1, 0.8],
                "is_near_duplicate_issue": [False, True, True, False, True],
                "near_duplicate_score": [0.5, 0.3, 0.1, 0.7, 0.2],
            },
        )
        monkeypatch.setattr(lab, "issues", mock_issues)

        mock_predicted_labels = np.array([0, 1, 2, 1, 2])

        mock_distance_to_nearest_neighbor = [0.1, 0.2, 0.3, 0.4, 0.5]

        lab.info.update(
            {
                "label": {
                    "given_label": lab.labels,
                    "predicted_label": mock_predicted_labels,
                },
                "near_duplicate": {
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

        pd.testing.assert_frame_equal(label_issues, expected_label_issues, check_dtype=False)

        near_duplicate_issues = lab.get_issues(issue_name="near_duplicate")

        expected_near_duplicate_issues = pd.DataFrame(
            {
                **{
                    key: mock_issues[key]
                    for key in ["is_near_duplicate_issue", "near_duplicate_score"]
                },
                "distance_to_nearest_neighbor": mock_distance_to_nearest_neighbor,
            },
        )
        pd.testing.assert_frame_equal(
            near_duplicate_issues, expected_near_duplicate_issues, check_dtype=False
        )

        issues = lab.get_issues()
        pd.testing.assert_frame_equal(issues, mock_issues, check_dtype=False)

    @pytest.mark.parametrize(
        "issue_types",
        [None, {"label": {}}],
        ids=["Default issues", "Only label issues"],
    )
    def test_find_issues_with_pred_probs(self, lab, pred_probs, issue_types):
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

    def test_find_issues_without_values_in_issue_types_raises_warning(self, lab, pred_probs):
        issue_types = {}
        with pytest.warns(UserWarning) as record:
            lab.find_issues(pred_probs=pred_probs, issue_types=issue_types)
        warning_message = record[0].message.args[0]
        assert (
            "No issue types were specified so no issues will be found in the dataset. Set `issue_types` as None to consider a default set of issues."
            in warning_message
        )

    @pytest.mark.parametrize(
        "issue_types",
        [
            None,
            {"label": {}},
            {"outlier": {}},
            {"near_duplicate": {}},
            {"non_iid": {}},
            {"outlier": {}, "near_duplicate": {}},
        ],
        ids=[
            "Defaults",
            "Only label issues",
            "Only outlier issues",
            "Only near_duplicate issues",
            "Only non_iid issues",
            "Both outlier and near_duplicate issues",
        ],
    )
    @pytest.mark.parametrize(
        "use_features",
        [True, False],
        ids=["Use features", "Don't use features"],
    )
    @pytest.mark.parametrize(
        "use_pred_probs",
        [True, False],
        ids=["Use pred_probs", "Don't use pred_probs"],
    )
    @pytest.mark.parametrize(
        "use_knn_graph",
        [True, False],
        ids=["Use knn_graph", "Don't use knn_graph"],
    )
    def test_repeat_find_issues_then_report_with_defaults(
        self,
        large_lab,
        issue_types,
        use_features,
        use_pred_probs,
        use_knn_graph,
    ):
        """Test "all" combinations of inputs to find_issues() and make sure repeated calls to it won't change any results. Same applies to report().

        This test does NOT test the correctness of the inputs, so some test cases may lead to missing arguments errors that are silently ignored.
        """

        # Extract features and pred_probs from Datalab object
        features, pred_probs = (
            np.array(large_lab.data[k]) if v else None
            for k, v in zip(["features", "pred_probs"], [use_features, use_pred_probs])
        )

        # Extract sparse knn_graph from Datalab object's info dictionary
        knn_graph = None
        if use_knn_graph:
            knn_graph = large_lab.info["statistics"]["unit_test_knn_graph"]

        # Run find_issues and report() once
        large_lab.find_issues(
            features=features, pred_probs=pred_probs, knn_graph=knn_graph, issue_types=issue_types
        )
        with contextlib.redirect_stdout(io.StringIO()) as f:
            large_lab.report()
        first_report = f.getvalue()
        issues = large_lab.issues.copy()
        issue_summary = large_lab.issue_summary.copy()

        # Rerunning find_issues() and report() with the same default parameters should not change the number of issues
        large_lab.find_issues(
            features=features, pred_probs=pred_probs, knn_graph=knn_graph, issue_types=issue_types
        )
        with contextlib.redirect_stdout(io.StringIO()) as f:
            large_lab.report()
        second_report = f.getvalue()
        pd.testing.assert_frame_equal(large_lab.issues, issues)
        pd.testing.assert_frame_equal(large_lab.issue_summary, issue_summary)
        assert first_report == second_report

    @pytest.mark.parametrize("k", [2, 3])
    @pytest.mark.parametrize("metric", ["euclidean", "cosine"])
    def test_find_issues_with_custom_hyperparams(self, lab, pred_probs, k, metric):
        dataset_size = lab.get_info("statistics")["num_examples"]
        embedding_size = 2
        mock_embeddings = np.random.rand(dataset_size, embedding_size)

        knn = NearestNeighbors(n_neighbors=k, metric=metric)
        issue_types = {"outlier": {"knn": knn}}
        assert lab.get_info("statistics").get("weighted_knn_graph") is None
        lab.find_issues(
            pred_probs=pred_probs,
            features=mock_embeddings,
            issue_types=issue_types,
        )
        assert lab.info["outlier"]["k"] == k
        statistics = lab.get_info("statistics")
        assert statistics["knn_metric"] == metric
        knn_graph = statistics["weighted_knn_graph"]
        assert isinstance(knn_graph, csr_matrix)
        assert knn_graph.shape == (dataset_size, dataset_size)
        assert knn_graph.nnz == dataset_size * k

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
                "score": [0.72, 0.4],
                "num_issues": [1, 0],
            }
        )
        pd.testing.assert_frame_equal(
            lab.issue_summary, expected_issue_summary_df, check_exact=False
        )

    def test_save(self, lab, tmp_path, monkeypatch):
        """Test that the save and load methods work."""
        lab.save(tmp_path, force=True)
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
        lab.save(tmp_path, force=True)
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

        lab.save(tmp_path, force=True)

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

    @pytest.mark.parametrize("list_possible_issue_types", [["erroneous_issue_type"]], indirect=True)
    def test_failed_issue_managers(self, lab, monkeypatch, list_possible_issue_types):
        """Test that a failed issue manager will not be added to the Datalab instance after
        the call to `find_issues`."""
        mock_issue_types = {"erroneous_issue_type": {}}

        mock_issue_manager = Mock()
        mock_issue_manager.issue_name = "erroneous_issue_type"
        mock_issue_manager.find_issues.side_effect = ValueError("Some error")

        class MockIssueManagerFactory:
            @staticmethod
            def from_list(*args, **kwargs):
                return [mock_issue_manager]

        monkeypatch.setattr(
            "cleanlab.datalab.internal.issue_finder._IssueManagerFactory", MockIssueManagerFactory
        )

        assert lab.issues.empty
        with patch("builtins.print") as mock_print:
            lab.find_issues(issue_types=mock_issue_types)
            for expected_msg_substr in [
                "Error in",
                "Audit complete",
                "Failed to check for these issue types: ",
            ]:
                assert any(expected_msg_substr in call[0][0] for call in mock_print.call_args_list)

        assert lab.issues.empty

    def test_report(self, lab, monkeypatch, capsys):
        class MockReporter:
            def __init__(self, *args, **kwargs):
                self.verbosity = kwargs.get("verbosity", None)
                assert self.verbosity is not None, "Reporter should be initialized with verbosity"

            def report(self, *args, **kwargs) -> None:
                print(
                    f"Report with verbosity={self.verbosity} and k={kwargs.get('num_examples', 5)}"
                )

        monkeypatch.setattr(cleanlab.datalab.internal.helper_factory, "Reporter", MockReporter)
        monkeypatch.setattr(
            lab.data_issues,
            "issue_summary",
            pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD")),
        )
        lab.report(verbosity=0)
        captured = capsys.readouterr()
        assert "Report with verbosity=0 and k=5" in captured.out

        lab.report(num_examples=10, verbosity=3)
        captured = capsys.readouterr()
        assert "Report with verbosity=3 and k=10" in captured.out

        lab.report()
        captured = capsys.readouterr()
        assert "Report with verbosity=1 and k=5" in captured.out


class TestDatalabUsingKNNGraph:
    """The Datalab class can accept a `knn_graph` argument to `find_issues` that should
    be used instead of computing a new one from the `features` argument."""

    @pytest.fixture
    def data_tuple(self):
        # from cleanlab.datalab.datalab import Datalab
        np.random.seed(SEED)
        N = 10
        data = {"label": np.random.randint(0, 2, size=N)}
        features = np.random.rand(N, 5)
        knn_graph = (
            NearestNeighbors(n_neighbors=3, metric="cosine")
            .fit(features)
            .kneighbors_graph(mode="distance")
        )
        return Datalab(data=data, label_name="label"), knn_graph, features

    def test_knn_graph(self, data_tuple):
        """Test that the `knn_graph` argument to `find_issues` is used instead of computing a new
        one from the `features` argument."""
        lab, knn_graph, _ = data_tuple
        assert lab.get_info("statistics").get("weighted_knn_graph") is None
        lab.find_issues(knn_graph=knn_graph)
        knn_graph_stats = lab.get_info("statistics").get("weighted_knn_graph")
        np.testing.assert_array_equal(knn_graph_stats.toarray(), knn_graph.toarray())

        assert lab.get_info("statistics").get("knn_metric") is None

    def test_features_and_knn_graph(self, data_tuple):
        """Test that the `knn_graph` argument to `find_issues` is used instead of computing a new
        one from the `features` argument."""
        lab, knn_graph, features = data_tuple
        k = 4
        lab.find_issues(knn_graph=knn_graph, features=features, issue_types={"outlier": {"k": k}})
        knn_graph_stats = lab.get_info("statistics").get("weighted_knn_graph")
        assert knn_graph_stats.nnz == k * len(
            lab.data
        ), f"Expected {k * len(lab.data)} nnz, got {knn_graph_stats.nnz}"
        three_nn_dists = knn_graph_stats.data.reshape(len(lab.data), k)[:, :3]
        knn_graph_three_nn_dists = knn_graph.data.reshape(len(lab.data), k - 1)
        np.testing.assert_array_equal(three_nn_dists, knn_graph_three_nn_dists)
        assert lab.get_info("statistics").get("knn_metric") == "cosine"

    def test_without_features_or_knn_graph(self, data_tuple):
        """Test that the `knn_graph` argument to `find_issues` is used instead of computing a new
        one from the `features` argument."""
        lab, _, _ = data_tuple

        # Test that a warning is raised
        lab.find_issues()
        assert lab.issues.empty  # No columns should be added to the issues dataframe

    def test_data_valuation_issue_with_knn_graph(self, data_tuple):
        lab, knn_graph, features = data_tuple
        assert lab.get_info("statistics").get("weighted_knn_graph") is None
        lab.find_issues(knn_graph=knn_graph, issue_types={"data_valuation": {}})
        score = lab.get_issues().get(["data_valuation_score"])
        assert isinstance(score, pd.DataFrame)
        assert len(score) == len(lab.data)

    def test_data_valuation_issue_with_existing_knn_graph(self, data_tuple):
        lab, knn_graph, features = data_tuple
        lab.find_issues(features=features, issue_types={"outlier": {"k": 3}})
        lab.find_issues(issue_types={"data_valuation": {}})
        score = lab.get_issues().get(["data_valuation_score"])
        assert isinstance(score, pd.DataFrame)
        assert len(score) == len(lab.data)

        # Compare this with directly passing in a knn_graph
        lab_2 = Datalab(data=lab.data, label_name=lab.label_name)
        lab_2.find_issues(knn_graph=knn_graph, issue_types={"data_valuation": {}})
        score_2 = lab_2.get_issues().get(["data_valuation_score"])
        pd.testing.assert_frame_equal(score, score_2)

    def test_data_valuation_issue_without_knn_graph(self, data_tuple):
        lab, _, features = data_tuple
        lab.find_issues(features=features, issue_types={"data_valuation": {}})
        assert (
            lab.issues.empty
        ), "The issues dataframe should be empty as the issue manager expects an existing knn_graph"


class TestDatalabIssueManagerInteraction:
    """The Datalab class should integrate with the IssueManager class correctly.

    Tests include:
    - Make sure a custom manager needs to be registered to work with Datalab
    - Make sure that `find_issues()` with different affects the outcome (e.g. `Datalab.issues`)
        differently depending on the issue manager.
    """

    def test_custom_issue_manager_not_registered(self, lab):
        """Test that a custom issue manager that is not registered will not be used."""
        # Mock registry dictionary
        mock_registry = MagicMock()
        mock_registry.__getitem__.side_effect = KeyError("issue type not registered")

        with patch("cleanlab.datalab.internal.issue_manager_factory.REGISTRY", mock_registry):
            with pytest.raises(ValueError) as excinfo:
                lab.find_issues(issue_types={"custom_issue": {}})

                assert "issue type not registered" in str(excinfo.value)

            assert mock_registry.__getitem__.called_once_with("custom_issue")

            assert lab.issues.empty
            assert lab.issue_summary.empty

    def test_custom_issue_manager_registered(self, lab, custom_issue_manager):
        """Test that a custom issue manager that is registered will be used."""
        from cleanlab.datalab.internal.issue_manager_factory import register

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

    @pytest.mark.parametrize("list_possible_issue_types", [["custom_issue"]], indirect=True)
    def test_find_issues_for_custom_issue_manager_with_custom_kwarg(
        self, lab, custom_issue_manager, list_possible_issue_types
    ):
        """Test that a custom issue manager that is registered will be used."""
        from cleanlab.datalab.internal.issue_manager_factory import register

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
        from cleanlab.datalab.internal.issue_manager_factory import REGISTRY

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

    reporter = Reporter(lab.data_issues, verbosity=0, include_description=False)
    report = reporter.get_report(num_examples=3)
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
    data = {"labels": np.random.randint(0, 2, size=N)}

    np.random.seed(SEED)
    features = np.random.rand(N, num_features)

    # Run 1: only near_duplicate
    lab = Datalab(data=data, label_name="labels")
    find_issues_kwargs = {"issue_types": {"near_duplicate": {"k": k}}}
    time_only_near_duplicates = timeit.timeit(
        lambda: lab.find_issues(features=features, **find_issues_kwargs),
        number=1,
    )

    # Run 2: near_duplicate and outlier with same k
    lab = Datalab(data=data, label_name="labels")
    # Outliers need more neighbors, so this should be slower, so the graph will be computed twice
    find_issues_kwargs = {
        "issue_types": {"near_duplicate": {"k": k}, "outlier": {"k": 2 * k}},
    }
    time_near_duplicates_and_outlier = timeit.timeit(
        lambda: lab.find_issues(features=features, **find_issues_kwargs),
        number=1,
    )

    # Run 3: Same Datalab instance with same issues, but in different order
    find_issues_kwargs = {
        "issue_types": {"outlier": {"k": 2 * k}, "near_duplicate": {"k": k}},
    }
    time_outliers_before_near_duplicates = timeit.timeit(
        lambda: lab.find_issues(features=features, **find_issues_kwargs),
        number=1,
    )

    # Run 2 does an extra check, so it should be slower
    assert time_only_near_duplicates < time_near_duplicates_and_outlier, (
        "Run 2 should be slower because it does an extra check "
        "for outliers, which requires a KNN graph."
    )

    # Run 3 should be faster because it reuses the KNN graph from Run 2
    # in both issue checks
    assert (
        time_outliers_before_near_duplicates < time_near_duplicates_and_outlier
    ), "KNN graph reuse should make this run of find_issues faster."


class TestDatalabFindNonIIDIssues:
    """This class focuses on testing the end-to-end functionality of calling Datalab.find_issues()
    only for non-IID issues. The tests in this class are not meant to test the underlying
    functionality of the non-IID issue finders themselves, but rather to test that the
    Datalab.find_issues() method correctly calls the non-IID issue finders and results are consistent.
    """

    @pytest.fixture
    def random_embeddings(self):
        np.random.seed(SEED)
        return np.random.rand(100, 10)

    @pytest.fixture
    def sorted_embeddings(self):
        np.random.seed(SEED)
        n_samples = 1000

        # Stack features to create a 3D dataset
        x = np.linspace(0, 4 * np.pi, n_samples)
        y = np.sin(x) + np.random.normal(0, 0.1, n_samples)
        z = np.cos(x) + np.random.normal(0, 0.1, n_samples)
        return np.column_stack((x, y, z))

    def test_find_non_iid_issues(self, random_embeddings):
        data = {"labels": [0, 1, 0]}
        lab = Datalab(data=data, label_name="labels")
        lab.find_issues(features=random_embeddings, issue_types={"non_iid": {}})
        summary = lab.get_issue_summary()
        assert ["non_iid"] == summary["issue_type"].values
        assert summary["score"].values[0] > 0.05
        assert lab.get_issues()["is_non_iid_issue"].sum() == 0

    def test_find_non_iid_issues_using_pred_probs(self, random_embeddings):
        data = {"labels": [0, 1, 0]}
        lab = Datalab(data=data, label_name="labels")
        pred_probs = random_embeddings / random_embeddings.sum(axis=1, keepdims=True)
        lab.find_issues(pred_probs=pred_probs, issue_types={"non_iid": {}})
        summary = lab.get_issue_summary()
        assert ["non_iid"] == summary["issue_type"].values
        assert summary["score"].values[0] > 0.05
        assert lab.get_issues()["is_non_iid_issue"].sum() == 0

    def test_find_non_iid_issues_sorted(self, sorted_embeddings):
        data = {"labels": [0, 1, 0]}
        lab = Datalab(data=data, label_name="labels")
        lab.find_issues(features=sorted_embeddings, issue_types={"non_iid": {}})
        summary = lab.get_issue_summary()
        assert ["non_iid"] == summary["issue_type"].values
        assert summary["score"].values[0] == 0
        assert lab.get_issues()["is_non_iid_issue"].sum() == 1

    def test_find_non_iid_issues_sorted_using_pred_probs(self, sorted_embeddings):
        data = {"labels": [0, 1, 0]}
        lab = Datalab(data=data, label_name="labels")
        pred_probs = sorted_embeddings / sorted_embeddings.sum(axis=1, keepdims=True)
        lab.find_issues(pred_probs=pred_probs, issue_types={"non_iid": {}})
        summary = lab.get_issue_summary()
        assert ["non_iid"] == summary["issue_type"].values
        assert summary["score"].values[0] == 0
        assert lab.get_issues()["is_non_iid_issue"].sum() == 1

    def test_incremental_search(self, sorted_embeddings):
        data = {"labels": [0, 1, 0]}
        lab = Datalab(data=data, label_name="labels")
        lab.find_issues(features=sorted_embeddings)
        summary = lab.get_issue_summary()
        assert len(summary) == 3
        lab.find_issues(features=sorted_embeddings, issue_types={"non_iid": {}})
        summary = lab.get_issue_summary()
        assert len(summary) == 3
        assert "non_iid" in summary["issue_type"].values
        non_iid_summary = lab.get_issue_summary("non_iid")
        assert non_iid_summary["score"].values[0] == 0
        assert non_iid_summary["num_issues"].values[0] == 1

    def test_incremental_search_using_pred_probs(self, sorted_embeddings):
        data = {"labels": [0, 1, 0]}
        lab = Datalab(data=data, label_name="labels")
        pred_probs = sorted_embeddings / sorted_embeddings.sum(axis=1, keepdims=True)
        lab.find_issues(pred_probs=pred_probs, issue_types={"non_iid": {}})
        summary = lab.get_issue_summary()
        assert len(summary) == 1
        lab.find_issues(pred_probs=pred_probs, issue_types={"non_iid": {}})
        summary = lab.get_issue_summary()
        assert len(summary) == 1
        assert "non_iid" in summary["issue_type"].values
        non_iid_summary = lab.get_issue_summary("non_iid")
        assert non_iid_summary["score"].values[0] == 0
        assert non_iid_summary["num_issues"].values[0] == 1


class TestDatalabFindLabelIssues:
    @pytest.fixture
    def random_embeddings(self):
        np.random.seed(SEED)
        return np.random.rand(100, 10)

    @pytest.fixture
    def pred_probs(self):
        np.random.seed(SEED)
        pred_probs_array = np.random.rand(100, 2)
        return pred_probs_array / pred_probs_array.sum(axis=1, keepdims=True)

    def test_incremental_search(self, pred_probs, random_embeddings):
        data = {"labels": np.random.randint(0, 2, 100)}
        lab = Datalab(data=data, label_name="labels")
        lab.find_issues(features=random_embeddings)
        summary = lab.get_issue_summary()
        assert len(summary) == 4
        assert "label" in summary["issue_type"].values
        lab.find_issues(pred_probs=pred_probs, issue_types={"label": {}})
        summary = lab.get_issue_summary()
        assert len(summary) == 4
        assert "label" in summary["issue_type"].values
        label_summary = lab.get_issue_summary("label")
        assert label_summary["num_issues"].values[0] > 0
        # Compare results with low_memory=True
        issues_df = lab.get_issues("label")
        issue_types = {"label": {"clean_learning_kwargs": {"low_memory": True}}}
        lab_lm = Datalab(data=data, label_name="labels")
        lab_lm.find_issues(pred_probs=pred_probs, issue_types=issue_types)
        issues_df_lm = lab_lm.get_issues("label")
        # jaccard similarity
        intersection = len(list(set(issues_df).intersection(set(issues_df_lm))))
        union = len(set(issues_df)) + len(set(issues_df_lm)) - intersection
        assert float(intersection) / union > 0.95

    def test_build_pred_probs_from_features(self, random_embeddings):
        data = {"labels": np.random.randint(0, 2, 100)}
        lab = Datalab(data=data, label_name="labels")
        lab.find_issues(features=random_embeddings, issue_types={"label": {}})
        summary = lab.get_issue_summary()
        assert len(summary) == 1
        assert "label" in summary["issue_type"].values
        lab.find_issues(features=random_embeddings, issue_types={"label": {"k": 5}})
        summary = lab.get_issue_summary()
        assert len(summary) == 1
        assert "label" in summary["issue_type"].values

    def test_pred_probs_precedence(self, pred_probs, random_embeddings):
        data = {"labels": np.random.randint(0, 2, 100)}
        lab = Datalab(data=data, label_name="labels")
        lab.find_issues(pred_probs=pred_probs, issue_types={"label": {}})
        summary = lab.get_issue_summary()
        assert "label" in summary["issue_type"].values
        label_summary_pred_probs = lab.get_issue_summary("label")
        assert label_summary_pred_probs["num_issues"].values[0] > 0
        lab = Datalab(data=data, label_name="labels")
        lab.find_issues(
            features=random_embeddings, pred_probs=pred_probs, issue_types={"label": {}}
        )
        summary = lab.get_issue_summary()
        assert "label" in summary["issue_type"].values
        label_summary_both = lab.get_issue_summary("label")
        assert (
            label_summary_both["num_issues"].values[0]
            == label_summary_pred_probs["num_issues"].values[0]
        )


class TestDatalabFindOutlierIssues:
    @pytest.fixture
    def random_embeddings(self):
        np.random.seed(SEED)
        X = np.random.rand(100, 10)
        X[-1] += 10 * np.random.rand(10)
        return np.random.rand(100, 10)

    @pytest.fixture
    def pred_probs(self):
        np.random.seed(SEED)
        pred_probs_array = np.random.rand(100, 2)
        return pred_probs_array / pred_probs_array.sum(axis=1, keepdims=True)

    def test_incremental_search(self, pred_probs, random_embeddings):
        data = {"labels": np.random.randint(0, 2, 100)}
        lab = Datalab(data=data, label_name="labels")
        lab.find_issues(pred_probs=pred_probs, issue_types={"label": {}})
        summary = lab.get_issue_summary()
        assert len(summary) == 1
        assert "outlier" not in summary["issue_type"].values
        lab.find_issues(features=random_embeddings, issue_types={"outlier": {}})
        summary = lab.get_issue_summary()
        assert len(summary) == 2
        assert "outlier" in summary["issue_type"].values
        outlier_summary = lab.get_issue_summary("outlier")
        assert outlier_summary["num_issues"].values[0] > 0


class TestDatalabFindNearDuplicateIssues:
    @pytest.fixture
    def random_embeddings(self):
        np.random.seed(SEED)
        X = np.random.rand(100, 10)
        X[-1] = X[-1] * -1
        X[-2] = X[-1] + 0.0001 * np.random.rand(10)
        return X

    @pytest.fixture
    def fixed_embeddings(self):
        near_duplicate_scale = 0.0001
        non_duplicate_scale = 100
        X = np.array(
            [[0, 0]] * 4  # Points with 3 exact duplicates
            + [[1, 1]] * 2  # Points with 1 exact duplicate
            + [[1, 0]] * 3
            + [[1 + near_duplicate_scale, 0]]
            + [
                [1, 0 + near_duplicate_scale]
            ]  # Points with 2 exact duplicates and 2 near duplicates
            + [
                [-1, -1] + np.random.rand(2) * near_duplicate_scale for _ in range(5)
            ]  # Points with 5 near duplicates
            + [
                [-1, 0] + np.random.rand(2) * near_duplicate_scale for _ in range(2)
            ]  # Points with 1 near duplicate
            + (np.random.rand(20, 2) * non_duplicate_scale).tolist()  # Random points
        )
        return X

    @pytest.fixture
    def pred_probs(self):
        np.random.seed(SEED)
        pred_probs_array = np.random.rand(100, 2)
        return pred_probs_array / pred_probs_array.sum(axis=1, keepdims=True)

    def test_incremental_search(self, pred_probs, random_embeddings):
        data = {"labels": np.random.randint(0, 2, 100)}
        lab = Datalab(data=data, label_name="labels")
        lab.find_issues(pred_probs=pred_probs, issue_types={"label": {}})
        summary = lab.get_issue_summary()
        assert len(summary) == 1
        assert "near_duplicate" not in summary["issue_type"].values
        lab.find_issues(features=random_embeddings, issue_types={"near_duplicate": {}})
        summary = lab.get_issue_summary()
        assert len(summary) == 2
        assert "near_duplicate" in summary["issue_type"].values
        near_duplicate_summary = lab.get_issue_summary("near_duplicate")
        assert near_duplicate_summary["num_issues"].values[0] > 1

    def test_fixed_embeddings_outputs(self, fixed_embeddings):
        lab = Datalab(data={"a": ["" for _ in range(len(fixed_embeddings))]})
        lab.find_issues(features=fixed_embeddings, issue_types={"near_duplicate": {}})
        issues = lab.get_issues("near_duplicate")

        assert issues["is_near_duplicate_issue"].sum() == 18
        assert all(
            issues["is_near_duplicate_issue"].values
            == [True] * 18 + [False] * (len(fixed_embeddings) - 18)
        )

        # Test the first set of near duplicates (only 3 exact duplicates)
        near_duplicate_sets = issues["near_duplicate_sets"].values

        expected_near_duplicate_sets = np.array(
            [
                # 3 exact duplicates
                np.array([3, 1, 2]),
                np.array([0, 3, 2]),
                np.array([0, 3, 1]),
                np.array([0, 1, 2]),
                # 1 exact duplicate
                np.array([5]),
                np.array([4]),
                # 2 exact duplicates and 2 near duplicates
                np.array([8, 7, 9, 10]),
                np.array([8, 6, 9, 10]),
                np.array([6, 7, 9, 10]),
                np.array([8, 6, 7, 10]),
                np.array([7, 8, 6, 9]),
                # 4 near duplicates
                np.array([15, 13, 14, 12]),
                np.array([13, 14, 15, 11]),
                np.array([14, 12, 15, 11]),
                np.array([13, 12, 15, 11]),
                np.array([11, 13, 14, 12]),
                # 1 near duplicate
                np.array([17]),
                np.array([16]),
            ]
            +
            # Random points
            [np.array([])] * 20,
            dtype=object,
        )

        # Exact duplicates may have arbitrary order, so sort the sets before comparing
        equal_sets = [
            np.array_equal(sorted(a), sorted(b))
            for a, b in zip(near_duplicate_sets, expected_near_duplicate_sets)
        ]
        assert all(equal_sets)

        # Assert self-idx is not included in near duplicate sets
        assert all([i not in s for i, s in enumerate(near_duplicate_sets)])

        # Assert near duplicate sets are unique, ignoring empty sets
        unique_non_empty_sets = [tuple(s) for s in near_duplicate_sets if len(s) > 0]
        assert len(set(unique_non_empty_sets)) == 18


class TestDatalabWithoutLabels:
    num_examples = 100
    num_features = 10
    K = 2

    @pytest.fixture
    def features(self):
        np.random.seed(SEED)
        return np.random.rand(self.num_examples, self.num_features)

    @pytest.fixture
    def pred_probs(self):
        np.random.seed(SEED)
        pred_probs_array = np.random.rand(self.num_examples, self.K)
        return pred_probs_array / pred_probs_array.sum(axis=1, keepdims=True)

    @pytest.fixture
    def lab(self, features):
        return Datalab(data={"X": features})

    @pytest.fixture
    def labels(self):
        np.random.seed(SEED)
        return np.random.randint(0, self.K, self.num_examples)

    def test_init(self, lab, features):
        assert np.array_equal(lab.data["X"], features)
        assert np.array_equal(lab.labels, [])

    def test_find_issues(self, lab, features, pred_probs):
        lab = Datalab(data={"X": features})
        lab.find_issues(pred_probs=pred_probs)
        assert set(lab.issues.columns) == {"is_non_iid_issue", "non_iid_score"}

        lab = Datalab(data={"X": features})
        lab.find_issues(features=features)
        assert not lab.issues.empty

    def test_find_issues_features_works_with_and_without_labels(self, features, labels):
        lab_without_labels = Datalab(data={"X": features})
        lab_without_labels.find_issues(features=features)

        lab_with_labels = Datalab(data={"X": features, "labels": labels}, label_name="labels")
        lab_with_labels.find_issues(features=features)

        lab_without_label_name = Datalab(data={"X": features, "labels": labels})
        lab_without_label_name.find_issues(features=features)

        issues_without_labels = lab_without_labels.issues
        issues_with_labels = lab_with_labels.issues
        issues_without_label_name = lab_without_label_name.issues

        # issues_with_labels should have two additional columns about label issues
        assert len(issues_without_labels.columns) + 2 == len(issues_with_labels.columns)
        pd.testing.assert_frame_equal(issues_without_labels, issues_without_label_name)


class TestDataLabClassImbalanceIssues:
    K = 3
    N = 100
    num_features = 2

    @pytest.fixture
    def random_embeddings(self):
        np.random.seed(SEED)
        return np.random.rand(self.N, self.num_features)

    @pytest.fixture
    def imbalance_labels(self):
        np.random.seed(SEED)
        labels = np.random.choice(np.arange(self.K - 1), 100, p=[0.5] * (self.K - 1))
        labels[0] = 2
        return labels

    @pytest.fixture
    def pred_probs(self):
        np.random.seed(SEED)
        pred_probs_array = np.random.rand(self.N, self.K)
        return pred_probs_array / pred_probs_array.sum(axis=1, keepdims=True)

    def test_incremental_search(self, pred_probs, random_embeddings, imbalance_labels):
        data = {"labels": imbalance_labels}
        lab = Datalab(data=data, label_name="labels")
        lab.find_issues(pred_probs=pred_probs, issue_types={"label": {}})
        summary = lab.get_issue_summary()
        assert len(summary) == 1
        assert "class_imbalance" not in summary["issue_type"].values
        lab.find_issues(features=random_embeddings, issue_types={"class_imbalance": {}})
        summary = lab.get_issue_summary()
        assert len(summary) == 2
        assert "class_imbalance" in summary["issue_type"].values
        class_imbalance_summary = lab.get_issue_summary("class_imbalance")
        assert class_imbalance_summary["num_issues"].values[0] > 0

    def test_find_imbalance_issues_no_args(self, imbalance_labels):
        data = {"labels": imbalance_labels}
        lab = Datalab(data=data, label_name="labels")
        lab.find_issues(issue_types={"class_imbalance": {}})
        summary = lab.get_issue_summary()
        assert len(summary) == 1
        assert "class_imbalance" in summary["issue_type"].values
        class_imbalance_summary = lab.get_issue_summary("class_imbalance")
        assert class_imbalance_summary["num_issues"].values[0] > 0
