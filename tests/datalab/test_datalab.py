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


import os
import pickle
from cleanlab.experimental.datalab.datalab import Datalab


from datasets.dataset_dict import DatasetDict
import numpy as np
import pandas as pd

from pathlib import Path

import pytest


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
        """Test that the method fetches the values from the info dict."""
        num_classes = lab.get_info("num_classes")
        assert num_classes == 3
        lab.info["num_classes"] = 4
        num_classes = lab.get_info("num_classes")
        assert num_classes == 4

    @pytest.mark.parametrize(
        "issue_types",
        [None, {"label": True}],
        ids=["Default issues", "Only label issues"],
    )
    def test_find_issues(self, lab, pred_probs, issue_types):
        assert lab.issues.empty, "Issues should be empty before calling find_issues"
        assert lab.issue_summary.empty, "Issue summary should be empty before calling find_issues"
        lab.find_issues(pred_probs=pred_probs, issue_types=issue_types)
        assert not lab.issues.empty, "Issues weren't updated"
        assert not lab.issue_summary.empty, "Issue summary wasn't updated"

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
            }
        )
        monkeypatch.setattr(lab, "issue_summary", mock_issue_summary)

        lab.find_issues(pred_probs=pred_probs)
        # Check that the issues dataframe has the right columns
        expected_issues_df = pd.DataFrame(
            {
                "is_foo_issue": mock_issues.is_foo_issue,
                "foo_score": mock_issues.foo_score,
                "is_label_issue": [False, False, False, False, False],
                "label_quality": [0.95071431, 0.15601864, 0.60111501, 0.70807258, 0.18182497],
            }
        )
        pd.testing.assert_frame_equal(lab.issues, expected_issues_df, check_exact=False)

        expected_issue_summary_df = pd.DataFrame(
            {
                "issue_type": ["foo", "label"],
                "score": [0.72, 0.6],
            }
        )
        pd.testing.assert_frame_equal(
            lab.issue_summary, expected_issue_summary_df, check_exact=False
        )

    def test_save(self, lab, tmp_path, monkeypatch):
        """Test that the save and load methods work."""
        lab.save(tmp_path)
        assert tmp_path.exists(), "File was not saved"
        assert (tmp_path / "data").is_dir(), "Data directory was not saved"

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
        assert loaded_lab.data.data == lab.data.data
        assert loaded_lab.label_name == lab.label_name
        assert all(loaded_lab._labels == lab._labels)
        assert loaded_lab._label_map == lab._label_map
        assert loaded_lab.info == lab.info
        pd.testing.assert_frame_equal(loaded_lab.issues, mock_issues)
        pd.testing.assert_frame_equal(loaded_lab.issue_summary, mock_issue_summary)

        # Load accepts a `Dataset`.
        loaded_lab = Datalab.load(tmp_path, data=dataset)
        assert loaded_lab.data.data == dataset.data

        # Misaligned dataset raises a ValueError
        with pytest.raises(ValueError) as excinfo:
            Datalab.load(tmp_path, data=dataset[:2])
            expected_error_msg = "Length of data (2) does not match length of labels (5)"
            assert expected_error_msg == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            Datalab.load(tmp_path, data=dataset.shuffle())
            expected_error_msg = (
                "Data has been modified since Lab was saved. Cannot load Lab with modified data."
            )
            assert expected_error_msg == str(excinfo.value)


def test_health_summary(lab, pred_probs):
    summary = lab._health_summary(pred_probs=pred_probs)
    assert isinstance(summary, dict), "Summary should be a dict"
    # Check that the summary has the right keys
    for key in [
        "overall_label_health_score",
        "joint",
        "classes_by_label_quality",
        "overlapping_classes",
    ]:
        assert key in summary.keys(), f"Summary missing key {key}"

    label_quality_df = summary["classes_by_label_quality"]
    class_names_in_df = label_quality_df["Class Name"].values
    # Make sure the class names appear in the dataframe
    assert (
        len(set(class_names_in_df) - set([4, 5, 3])) == 0
    ), "Class names don't match class indices in dataframe"
    assert (
        len(set(label_quality_df["Class Index"].values) - set([0, 1, 2])) == 0
    ), "Class indices don't match class indices in dataframe"
    # Make sure the class indices are properly mapped to class names in the dataframe
    assert (
        label_quality_df["Class Index"].map(lab._label_map).equals(label_quality_df["Class Name"])
    ), "Class indices don't match class names in dataframe"
