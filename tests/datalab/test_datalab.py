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
from cleanlab.experimental.datalab.datalab import (
    Datalab,
    IssueManager,
    HealthIssueManager,
    LabelIssueManager,
    _IssueManagerFactory,
)

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
import numpy as np
import pandas as pd

from pathlib import Path

import pytest


LABEL_NAME = "star"
SEED = 42


@pytest.fixture
def dataset():
    data_dict = {
        "id": [
            "7bd227d9-afc9-11e6-aba1-c4b301cdf627",
            "7bd22905-afc9-11e6-a5dc-c4b301cdf627",
            "7bd2299c-afc9-11e6-85d6-c4b301cdf627",
            "7bd22a26-afc9-11e6-9309-c4b301cdf627",
            "7bd22aba-afc9-11e6-8293-c4b301cdf627",
        ],
        "package_name": [
            "com.mantz_it.rfanalyzer",
            "com.mantz_it.rfanalyzer",
            "com.mantz_it.rfanalyzer",
            "com.mantz_it.rfanalyzer",
            "com.mantz_it.rfanalyzer",
        ],
        "review": [
            "Great app! The new version now works on my Bravia Android TV which is great as it's right by my rooftop aerial cable. The scan feature would be useful...any ETA on when this will be available? Also the option to import a list of bookmarks e.g. from a simple properties file would be useful.",
            "Great It's not fully optimised and has some issues with crashing but still a nice app  especially considering the price and it's open source.",
            "Works on a Nexus 6p I'm still messing around with my hackrf but it works with my Nexus 6p  Trond usb-c to usb host adapter. Thanks!",
            "The bandwidth seemed to be limited to maximum 2 MHz or so. I tried to increase the bandwidth but not possible. I purchased this is because one of the pictures in the advertisement showed the 2.4GHz band with around 10MHz or more bandwidth. Is it not possible to increase the bandwidth? If not  it is just the same performance as other free APPs.",
            "Works well with my Hackrf Hopefully new updates will arrive for extra functions",
        ],
        "date": [
            "October 12 2016",
            "August 23 2016",
            "August 04 2016",
            "July 25 2016",
            "July 22 2016",
        ],
        "star": [4, 4, 5, 3, 5],
        "version_id": [1487, 1487, 1487, 1487, 1487],
    }
    return Dataset.from_dict(data_dict)


@pytest.fixture
def lab(dataset):
    return Datalab(data=dataset, label_name=LABEL_NAME)


@pytest.fixture
def pred_probs(dataset):
    np.random.seed(SEED)
    return np.random.rand(len(dataset), 3)


def test_datalab_class(dataset):
    lab = Datalab(dataset, LABEL_NAME)
    # Has the right attributes
    for attr in ["data", "label_name", "_labels", "info", "issues"]:
        assert hasattr(lab, attr), f"Missing attribute {attr}"


def test_datalab_invalid_datasetdict():
    with pytest.raises(AssertionError) as e:
        datadict = DatasetDict({"train": dataset, "test": dataset})
        Datalab(datadict, LABEL_NAME)  # type: ignore
        assert "Please pass a single dataset, not a DatasetDict." in str(e)


def test_data_features_and_labels(dataset):
    lab = Datalab(dataset, LABEL_NAME)
    assert all(lab._labels == np.array([1, 1, 2, 0, 2]))


class TestDatalab:
    """Tests for the Datalab class."""

    @pytest.fixture
    def lab(self, dataset):
        return Datalab(data=dataset, label_name=LABEL_NAME)

    def tmp_path(self):
        # A path for temporarily saving the instance during tests.
        # This is a workaround for the fact that the Datalab class
        # does not have a save method.
        return Path(__file__).parent / "tmp.pkl"

    def test_attributes(self, dataset):
        lab = Datalab(dataset, LABEL_NAME)
        # Has the right attributes
        for attr in ["data", "label_name", "_labels", "info", "issues"]:
            assert hasattr(lab, attr), f"Missing attribute {attr}"

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
        assert lab.issues is None
        assert lab.results is None, "Results should be None before calling find_issues"
        lab.find_issues(pred_probs=pred_probs, issue_types=issue_types)
        assert lab.issues is not None, "Issues weren't updated"
        assert isinstance(lab.issues, pd.DataFrame), "Issues should by in a dataframe"
        assert all(lab.issues.predicted_label == np.array([1, 0, 1, 2, 0]))

        if issue_types is None:
            assert isinstance(lab.results, float), "Results should be a float"
            assert isinstance(lab.info.get("summary", None), dict), "Summary should be a dict"

    def test_save(self, lab, tmp_path):
        """Test that the save and load methods work."""
        lab.save(tmp_path)
        assert tmp_path.exists(), "File was not saved"

    def test_pickle(self, lab, tmp_path):
        """Test that the class can be pickled."""
        pickle_file = os.path.join(tmp_path, "lab.pkl")
        with open(pickle_file, "wb") as f:
            pickle.dump(lab, f)
        with open(pickle_file, "rb") as f:
            lab2 = pickle.load(f)

        assert lab2.label_name == "star"

    def test_load(self, lab, tmp_path):
        """Test that the save and load methods work."""
        lab.save(tmp_path)
        loaded_lab = Datalab.load(tmp_path)
        assert loaded_lab.data.data == lab.data.data
        assert loaded_lab.label_name == lab.label_name
        assert all(loaded_lab._labels == lab._labels)
        assert loaded_lab._label_map == lab._label_map
        assert loaded_lab.info == lab.info
        assert loaded_lab.issues == lab.issues


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


@pytest.mark.parametrize(
    "issue_manager_class",
    [HealthIssueManager, LabelIssueManager],
    ids=["HealthIssueManager", "LabelIssueManager"],
)
class TestIssueManager:
    @pytest.fixture
    def issue_manager(self, lab, issue_manager_class):
        return issue_manager_class(datalab=lab)

    def test_init(self, lab, issue_manager_class):
        """Test that the init method works."""
        issue_manager = issue_manager_class(datalab=lab)
        assert issue_manager.datalab == lab

    def test_find_issues(self, lab, pred_probs, issue_manager):
        """Test that the find_issues method works."""
        info_before = lab.info.copy()
        _ = issue_manager.find_issues(pred_probs=pred_probs)

        assert lab.info != info_before, "Info should be updated"

    def test_update_info(self, lab, pred_probs, issue_manager):
        """Test that the update_info method works."""
        info_before = lab.info.copy()
        issues = issue_manager.find_issues(pred_probs=pred_probs)
        issue_manager.update_info(issues=issues)

        assert lab.info != info_before, "Info should be updated"
