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


from cleanlab.experimental.datalab.datalab import Datalab

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
import numpy as np
import pandas as pd

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
    for attr in ["data", "label_name", "_labels", "info", "issues", "_silo"]:
        assert hasattr(lab, attr), f"Missing attribute {attr}"


def test_datalab_invalid_datasetdict():
    with pytest.raises(AssertionError) as e:
        datadict = DatasetDict({"train": dataset, "test": dataset})
        Datalab(datadict, LABEL_NAME)  # type: ignore
        assert "Please pass a single dataset, not a DatasetDict." in str(e)


def test_data_features_and_labels(dataset):
    lab = Datalab(dataset, LABEL_NAME)
    assert all(lab._labels == np.array([1, 1, 2, 0, 2]))


def test_find_issues(lab, pred_probs):
    assert lab.issues is None
    lab.find_issues(pred_probs=pred_probs)
    assert lab.issues is not None, "Issues weren't updated"
    assert isinstance(lab.issues, pd.DataFrame), "Issues should by in a dataframe"
    assert all(lab.issues.predicted_label == np.array([1, 0, 1, 2, 0]))


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


def test_getter_methods(lab, pred_probs):
    # Expect errors if we call getters before calling self.find_issues
    with pytest.raises(ValueError) as e:
        lab.get_health_score()
        assert "Health summary has not been computed" in str(e)

    with pytest.raises(ValueError) as e:
        lab.get_label_quality_score()
        assert "Labels errors have not been found yet" in str(e)

    # Call self.find_issues and check outputs
    lab.find_issues(pred_probs=pred_probs)
    assert isinstance(lab.get_health_score(), float)
    assert isinstance(lab.get_label_quality_score(), np.ndarray)
