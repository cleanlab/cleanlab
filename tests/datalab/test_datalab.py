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

from datasets import load_dataset
import numpy as np
import pandas as pd

import pytest


LABEL_NAME = "star"
SEED = 42


@pytest.fixture
def dataset():
    return load_dataset("lhoestq/demo1", split="train")


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
        dataset = load_dataset("lhoestq/demo1")
        Datalab(dataset, LABEL_NAME)
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
