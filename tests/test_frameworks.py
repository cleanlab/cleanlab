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

"""
Scripts to test cleanlab usage with deep learning frameworks:
pytorch, skorch, tensorflow, keras 
"""

import pytest
import warnings

# pytest.mark.filterwarnings is unable to catch filterbuffers library DeprecationWarning
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

from copy import deepcopy
import random
import tensorflow as tf
import torch
import skorch
import numpy as np
import pandas as pd

from cleanlab.classification import CleanLearning
from cleanlab.experimental.keras import KerasWrapper

SEED = 1
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(SEED)


def dataset_w_errors():
    num_classes = 2
    num_features = 3
    n = 50
    margin = 5
    X = np.vstack(
        [np.random.random((n, num_features)), np.random.random((n, num_features)) + margin]
    )
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # normalize columns
    y = np.array([0] * n + [1] * n)
    y_og = np.array(y)
    # Introduce label errors
    error_indices = [n - 3, n - 2, n - 1, n, n + 1, n + 2]
    for idx in error_indices:
        y[idx] = 1 - y[idx]  # Flip label

    return {
        "X": X,
        "y": y,
        "y_og": y_og,
        "error_indices": error_indices,
        "num_classes": num_classes,
        "num_features": num_features,
    }


def make_rare_label(data):
    """Makes one label really rare in the dataset."""
    data = deepcopy(data)
    y = data["y"]
    class0_inds = np.where(y == 0)[0]
    if len(class0_inds) < 1:
        raise ValueError("Class 0 too rare already")
    class0_inds_remove = class0_inds[1:]
    if len(class0_inds_remove) > 0:
        y[class0_inds_remove] = 1
    data["y"] = y
    return data


DATA = dataset_w_errors()
DATA_RARE_LABEL = make_rare_label(DATA)


@pytest.mark.parametrize("batch_size", [1, 32])
def test_tensorflow(batch_size, data=DATA, hidden_units=128):
    dataset_tf = tf.data.Dataset.from_tensor_slices((data["X"], data["y"])).batch(batch_size)

    model = KerasWrapper(
        [
            tf.keras.layers.Dense(
                hidden_units, input_shape=[data["num_features"]], activation="relu"
            ),
            tf.keras.layers.Dense(data["num_classes"]),
        ],
    )

    # Test base model works:
    model.fit(
        X=dataset_tf,
        y=data["y"],
        epochs=2,
    )
    preds_base = model.predict_proba(dataset_tf)

    # Test Cleanlearning performs well:
    cl = CleanLearning(model)
    cl.fit(dataset_tf, data["y"], clf_kwargs={"epochs": 10}, clf_final_kwargs={"epochs": 15})

    preds = cl.predict(dataset_tf)
    err = np.sum(preds != data["y_og"]) / len(data["y_og"])
    assert err < 1e-3

    issue_indices = list(cl.label_issues_df[cl.label_issues_df["is_label_issue"]].index.values)
    assert issue_indices == data["error_indices"]


@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.filterwarnings("ignore")
def test_tensorflow_rarelabel(batch_size, data=DATA_RARE_LABEL, hidden_units=8):
    dataset_tf = tf.data.Dataset.from_tensor_slices((data["X"], data["y"])).batch(batch_size)

    model = KerasWrapper(
        [
            tf.keras.layers.Dense(
                hidden_units, input_shape=[data["num_features"]], activation="relu"
            ),
            tf.keras.layers.Dense(data["num_classes"]),
        ],
    )
    # Test Cleanlearning works:
    cl = CleanLearning(model)
    cl.fit(dataset_tf, data["y"], clf_kwargs={"epochs": 10}, clf_final_kwargs={"epochs": 15})
    preds = cl.predict(dataset_tf)


def test_torch(data=DATA, hidden_units=128):
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(data["X"]).float(), torch.from_numpy(data["y"])
    )

    class TorchNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(data["num_features"], hidden_units),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_units, data["num_classes"]),
            )

        def forward(self, X):
            return self.layers(X)

    # Test base model works:
    net = skorch.NeuralNet(TorchNetwork, criterion=torch.nn.CrossEntropyLoss)
    net.fit(dataset, data["y"], epochs=2)
    preds_base = net.predict(dataset)

    # Test Cleanlearning performs well:
    net = skorch.NeuralNet(TorchNetwork, criterion=torch.nn.CrossEntropyLoss)
    cl = CleanLearning(net)
    cl.fit(dataset, data["y"], clf_kwargs={"epochs": 30}, clf_final_kwargs={"epochs": 40})

    preds = cl.predict(dataset).argmax(axis=1)
    err = np.sum(preds != data["y_og"]) / len(data["y_og"])
    assert err < 1e-3

    issue_indices = list(cl.label_issues_df[cl.label_issues_df["is_label_issue"]].index.values)
    assert issue_indices == data["error_indices"]


@pytest.mark.filterwarnings("ignore")
def test_torch_rarelabel(data=DATA_RARE_LABEL, hidden_units=8):
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(data["X"]).float(), torch.from_numpy(data["y"])
    )

    class TorchNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(data["num_features"], hidden_units),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_units, data["num_classes"]),
            )

        def forward(self, X):
            return self.layers(X)

    # Test Cleanlearning works:
    net = skorch.NeuralNet(TorchNetwork, criterion=torch.nn.CrossEntropyLoss)
    cl = CleanLearning(net)
    cl.fit(dataset, data["y"], clf_kwargs={"epochs": 2})
    pred_probs = cl.predict(dataset)
