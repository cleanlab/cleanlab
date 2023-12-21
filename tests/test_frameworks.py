# Copyright (C) 2017-2023  Cleanlab Inc.
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
Scripts to test cleanlab usage with various ML frameworks:
pytorch, skorch, tensorflow, keras, fasttext
"""

import pytest
import warnings

# pytest.mark.filterwarnings is unable to catch filterbuffers library DeprecationWarning
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

import sys
import os
import wget
from copy import deepcopy
import random
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF warnings on some systems
if os.name == "nt":  # check if we are on Windows
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ensure tensorflows runs on CPU

import tensorflow as tf
import torch
import skorch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from cleanlab.classification import CleanLearning
from cleanlab.models.keras import KerasWrapperSequential, KerasWrapperModel
from cleanlab.internal.util import format_labels


def python_version_ok():  # tensorflow and torch do not play nice with older Python
    version = sys.version_info
    return (version.major >= 3) and (version.minor >= 7)


def run_fasttext_test():
    # run test only if os enviroment is set of true and os is not Windows
    return os.environ.get("TEST_FASTTEXT") == "true" and os.name != "nt"


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

    if os.name == "nt":  # running on Windows
        # numpy converts to int32 instead of int64 on Windows, incompatible with neural nets
        # https://github.com/numpy/numpy/issues/17640
        X = np.float64(X)
        y = np.int64(y)
        y_og = np.int64(y_og)

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


SEED = 1
np.random.seed(SEED)
random.seed(SEED)
if python_version_ok():
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(SEED)

DATA = dataset_w_errors()
DATA_RARE_LABEL = make_rare_label(DATA)


@pytest.mark.slow
@pytest.mark.skipif("not python_version_ok()", reason="need at least python 3.7")
@pytest.mark.parametrize("batch_size,shuffle_config", [(1, 0), (32, 0), (32, 1), (32, 2)])
def test_tensorflow_sequential(batch_size, shuffle_config, data=DATA, hidden_units=128):
    dataset_tf = tf.data.Dataset.from_tensor_slices((data["X"], data["y"]))
    if shuffle_config == 0:  # proper shuffling for SGD
        dataset_shuffled = dataset_tf.shuffle(buffer_size=len(data["X"]))
    elif shuffle_config == 1:  # shuffling for datasets that don't fit in memory
        dataset_shuffled = dataset_tf.shuffle(buffer_size=60)
    else:
        dataset_shuffled = dataset_tf  # no shuffling

    dataset_og_order = dataset_tf.batch(batch_size)
    dataset_tf = dataset_shuffled.batch(batch_size)

    model = KerasWrapperSequential(
        [
            tf.keras.layers.Dense(
                hidden_units, input_shape=[data["num_features"]], activation="relu"
            ),
            tf.keras.layers.Dense(data["num_classes"]),
        ],
    )

    model.summary()

    # Test base model works:
    model.fit(
        X=dataset_tf,
        y=data["y"],
        epochs=2,
    )
    preds_base = model.predict_proba(dataset_tf)

    # Test CleanLearning performs well:
    cl = CleanLearning(model)
    cl.fit(dataset_tf, data["y"], clf_kwargs={"epochs": 10}, clf_final_kwargs={"epochs": 15})

    preds = cl.predict(dataset_og_order)
    err = np.sum(preds != data["y_og"]) / len(data["y_og"])
    issue_indices = list(cl.label_issues_df[cl.label_issues_df["is_label_issue"]].index.values)
    assert issue_indices == data["error_indices"]
    assert err < 1e-3

    # Test wrapper works with numpy array
    cl = CleanLearning(model)
    cl.fit(data["X"], data["y"])


@pytest.mark.slow
@pytest.mark.skipif("not python_version_ok()", reason="need at least python 3.7")
@pytest.mark.parametrize("batch_size,shuffle_config", [(1, 0), (32, 0), (32, 1), (32, 2)])
def test_tensorflow_functional(batch_size, shuffle_config, data=DATA, hidden_units=64):
    dataset_tf = tf.data.Dataset.from_tensor_slices((data["X"], data["y"]))
    if shuffle_config == 0:  # proper shuffling for SGD
        dataset_shuffled = dataset_tf.shuffle(buffer_size=len(data["X"]))
    elif shuffle_config == 1:  # shuffling for datasets that don't fit in memory
        dataset_shuffled = dataset_tf.shuffle(buffer_size=60)
    else:
        dataset_shuffled = dataset_tf  # no shuffling

    dataset_og_order = dataset_tf.batch(batch_size)
    dataset_tf = dataset_shuffled.batch(batch_size)

    def make_model(num_features, num_classes):
        inputs = tf.keras.Input(shape=(num_features,))
        x = tf.keras.layers.Dense(hidden_units, activation="relu")(inputs)
        outputs = tf.keras.layers.Dense(num_classes)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="test_model")

        return model

    model = KerasWrapperModel(
        make_model,
        model_kwargs={"num_features": data["num_features"], "num_classes": data["num_classes"]},
    )

    model.summary()

    # Test base model works:
    model.fit(
        X=dataset_tf,
        y=data["y"],
        epochs=2,
    )
    preds_base = model.predict_proba(dataset_tf)

    # Test CleanLearning performs well:
    cl = CleanLearning(model)
    cl.fit(dataset_tf, data["y"], clf_kwargs={"epochs": 10}, clf_final_kwargs={"epochs": 15})

    preds = cl.predict(dataset_og_order)
    err = np.sum(preds != data["y_og"]) / len(data["y_og"])
    issue_indices = list(cl.label_issues_df[cl.label_issues_df["is_label_issue"]].index.values)
    assert len(set(issue_indices) & set(data["error_indices"])) != 0
    assert err < 1e-3

    # Test wrapper works with numpy array
    cl = CleanLearning(model)
    cl.fit(data["X"], data["y"])


@pytest.mark.slow
@pytest.mark.skipif("not python_version_ok()", reason="need at least python 3.7")
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.filterwarnings("ignore")
def test_tensorflow_rarelabel(batch_size, data=DATA_RARE_LABEL, hidden_units=8):
    dataset_tf = tf.data.Dataset.from_tensor_slices((data["X"], data["y"])).batch(batch_size)

    model = KerasWrapperSequential(
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


@pytest.mark.slow
def test_keras_sklearn_compatability(data=DATA, hidden_units=32):
    # test pipeline on Sequential API
    model = KerasWrapperSequential(
        [
            tf.keras.layers.Dense(128, input_shape=[data["num_features"]], activation="relu"),
            tf.keras.layers.Dense(data["num_classes"]),
        ],
    )

    pipeline = Pipeline([("scale", StandardScaler()), ("net", model)])
    pipeline.fit(data["X"], data["y"])
    preds = pipeline.predict(data["X"])

    # test gridsearch on Sequential API
    model = KerasWrapperSequential(
        [
            tf.keras.layers.Dense(
                hidden_units, input_shape=[data["num_features"]], activation="relu"
            ),
            tf.keras.layers.Dense(data["num_classes"]),
        ],
    )

    params = {"batch_size": [32, 64], "epochs": [2, 3]}
    gs = GridSearchCV(
        model, params, refit=False, cv=3, verbose=2, scoring="accuracy", error_score="raise"
    )
    gs.fit(data["X"], data["y"])

    # test pipeline on functional API
    def make_model(num_features, num_classes):
        inputs = tf.keras.Input(shape=(num_features,))
        x = tf.keras.layers.Dense(64, activation="relu")(inputs)
        outputs = tf.keras.layers.Dense(num_classes)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="test_model")

        return model

    model = KerasWrapperModel(
        make_model,
        model_kwargs={"num_features": data["num_features"], "num_classes": data["num_classes"]},
    )

    pipeline = Pipeline([("scale", StandardScaler()), ("net", model)])
    pipeline.fit(data["X"], data["y"])
    preds = pipeline.predict(data["X"])

    # test gridsearch on Sequential API
    model = KerasWrapperModel(
        make_model,
        model_kwargs={"num_features": data["num_features"], "num_classes": data["num_classes"]},
    )

    params = {"batch_size": [32, 64], "epochs": [2, 3]}
    gs = GridSearchCV(
        model, params, refit=False, cv=3, verbose=2, scoring="accuracy", error_score="raise"
    )
    gs.fit(data["X"], data["y"])


@pytest.mark.skipif("not python_version_ok()", reason="need at least python 3.7")
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
    skorch_config = {"criterion": torch.nn.CrossEntropyLoss, "optimizer": torch.optim.Adam}
    net = skorch.NeuralNet(TorchNetwork, **skorch_config)
    net.fit(dataset, data["y"], epochs=2)
    preds_base = net.predict(dataset)

    # Test Cleanlearning performs well:
    net = skorch.NeuralNet(TorchNetwork, **skorch_config)
    cl = CleanLearning(net)
    cl.fit(dataset, data["y"], clf_kwargs={"epochs": 30}, clf_final_kwargs={"epochs": 60})

    preds = cl.predict(dataset).argmax(axis=1)
    err = np.sum(preds != data["y_og"]) / len(data["y_og"])
    issue_indices = list(cl.label_issues_df[cl.label_issues_df["is_label_issue"]].index.values)
    assert issue_indices == data["error_indices"]
    assert err < 1e-3


@pytest.mark.skipif("not python_version_ok()", reason="need at least python 3.7")
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


# test fasttext only if not on windows and environment variable TEST_FASTTEXT has been set to "true"
@pytest.mark.skipif(
    "not run_fasttext_test()", reason="fasttext is not easily pip install-able on windows"
)
def test_fasttext():
    from cleanlab.models.fasttext import FastTextClassifier, data_loader

    dir = "tests/fasttext_data"
    if not os.path.isdir(dir):
        os.makedirs(dir)

    try:
        if not os.path.isfile("tests/fasttext_data/tweets_train.txt"):
            wget.download(
                "http://s.cleanlab.ai/tweets_fasttext/tweets_train.txt", "tests/fasttext_data"
            )
        if not os.path.isfile("tests/fasttext_data/tweets_test.txt"):
            wget.download(
                "http://s.cleanlab.ai/tweets_fasttext/tweets_test.txt", "tests/fasttext_data"
            )
    except:
        raise RuntimeError(
            "Download failed (potentially due to lack of internet connection or invalid url). "
            "To skip this unittest, set the env variable TEST_FASTTEXT = false."
        )

    labels = np.ravel([x[0] for x in data_loader("tests/fasttext_data/tweets_train.txt")])
    labels = [lab[9:] for lab in labels]
    labels, label_map = format_labels(labels)
    X = np.array(range(len(labels)))

    # test basic fasttext methods
    ftc = FastTextClassifier(
        train_data_fn="tests/fasttext_data/tweets_train.txt",
        test_data_fn="tests/fasttext_data/tweets_test.txt",
    )
    ftc.fit()
    pred_labels = ftc.predict()
    pred_probs = ftc.predict_proba()

    # test CleanLearning
    ftc = FastTextClassifier(
        train_data_fn="tests/fasttext_data/tweets_train.txt",
        test_data_fn="tests/fasttext_data/tweets_test.txt",
    )
    cl = CleanLearning(ftc)

    issues = cl.find_label_issues(X=X, labels=labels)
    cl.fit(X=X, labels=labels, label_issues=issues)
    pred_labels = cl.predict()
    pred_probs = cl.predict_proba()
