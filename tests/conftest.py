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

from typing import Union

import pytest
import numpy as np
import sklearn
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from cleanlab import count
from cleanlab.benchmarking.noise_generation import (
    generate_noise_matrix_from_trace,
    generate_noisy_labels,
)
from cleanlab.internal.util import train_val_split


def multilabel_py(y):
    unique_labels = np.unique(y, axis=0)
    n = y.shape[0]
    multilabel_counts = [
        np.sum(np.prod([label[i] == y[:, i] for i in range(y.shape[1])], axis=0))
        for label in unique_labels
    ]
    return np.array(multilabel_counts) / float(n)


DATASET_KWARGS = {
    "n_samples": 500,
    "n_features": 3,
    "n_classes": 3,
    "n_labels": 2,
    "length": 500,
    "allow_unlabeled": True,
    "sparse": False,
    "return_indicator": "dense",
    "return_distributions": True,
}


def get_split_generator(labels, cv):
    unique_labels = np.unique(labels, axis=0)
    label_to_index = {tuple(label): i for i, label in enumerate(unique_labels)}
    multilabel_ids = np.array([label_to_index[tuple(label)] for label in labels])
    split_generator = cv.split(X=multilabel_ids, y=multilabel_ids)
    return split_generator


def train_fold(X, labels, *, clf, pred_probs, cv_train_idx, cv_holdout_idx):
    clf_copy = sklearn.base.clone(clf)
    X_train_cv, X_holdout_cv, s_train_cv, _ = train_val_split(
        X, labels, cv_train_idx, cv_holdout_idx
    )
    clf_copy.fit(X_train_cv, s_train_cv)
    pred_probs[cv_holdout_idx] = clf_copy.predict_proba(X_holdout_cv)


def get_cross_validated_multilabel_pred_probs(X, labels, *, clf, cv):
    split_generator = get_split_generator(labels, cv)
    pred_probs = np.zeros(shape=labels.shape)
    for cv_train_idx, cv_holdout_idx in split_generator:
        train_fold(
            X,
            labels,
            clf=clf,
            pred_probs=pred_probs,
            cv_train_idx=cv_train_idx,
            cv_holdout_idx=cv_holdout_idx,
        )
    return pred_probs


@pytest.fixture
def multilabeled_data(
    *,
    datasets_kwargs: dict = DATASET_KWARGS,
    avg_trace: float = 0.8,
    seed: int = 3,
    test_size: Union[float, int] = 0.5,
    verbose: bool = False,
    cv_n_folds: int = 5,
) -> dict:
    if "random_state" in datasets_kwargs and datasets_kwargs["random_state"] != seed:
        datasets_kwargs["random_state"] = seed

    X, y, *_ = make_multilabel_classification(**datasets_kwargs)

    # Normalize features
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    # Count unique labels
    unique_labels = np.unique(y, axis=0)
    if verbose:
        print(f"unique_labels.shape={unique_labels.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )

    # Compute p(true_label=k)
    py = multilabel_py(y_train)

    m = len(unique_labels)
    trace = avg_trace * m
    noise_matrix = generate_noise_matrix_from_trace(
        m,
        trace=trace,
        py=py,
        valid_noise_matrix=True,
        seed=seed,
    )

    # Map labels to to unique label indices
    label_to_index = {tuple(label): i for i, label in enumerate(unique_labels)}
    y_train_index = np.array([label_to_index[tuple(label)] for label in y_train])

    # Generate our noisy labels using the noise_matrix.
    s_index = generate_noisy_labels(y_train_index, noise_matrix)
    ps = np.bincount(s_index) / float(len(s_index))
    s = np.array([unique_labels[i] for i in s_index])

    # Compute inverse noise matrix
    inv = count.compute_inv_noise_matrix(py, noise_matrix, ps=ps)

    clf = OneVsRestClassifier(LogisticRegression())
    kf = sklearn.model_selection.StratifiedKFold(
        n_splits=cv_n_folds,
        shuffle=True,
        random_state=seed,
    )
    pred_probs = get_cross_validated_multilabel_pred_probs(X_train, s, clf=clf, cv=kf)

    label_errors_mask = s_index != y_train_index

    return {
        "X_train": X_train,
        "true_labels_train": y_train,
        "X_test": X_test,
        "true_labels_test": y_test,
        "labels": s,
        "label_errors_mask": label_errors_mask,
        "ps": ps,
        "py": py,
        "noise_matrix": noise_matrix,
        "inverse_noise_matrix": inv,
        "pred_probs": pred_probs,
        "m": m,
        "n": datasets_kwargs.get("n_samples", X.shape[0]),
    }
