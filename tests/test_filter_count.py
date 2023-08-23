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

import cleanlab.multilabel_classification.dataset
from cleanlab import count, filter
from cleanlab.count import (
    get_confident_thresholds,
    estimate_py_and_noise_matrices_from_probabilities,
)
from cleanlab.internal.latent_algebra import compute_inv_noise_matrix
from cleanlab.benchmarking.noise_generation import generate_noise_matrix_from_trace
from cleanlab.benchmarking.noise_generation import generate_noisy_labels
from cleanlab.internal.util import value_counts
from cleanlab.internal.multilabel_utils import int2onehot
from cleanlab.experimental.label_issues_batched import find_label_issues_batched
import numpy as np
import scipy
import pytest
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from tempfile import mkdtemp
import os.path as path


def make_data(
    sparse=False,
    means=[[3, 2], [7, 7], [0, 8]],
    covs=[[[5, -1.5], [-1.5, 1]], [[1, 0.5], [0.5, 4]], [[5, 1], [1, 5]]],
    sizes=[80, 40, 40],
    avg_trace=0.8,
    seed=1,  # set to None for non-reproducible randomness
):
    np.random.seed(seed=seed)

    m = len(means)  # number of classes
    n = sum(sizes)
    local_data = []
    labels = []
    test_data = []
    test_labels = []

    for idx in range(m):
        local_data.append(
            np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx])
        )
        test_data.append(
            np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx])
        )
        labels.append(np.array([idx for i in range(sizes[idx])]))
        test_labels.append(np.array([idx for i in range(sizes[idx])]))
    X_train = np.vstack(local_data)
    true_labels_train = np.hstack(labels)
    X_test = np.vstack(test_data)
    true_labels_test = np.hstack(test_labels)

    if sparse:
        X_train = scipy.sparse.csr_matrix(X_train)
        X_test = scipy.sparse.csr_matrix(X_test)

    # Compute p(true_label=k)
    py = np.bincount(true_labels_train) / float(len(true_labels_train))

    noise_matrix = generate_noise_matrix_from_trace(
        m,
        trace=avg_trace * m,
        py=py,
        valid_noise_matrix=True,
        seed=seed,
    )

    # Generate our noisy labels using the noise_matrix.
    labels = generate_noisy_labels(true_labels_train, noise_matrix)
    ps = np.bincount(labels) / float(len(labels))

    # Compute inverse noise matrix
    inv = compute_inv_noise_matrix(py, noise_matrix, ps=ps)

    # Estimate pred_probs
    latent = count.estimate_py_noise_matrices_and_cv_pred_proba(
        X=X_train,
        labels=labels,
        cv_n_folds=3,
    )
    return {
        "X_train": X_train,
        "true_labels_train": true_labels_train,
        "X_test": X_test,
        "true_labels_test": true_labels_test,
        "labels": labels,
        "ps": ps,
        "py": py,
        "noise_matrix": noise_matrix,
        "inverse_noise_matrix": inv,
        "est_py": latent[0],
        "est_nm": latent[1],
        "est_inv": latent[2],
        "cj": latent[3],
        "pred_probs": latent[4],
        "m": m,
        "n": n,
    }


def make_multilabel_data(
    means=[[-5, 2], [0, 2], [-3, 6]],
    covs=[[[3, -1.5], [-1.5, 1]], [[5, -1.5], [-1.5, 1]], [[3, -1.5], [-1.5, 1]]],
    boxes_coordinates=[[-3.5, 0, -1.5, 1.7], [-1, 3, 2, 4], [-5, 2, -3, 4], [-3, 2, -1, 4]],
    box_multilabels=[[0, 1], [1, 2], [0, 2], [0, 1, 2]],
    sizes=[100, 80, 100],
    avg_trace=0.9,
    seed=1,
):
    np.random.seed(seed=seed)
    num_classes = len(means)
    m = num_classes + len(
        box_multilabels
    )  # number of classes by treating each multilabel as 1 unique label
    n = sum(sizes)
    local_data = []
    labels = []
    test_data = []
    test_labels = []
    for i in range(0, len(means)):
        local_data.append(np.random.multivariate_normal(mean=means[i], cov=covs[i], size=sizes[i]))
        test_data.append(np.random.multivariate_normal(mean=means[i], cov=covs[i], size=sizes[i]))
        test_labels += [[i]] * sizes[i]
        labels += [[i]] * sizes[i]

    def make_multi(X, Y, bx1, by1, bx2, by2, label_list):
        ll = np.array([bx1, by1])  # lower-left
        ur = np.array([bx2, by2])  # upper-right

        inidx = np.all(np.logical_and(X.tolist() >= ll, X.tolist() <= ur), axis=1)
        for i in range(0, len(Y)):
            if inidx[i]:
                Y[i] = label_list
        return Y

    X_train = np.vstack(local_data)
    X_test = np.vstack(test_data)

    for i in range(0, len(box_multilabels)):
        bx1, by1, bx2, by2 = boxes_coordinates[i]
        multi_label = box_multilabels[i]
        labels = make_multi(X_train, labels, bx1, by1, bx2, by2, multi_label)
        test_labels = make_multi(X_test, test_labels, bx1, by1, bx2, by2, multi_label)

    d = {}
    for i in labels:
        if str(i) not in d:
            d[str(i)] = len(d)
    inv_d = {v: k for k, v in d.items()}

    labels_idx = [d[str(i)] for i in labels]

    py = np.bincount(labels_idx) / float(len(labels_idx))

    noise_matrix = generate_noise_matrix_from_trace(
        m,
        trace=avg_trace * m,
        py=py,
        valid_noise_matrix=True,
        seed=seed,
    )
    noisy_labels_idx = generate_noisy_labels(labels_idx, noise_matrix)
    noisy_labels = [eval(inv_d[i]) for i in noisy_labels_idx]
    ps = np.bincount(labels_idx) / float(len(labels_idx))
    inv = compute_inv_noise_matrix(py, noise_matrix, ps=ps)

    y_train = int2onehot(noisy_labels, K=num_classes)
    clf = MultiOutputClassifier(LogisticRegression())
    pyi = cross_val_predict(clf, X_train, y_train, method="predict_proba")
    pred_probs = np.zeros(y_train.shape)
    for i, p in enumerate(pyi):
        pred_probs[:, i] = p[:, 1]
    cj = count.compute_confident_joint(labels=noisy_labels, pred_probs=pred_probs, multi_label=True)
    return {
        "X_train": X_train,
        "true_labels_train": labels,
        "X_test": X_test,
        "true_labels_test": test_labels,
        "labels": noisy_labels,
        "noise_matrix": noise_matrix,
        "inverse_noise_matrix": inv,
        "cj": cj,
        "pred_probs": pred_probs,
        "m": m,
        "n": n,
    }


# Global to be used by all test methods.
# Only compute this once for speed.
seed = 1
data = make_data(sparse=False, seed=1)
multilabel_data = make_multilabel_data(seed=1)
# Create some simple data to test
pred_probs_ = np.array(
    [
        [0.9, 0.1, 0],
        [0.6, 0.2, 0.2],
        [0.1, 0, 0.9],
        [0.1, 0.8, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ]
)
labels_ = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 2])


def test_exact_prune_count():
    remove = 5
    s = data["labels"]
    noise_idx = filter.find_label_issues(
        labels=s,
        pred_probs=data["pred_probs"],
        num_to_remove_per_class=remove,
        filter_by="prune_by_class",
    )
    assert all(value_counts(s[noise_idx]) == remove)


@pytest.mark.parametrize("n_jobs", [None, 1, 2])
def test_pruning_both(n_jobs):
    remove = 5
    s = data["labels"]
    class_idx = filter.find_label_issues(
        labels=s,
        pred_probs=data["pred_probs"],
        num_to_remove_per_class=remove,
        filter_by="prune_by_class",
        n_jobs=n_jobs,
    )
    nr_idx = filter.find_label_issues(
        labels=s,
        pred_probs=data["pred_probs"],
        num_to_remove_per_class=remove,
        filter_by="prune_by_noise_rate",
        n_jobs=n_jobs,
    )
    both_idx = filter.find_label_issues(
        labels=s,
        pred_probs=data["pred_probs"],
        num_to_remove_per_class=remove,
        filter_by="both",
        n_jobs=n_jobs,
    )
    assert all(s[both_idx] == s[class_idx & nr_idx])


@pytest.mark.parametrize(
    "filter_by",
    ["prune_by_noise_rate", "prune_by_class", "both", "confident_learning", "predicted_neq_given"],
)
def test_prune_on_small_data(filter_by):
    data = make_data(sizes=[3, 3, 3])
    noise_idx = filter.find_label_issues(
        labels=data["labels"], pred_probs=data["pred_probs"], filter_by=filter_by
    )
    # Num in each class <= 1 (when splitting for cross validation). Nothing should be pruned.
    assert not any(noise_idx)


def test_calibrate_joint():
    dataset = data
    cj = count.compute_confident_joint(
        labels=dataset["labels"],
        pred_probs=dataset["pred_probs"],
        calibrate=False,
    )
    calibrated_cj = count.calibrate_confident_joint(
        confident_joint=cj,
        labels=dataset["labels"],
    )

    # Check calibration
    label_counts = np.bincount(data["labels"])
    assert all(calibrated_cj.sum(axis=1).round().astype(int) == label_counts)
    assert len(dataset["labels"]) == int(round(np.sum(calibrated_cj)))

    calibrated_cj2 = count.compute_confident_joint(
        labels=dataset["labels"],
        pred_probs=dataset["pred_probs"],
        calibrate=True,
    )

    # Check equivalency
    assert np.all(calibrated_cj == calibrated_cj2)


def test_calibrate_joint_multilabel():
    dataset = multilabel_data
    cj = count.compute_confident_joint(
        labels=dataset["labels"],
        pred_probs=dataset["pred_probs"],
        multi_label=True,
        calibrate=False,
    )
    calibrated_cj = count.calibrate_confident_joint(
        confident_joint=cj,
        labels=dataset["labels"],
        multi_label=True,
    )
    y_one = int2onehot(dataset["labels"], K=dataset["pred_probs"].shape[1])
    # Check calibration
    for class_num in range(0, len(calibrated_cj)):
        label_counts = np.bincount(y_one[:, class_num])
        assert all(calibrated_cj[class_num].sum(axis=1).round().astype(int) == label_counts)
        assert len(dataset["labels"]) == int(round(np.sum(calibrated_cj[class_num])))
    calibrated_cj2 = count.compute_confident_joint(
        labels=dataset["labels"],
        pred_probs=dataset["pred_probs"],
        calibrate=True,
        multi_label=True,
    )
    # Check equivalency
    assert np.all(calibrated_cj == calibrated_cj2)
    assert calibrated_cj.shape == calibrated_cj2.shape == (3, 2, 2)


@pytest.mark.parametrize("use_confident_joint", [True, False])
def test_estimate_joint(use_confident_joint):
    dataset = data
    joint = count.estimate_joint(
        labels=dataset["labels"],
        pred_probs=dataset["pred_probs"],
        confident_joint=dataset["cj"] if use_confident_joint else None,
    )

    # Check that joint sums to 1.
    assert abs(np.sum(joint) - 1.0) < 1e-6


def test_estimate_joint_multilabel():
    dataset = multilabel_data
    cj = count.compute_confident_joint(
        labels=dataset["labels"], pred_probs=dataset["pred_probs"], multi_label=True
    )
    assert cj.shape == (3, 2, 2)
    joint = count.estimate_joint(
        labels=dataset["labels"],
        pred_probs=dataset["pred_probs"],
        confident_joint=cj,
        multi_label=True,
    )
    joint_2 = count.estimate_joint(
        labels=dataset["labels"],
        pred_probs=dataset["pred_probs"],
        multi_label=True,
    )
    assert np.array_equal(joint, joint_2)
    assert joint.shape == (3, 2, 2)
    # Check that each joint sums to 1.
    for j in joint:
        assert abs(np.sum(j) - 1.0) < 1e-6


def test_confidence_thresholds():
    dataset = data
    cft = get_confident_thresholds(pred_probs=dataset["pred_probs"], labels=dataset["labels"])
    assert cft.shape == (dataset["pred_probs"].shape[1],)


def test_confidence_thresholds_multilabel():
    dataset = multilabel_data
    cft = get_confident_thresholds(
        pred_probs=dataset["pred_probs"], labels=dataset["labels"], multi_label=True
    )
    assert cft.shape == (dataset["pred_probs"].shape[1], 2)


def test_compute_confident_joint():
    cj = count.compute_confident_joint(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
    )
    # Check that confident joint doesn't overcount number of examples.
    assert np.sum(cj) <= data["n"]
    # Check that confident joint is correct shape
    assert np.shape(cj) == (data["m"], data["m"])


def test_estimate_latent_py_method():
    for py_method in ["cnt", "eqn", "marginal"]:
        py, nm, inv = count.estimate_latent(
            confident_joint=data["cj"],
            labels=data["labels"],
            py_method=py_method,
        )
        assert sum(py) - 1 < 1e-4
    try:
        py, nm, inv = count.estimate_latent(
            confident_joint=data["cj"],
            labels=data["labels"],
            py_method="INVALID",
        )
    except ValueError as e:
        assert "should be" in str(e)
        with pytest.raises(ValueError) as e:
            py, nm, inv = count.estimate_latent(
                confident_joint=data["cj"],
                labels=data["labels"],
                py_method="INVALID",
            )


def test_estimate_latent_converge():
    py, nm, inv = count.estimate_latent(
        confident_joint=data["cj"],
        labels=data["labels"],
        converge_latent_estimates=True,
    )

    py2, nm2, inv2 = count.estimate_latent(
        confident_joint=data["cj"],
        labels=data["labels"],
        converge_latent_estimates=False,
    )
    # Check results are similar, but not the same.
    assert np.any(inv != inv2)
    assert np.any(py != py2)
    assert np.all(abs(py - py2) < 0.1)
    assert np.all(abs(nm - nm2) < 0.1)
    assert np.all(abs(inv - inv2) < 0.1)


@pytest.mark.parametrize("sparse", [True, False])
def test_estimate_noise_matrices(sparse):
    data = make_data(sparse=sparse, seed=seed)
    nm, inv = count.estimate_noise_matrices(
        X=data["X_train"],
        labels=data["labels"],
    )
    assert np.all(abs(nm - data["est_nm"]) < 0.1)
    assert np.all(abs(inv - data["est_inv"]) < 0.1)


def test_pruning_reduce_prune_counts():
    """Make sure it doesnt remove when its not supposed to"""
    cj = np.array(
        [
            [325, 16, 22],
            [47, 178, 10],
            [36, 8, 159],
        ]
    )
    cj2 = filter._reduce_prune_counts(cj, frac_noise=1.0)
    assert np.all(cj == cj2)


def test_pruning_keep_at_least_n_per_class():
    """Make sure it doesnt remove when its not supposed to"""
    cj = np.array(
        [
            [325, 16, 22],
            [47, 178, 10],
            [36, 8, 159],
        ]
    )
    prune_count_matrix = filter._keep_at_least_n_per_class(
        prune_count_matrix=cj.T,
        n=5,
    )
    assert np.all(cj == prune_count_matrix.T)


def test_pruning_order_method():
    order_methods = ["self_confidence", "normalized_margin"]
    results = []
    for method in order_methods:
        results.append(
            filter.find_label_issues(
                labels=data["labels"],
                pred_probs=data["pred_probs"],
                return_indices_ranked_by=method,
            )
        )
    assert len(results[0]) == len(results[1])


@pytest.mark.parametrize("multi_label", [True, False])
@pytest.mark.parametrize("use_dataset_function", [True, False])
@pytest.mark.parametrize(
    "filter_by", ["prune_by_noise_rate", "prune_by_class", "both", "confident_learning"]
)
@pytest.mark.parametrize(
    "return_indices_ranked_by",
    [None, "self_confidence", "normalized_margin", "confidence_weighted_entropy"],
)
def test_find_label_issues_multi_label(
    multi_label, use_dataset_function, filter_by, return_indices_ranked_by
):
    """Note: argmax_not_equal method is not compatible with multi_label == True"""
    dataset = multilabel_data if multi_label else data
    if multi_label:
        if use_dataset_function:
            noise_idx = cleanlab.multilabel_classification.filter.find_label_issues(
                labels=dataset["labels"],
                pred_probs=dataset["pred_probs"],
                filter_by=filter_by,
                return_indices_ranked_by=return_indices_ranked_by,
            )
        else:
            with pytest.warns(DeprecationWarning):
                noise_idx = filter.find_label_issues(
                    labels=dataset["labels"],
                    pred_probs=dataset["pred_probs"],
                    filter_by=filter_by,
                    multi_label=multi_label,
                    return_indices_ranked_by=return_indices_ranked_by,
                )
    else:
        noise_idx = filter.find_label_issues(
            labels=dataset["labels"],
            pred_probs=dataset["pred_probs"],
            filter_by=filter_by,
            multi_label=multi_label,
            return_indices_ranked_by=return_indices_ranked_by,
        )

    if return_indices_ranked_by is not None:
        noise_bool = np.zeros(len(dataset["labels"])).astype(bool)
        noise_bool[noise_idx] = True
        noise_idx = noise_bool
    acc = np.mean(
        (
            np.array(dataset["labels"], dtype=np.object_)
            != np.array(dataset["true_labels_train"], dtype=np.object_)
        )
        == noise_idx
    )
    # Make sure cleanlab does reasonably well finding the errors.
    # acc is the accuracy of detecting a label error.
    assert acc > 0.85


@pytest.mark.parametrize(
    "confident_joint",
    [
        None,
        [[[1, 0], [0, 4]], [[3, 0], [0, 2]], [[3, 0], [1, 1]], [[3, 1], [0, 1]]],
        [[1, 1, 0, 2], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]],
    ],
)
@pytest.mark.parametrize(
    "return_indices_ranked_by",
    [None, "self_confidence", "normalized_margin", "confidence_weighted_entropy"],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_find_label_issues_multi_label_small(confident_joint, return_indices_ranked_by):
    pred_probs = np.array(
        [
            [0.9, 0.1, 0.0, 0.4],
            [0.7, 0.8, 0.2, 0.3],
            [0.9, 0.8, 0.4, 0.2],
            [0.1, 0.1, 0.8, 0.3],
            [0.4, 0.5, 0.1, 0.1],
        ]
    )
    labels = [[0], [0, 1], [0, 1], [2], [0, 2, 3]]
    cj = count.compute_confident_joint(labels=labels, pred_probs=pred_probs, multi_label=True)
    assert cj.shape == (4, 2, 2)
    noise_idx = filter.find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        multi_label=True,
        confident_joint=np.array(confident_joint) if confident_joint else None,
        return_indices_ranked_by=return_indices_ranked_by,
    )
    noise_idx2 = filter.find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        multi_label=True,
        confident_joint=cj,
        return_indices_ranked_by=return_indices_ranked_by,
    )
    if confident_joint is not None:
        noise_idx3 = filter.find_label_issues(
            labels=labels,
            pred_probs=pred_probs,
            multi_label=True,
            confident_joint=cj[::-1],
            return_indices_ranked_by=return_indices_ranked_by,
        )

    def _idx_to_bool(idx):
        noise_bool = np.zeros(len(labels)).astype(bool)
        noise_bool[idx] = True
        return noise_bool

    if return_indices_ranked_by is not None:
        noise_idx = _idx_to_bool(noise_idx)
        noise_idx2 = _idx_to_bool(noise_idx2)
        if confident_joint is not None:
            noise_idx3 = _idx_to_bool(noise_idx3)
    expected_output = [False, False, False, False, True]
    assert noise_idx.tolist() == noise_idx2.tolist() == expected_output
    if confident_joint is not None:
        assert noise_idx3.tolist() != expected_output


@pytest.mark.parametrize("return_indices_of_off_diagonals", [True, False])
def test_confident_learning_filter(return_indices_of_off_diagonals):
    dataset = data
    if return_indices_of_off_diagonals:
        cj, indices = count.compute_confident_joint(
            labels=dataset["labels"],
            pred_probs=dataset["pred_probs"],
            calibrate=False,
            return_indices_of_off_diagonals=True,
        )
        # Check that the number of 'label issues' found in off diagonals
        # matches the off diagonals of the uncalibrated confident joint

        assert len(indices) == (np.sum(cj) - np.trace(cj))
    else:
        cj = count.compute_confident_joint(
            labels=dataset["labels"],
            pred_probs=dataset["pred_probs"],
            calibrate=False,
            return_indices_of_off_diagonals=return_indices_of_off_diagonals,
        )

        assert np.trace(cj) > -1


@pytest.mark.parametrize("return_indices_of_off_diagonals", [True, False])
def test_confident_learning_filter_multilabel(return_indices_of_off_diagonals):
    dataset = multilabel_data

    if return_indices_of_off_diagonals:
        cj, indices = count.compute_confident_joint(
            labels=dataset["labels"],
            pred_probs=dataset["pred_probs"],
            calibrate=False,
            return_indices_of_off_diagonals=True,
            multi_label=True,
        )
        # Check that the number of 'label issues' found in off diagonals
        # matches the off diagonals of the uncalibrated confident joint

        for c, ind in zip(cj, indices):
            assert len(ind) == (np.sum(c) - np.trace(c))
    else:
        cj = count.compute_confident_joint(
            labels=dataset["labels"],
            pred_probs=dataset["pred_probs"],
            calibrate=False,
            return_indices_of_off_diagonals=return_indices_of_off_diagonals,
            multi_label=True,
        )
        for c in cj:
            assert np.trace(c) > -1


def test_predicted_neq_given_filter():
    pred_probs = np.array(
        [
            [0.9, 0.1, 0],
            [0.6, 0.2, 0.2],
            [0.3, 0.3, 0.4],
            [0.1, 0.1, 0.8],
            [0.4, 0.5, 0.1],
        ]
    )
    s = np.array([0, 0, 1, 1, 2])
    label_issues = filter.find_predicted_neq_given(s, pred_probs)
    assert all(label_issues == [False, False, True, True, True])

    label_issues = filter.find_predicted_neq_given(labels_, pred_probs_)
    assert all(
        label_issues
        == np.array([False, False, True, False, False, False, False, False, False, False])
    )


def test_predicted_neq_given_filter_multilabel():
    pred_probs = np.array(
        [
            [0.9, 0.1, 0.0, 0.4],
            [0.7, 0.8, 0.2, 0.3],
            [0.9, 0.8, 0.4, 0.2],
            [0.1, 0.1, 0.8, 0.3],
            [0.4, 0.5, 0.1, 0.1],
        ]
    )
    labels = [[0], [0, 1], [0, 1], [2], [0, 2, 3]]
    label_issues = filter.find_predicted_neq_given(labels, pred_probs, multi_label=True)
    assert all(label_issues == [False, False, False, False, True])


@pytest.mark.parametrize("calibrate", [True, False])
@pytest.mark.parametrize("filter_by", ["prune_by_noise_rate", "prune_by_class", "both"])
def test_find_label_issues_using_argmax_confusion_matrix(calibrate, filter_by):
    label_issues = filter.find_label_issues_using_argmax_confusion_matrix(
        labels_, pred_probs_, calibrate=calibrate, filter_by=filter_by
    )
    assert all(
        label_issues
        == np.array([False, False, True, False, False, False, False, False, False, False])
    )


@pytest.mark.filterwarnings("ignore:WARNING!")
def test_find_label_issue_filters_match_origin_functions():
    label_issues = filter.find_label_issues(labels_, pred_probs_, filter_by="predicted_neq_given")
    label_issues2 = filter.find_predicted_neq_given(labels_, pred_probs_)
    assert all(label_issues == label_issues2)
    label_issues3 = filter.find_label_issues(
        labels_, pred_probs_, filter_by="confident_learning", verbose=True
    )
    label_issues4 = filter.find_predicted_neq_given(labels_, pred_probs_)
    assert all(label_issues3 == label_issues4)
    try:
        _ = filter.find_label_issues(
            labels_,
            pred_probs_,
            filter_by="predicted_neq_given",
            num_to_remove_per_class=[1] * pred_probs_.shape[1],
        )
    except ValueError as e:
        assert "not supported" in str(e)


def test_num_label_issues_different_estimation_types():
    # these numbers are hardcoded as data[] does not create a difference in both functions
    y = np.array([0, 1, 1, 1, 1, 0, 0, 1, 0])
    pred_probs = np.array(
        [
            [0.7110397298505661, 0.2889602701494339],
            [0.6367131487519773, 0.36328685124802274],
            [0.7571834730987641, 0.24281652690123584],
            [0.6394163729473307, 0.3605836270526695],
            [0.5853684039196656, 0.4146315960803345],
            [0.6675968116482668, 0.33240318835173316],
            [0.7240647829106976, 0.2759352170893023],
            [0.740474240697777, 0.25952575930222266],
            [0.7148252196621883, 0.28517478033781196],
        ]
    )

    n3 = count.num_label_issues(
        labels=y,
        pred_probs=pred_probs,
        estimation_method="off_diagonal_calibrated",
    )

    n2 = count.num_label_issues(
        labels=y,
        pred_probs=pred_probs,
        estimation_method="off_diagonal",
    )

    f2 = filter.find_label_issues(labels=y, pred_probs=pred_probs, filter_by="confident_learning")

    assert np.sum(f2) == n2
    assert n3 != n2


def test_find_label_issues_same_value():
    f1 = filter.find_label_issues(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        filter_by="confident_learning",
    )

    f2 = filter.find_label_issues(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        filter_by="low_self_confidence",
    )

    f3 = filter.find_label_issues(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        filter_by="low_normalized_margin",
    )

    assert np.sum(f1) == np.sum(f2)
    assert np.sum(f2) == np.sum(f3)


@pytest.mark.filterwarnings()
def test_num_label_issues():
    cj_calibrated_off_diag_sum = data["cj"].sum() - data["cj"].trace()

    n1 = count.num_label_issues(  # should throw warning as cj is passed in but also recalculated
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        confident_joint=data["cj"],
        estimation_method="off_diagonal_calibrated",
    )

    n2 = count.num_label_issues(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        estimation_method="off_diagonal_calibrated",
    )  # this should calculate and calibrate the confident joint into same matrix as data["cj"]

    n_custom = count.num_label_issues(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        confident_joint=data["cj"],
        estimation_method="off_diagonal_custom",
    )

    ones_joint = np.ones_like(data["cj"])
    n_custom_bad = count.num_label_issues(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        confident_joint=ones_joint,
        estimation_method="off_diagonal_custom",
    )

    # data["cj"] is already calibrated and recalibrating it should not change the values
    assert n2 == cj_calibrated_off_diag_sum
    # should calculate and calibrate the confident joint into same matrix as data["cj"]
    assert n1 == n2
    # estimation_method='off_diagonal_custom' should use the passed in confident joint correctly
    assert n_custom == n1
    assert n_custom_bad != n1

    f = filter.find_label_issues(  # this should throw warning since cj passed in and filter by confident_learning
        labels=data["labels"], pred_probs=data["pred_probs"], confident_joint=data["cj"]
    )

    assert sum(f) == 35

    f1 = filter.find_label_issues(  # this should throw warning since cj passed in and filter by confident_learning
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        filter_by="confident_learning",
        confident_joint=data["cj"],
    )

    n = count.num_label_issues(  # should throw warning as cj is passed in but also recalculated
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        confident_joint=data["cj"],
        estimation_method="off_diagonal",
    )

    n3 = count.num_label_issues(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
    )

    assert sum(f1) == n3  # values should be equivalent for `filter_by='confident_learning'`
    assert n == n3  # passing in cj should not affect calculation

    # check wrong estimation_method throws ValueError
    try:
        count.num_label_issues(
            labels=data["labels"],
            pred_probs=data["pred_probs"],
            estimation_method="not_a_real_method",
        )
    except Exception as e:
        assert "not a valid estimation method" in str(e)
        with pytest.raises(ValueError) as e:
            count.num_label_issues(
                labels=data["labels"],
                pred_probs=data["pred_probs"],
                estimation_method="not_a_real_method",
            )

    # check not passing in cj with estimation_method_custom throws ValueError
    try:
        count.num_label_issues(
            labels=data["labels"],
            pred_probs=data["pred_probs"],
            estimation_method="off_diagonal_custom",
        )
    except Exception as e:
        assert "you need to provide pre-calculated" in str(e)
        with pytest.raises(ValueError) as e:
            count.num_label_issues(
                labels=data["labels"],
                pred_probs=data["pred_probs"],
                estimation_method="off_diagonal_custom",
            )


@pytest.mark.parametrize("confident_joint", [None, True])
def test_num_label_issues_multilabel(confident_joint):
    dataset = multilabel_data
    n = count.num_label_issues(
        labels=dataset["labels"],
        pred_probs=dataset["pred_probs"],
        confident_joint=dataset["cj"] if confident_joint else None,
        estimation_method="off_diagonal",
        multi_label=True,
    )
    f = filter.find_label_issues(
        labels=dataset["labels"],
        pred_probs=dataset["pred_probs"],
        confident_joint=dataset["cj"] if confident_joint else None,
        filter_by="confident_learning",
        multi_label=True,
    )
    assert sum(f) == n


def test_batched_label_issues():
    f1 = filter.find_label_issues(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        return_indices_ranked_by="self_confidence",
        filter_by="low_self_confidence",
    )
    f1_mask = filter.find_label_issues(
        labels=data["labels"], pred_probs=data["pred_probs"], filter_by="low_self_confidence"
    )
    f2 = find_label_issues_batched(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        batch_size=int(len(data["labels"]) / 4.0),
    )
    f3 = find_label_issues_batched(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        batch_size=int(len(data["labels"]) / 2.0),
        n_jobs=None,
    )
    f4 = find_label_issues_batched(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        batch_size=len(data["labels"]) + 100,
        n_jobs=4,
    )
    f5 = find_label_issues_batched(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        batch_size=1,
    )
    f_single = find_label_issues_batched(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        batch_size=len(data["labels"]),
        n_jobs=1,
    )
    f_single_mask = find_label_issues_batched(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        batch_size=len(data["labels"]),
        n_jobs=1,
        return_mask=True,
    )
    assert np.all(f4 == f5)
    assert np.all(f4 == f3)
    assert np.all(f4 == f2)
    assert np.all(f_single == f4)
    assert len(f2) == len(f1)
    # check jaccard similarity:
    intersection = len(list(set(f1).intersection(set(f2))))
    union = len(set(f1)) + len(set(f2)) - intersection
    assert float(intersection) / union > 0.95
    # check jaccard similarity for mask:
    f1_mask_indices = np.where(f1_mask)[0]
    f_single_mask_indices = np.where(f_single_mask)[0]
    intersection = len(list(set(f1_mask_indices).intersection(set(f_single_mask_indices))))
    union = len(set(f1_mask_indices)) + len(set(f_single_mask_indices)) - intersection
    assert float(intersection) / union > 0.95

    n1 = count.num_label_issues(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        estimation_method="off_diagonal_calibrated",
    )
    quality_score_kwargs = {"method": "normalized_margin"}
    num_issue_kwargs = {"estimation_method": "off_diagonal_calibrated"}
    extra_args = {
        "quality_score_kwargs": quality_score_kwargs,
        "num_issue_kwargs": num_issue_kwargs,
    }
    f5 = find_label_issues_batched(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        batch_size=int(len(data["labels"]) / 4.0),
        **extra_args,
    )
    f6 = find_label_issues_batched(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        batch_size=int(len(data["labels"]) / 2.0),
        n_jobs=None,
        **extra_args,
    )
    f7 = find_label_issues_batched(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        batch_size=len(data["labels"]) + 100,
        n_jobs=4,
        **extra_args,
    )
    f_single = find_label_issues_batched(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        batch_size=len(data["labels"]),
        n_jobs=1,
        **extra_args,
    )
    assert not np.array_equal(f5, f2)
    assert np.all(f7 == f5)
    assert np.all(f6 == f5)
    assert np.all(f_single == f5)
    assert np.abs(len(f5) - n1) < 2
    # Test batches loaded from file:
    labels_file = path.join(mkdtemp(), "labels.npy")
    pred_probs_file = path.join(mkdtemp(), "pred_probs.npy")
    np.save(labels_file, data["labels"])
    np.save(pred_probs_file, data["pred_probs"])
    f8 = find_label_issues_batched(
        labels_file=labels_file,
        pred_probs_file=pred_probs_file,
        batch_size=int(len(data["labels"]) / 4.0),
    )
    assert np.all(f8 == f3)


def test_issue_158():
    # ref: https://github.com/cleanlab/cleanlab/issues/158
    pred_probs = np.array(
        [
            [0.27916167, 0.14589103, 0.29264585, 0.28230144],
            [0.13429196, 0.12536383, 0.47943979, 0.26090442],
            [0.41348584, 0.13463275, 0.25845595, 0.19342546],
            [0.27753469, 0.12295569, 0.33125886, 0.26825075],
            [0.11649856, 0.11219034, 0.51857122, 0.25273988],
            [0.38010026, 0.25572261, 0.13305410, 0.23112304],
            [0.31583755, 0.13630690, 0.29246806, 0.25538750],
            [0.30240076, 0.16925207, 0.26499082, 0.26335636],
            [0.44524505, 0.27410085, 0.08305069, 0.19760341],
            [0.22903975, 0.07783631, 0.38035414, 0.31276980],
            [0.25071560, 0.12072900, 0.32551729, 0.30303812],
            [0.43809229, 0.14401381, 0.20839300, 0.20950090],
            [0.20749181, 0.11883556, 0.38402152, 0.28965111],
            [0.43840254, 0.13538447, 0.24518806, 0.18102493],
            [0.28504779, 0.10309750, 0.34258602, 0.26926868],
            [0.38425408, 0.29168969, 0.15181255, 0.17224368],
            [0.19339907, 0.10804265, 0.37570368, 0.32285460],
            [0.21509781, 0.07190167, 0.38914722, 0.32385330],
            [0.27040334, 0.13037840, 0.32842320, 0.27079507],
            [0.40590210, 0.16713560, 0.24889193, 0.17807036],
        ]
    )
    labels = np.array([3, 3, 1, 3, 0, 2, 2, 2, 0, 2, 2, 1, 0, 0, 0, 0, 2, 1, 3, 3])

    cj = count.compute_confident_joint(labels, pred_probs, calibrate=False)
    # should be no zeros on the diagonal
    assert np.all(cj.diagonal() != 0)

    py, noise_matrix, inv_noise_matrix = count.estimate_latent(cj, labels)
    # no nans anywhere
    assert not np.any(np.isnan(py))
    assert not np.any(np.isnan(noise_matrix))
    assert not np.any(np.isnan(inv_noise_matrix))


@pytest.mark.filterwarnings("ignore:May not flag all label issues")
def test_missing_classes():
    labels = np.array([0, 0, 2, 2])
    pred_probs = np.array(
        [[0.9, 0.0, 0.1, 0.0], [0.8, 0.0, 0.2, 0.0], [0.1, 0.0, 0.9, 0.0], [0.95, 0.0, 0.05, 0.0]]
    )
    issues = filter.find_label_issues(labels, pred_probs)
    assert np.all(issues == np.array([False, False, False, True]))
    # check results with pred-prob on missing classes = 0 match results without these missing classes in pred_probs
    pred_probs2 = pred_probs[:, list(sorted(np.unique(labels)))]
    labels2 = np.array([0, 0, 1, 1])
    issues2 = filter.find_label_issues(labels2, pred_probs2)
    assert all(issues2 == issues)
    # check this still works with nonzero pred_prob on missing class
    pred_probs3 = np.array(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.1, 0.1, 0.0],
            [0.1, 0.0, 0.9, 0.0],
            [0.9, 0.025, 0.025, 0.05],
        ]
    )
    issues3 = filter.find_label_issues(labels, pred_probs3)
    assert all(issues3 == issues)
    # check this works with n_jobs = 1
    issues4 = filter.find_label_issues(labels, pred_probs, n_jobs=1)
    assert all(issues4 == issues)
    # check this works with different filter_by
    for fb in [
        "prune_by_class",
        "prune_by_noise_rate",
        "both",
        "confident_learning",
        "predicted_neq_given",
    ]:
        assert all(filter.find_label_issues(labels, pred_probs, filter_by=fb) == issues)


@pytest.mark.filterwarnings("ignore:WARNING!")
def test_find_label_issues_match_multiprocessing():
    # minimal version of this test was run in test_missing_classes
    # here testing with larger input matrices

    # test with ground truth labels:
    n = 5000  # consider replacing this with larger value
    # some past bugs observed only with larger sample-sizes like n=200000
    m = 100
    labels = np.ones(n, dtype=int)
    labels[(n // 2) :] = 0
    pred_probs = np.zeros((n, 4))
    pred_probs[:, 0] = 0.95
    pred_probs[:, 1] = 0.05
    pred_probs[0, 0] = 0.94
    pred_probs[0, 1] = 0.06
    ground_truth = np.ones(n, dtype=bool)
    ground_truth[(n // 2) :] = False
    ground_truth[0] = False  # leave one example for min_example_per_class
    # TODO: consider also testing this line without psutil installed
    issues = filter.find_label_issues(labels, pred_probs)
    issues1 = filter.find_label_issues(labels, pred_probs, n_jobs=1)
    issues2 = filter.find_label_issues(labels, pred_probs, n_jobs=2)
    assert all(issues == ground_truth)
    assert all(issues == issues1)
    assert all(issues == issues2)
    issues = filter.find_label_issues(labels, pred_probs, filter_by="prune_by_class")
    issues1 = filter.find_label_issues(labels, pred_probs, n_jobs=1, filter_by="prune_by_class")
    issues2 = filter.find_label_issues(labels, pred_probs, n_jobs=2, filter_by="prune_by_class")
    assert all(issues == ground_truth)
    assert all(issues == issues1)
    assert all(issues == issues2)

    # test with random labels
    normalize = np.random.randint(low=1, high=100, size=[n, m], dtype=np.uint8)
    pred_probs = np.zeros((n, m))
    for i, col in enumerate(normalize):
        pred_probs[i] = col / np.sum(col)
    labels = np.repeat(np.arange(m), n // m)
    issues = filter.find_label_issues(labels, pred_probs)
    issues1 = filter.find_label_issues(labels, pred_probs, n_jobs=1)
    issues2 = filter.find_label_issues(labels, pred_probs, n_jobs=2)
    assert all(issues == issues1)
    assert all(issues == issues2)
    issues = filter.find_label_issues(labels, pred_probs, filter_by="prune_by_class")
    issues1 = filter.find_label_issues(labels, pred_probs, n_jobs=1, filter_by="prune_by_class")
    issues2 = filter.find_label_issues(labels, pred_probs, n_jobs=2, filter_by="prune_by_class")
    assert all(issues == issues1)
    assert all(issues == issues2)


@pytest.mark.parametrize(
    "return_indices_ranked_by",
    [None, "self_confidence", "normalized_margin", "confidence_weighted_entropy"],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_missing_classes_multilabel(return_indices_ranked_by):
    pred_probs = np.array(
        [
            [0.9, 0.1, 0.0, 0.4, 0.1],
            [0.7, 0.8, 0.2, 0.3, 0.1],
            [0.9, 0.8, 0.4, 0.2, 0.1],
            [0.1, 0.1, 0.8, 0.3, 0.1],
            [0.4, 0.5, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.2, 0.1, 0.1],
            [0.8, 0.1, 0.2, 0.1, 0.1],
        ]
    )
    labels = [[0], [0, 1], [0, 1], [2], [0, 2, 3], [], []]
    cj = count.compute_confident_joint(labels=labels, pred_probs=pred_probs, multi_label=True)
    assert cj.shape == (5, 2, 2)
    noise_idx = filter.find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        multi_label=True,
        return_indices_ranked_by=return_indices_ranked_by,
    )
    noise_idx2 = filter.find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        multi_label=True,
        confident_joint=cj,
        return_indices_ranked_by=return_indices_ranked_by,
    )

    def _idx_to_bool(idx):
        noise_bool = np.zeros(len(labels)).astype(bool)
        noise_bool[idx] = True
        return noise_bool

    if return_indices_ranked_by is not None:
        noise_idx = _idx_to_bool(noise_idx)
        noise_idx2 = _idx_to_bool(noise_idx2)

    expected_output = [False, False, False, False, True, False, True]
    assert noise_idx.tolist() == noise_idx2.tolist() == expected_output


def test_removing_class_consistent_results():
    # Note that only one label is class 1 (we're going to change it to class 2 later...)
    labels = np.array([0, 0, 0, 0, 1, 2, 2, 2])
    # Third example is a label error
    pred_probs = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.1, 0.1],
            [0.1, 0.0, 0.9],
            [0.9, 0.0, 0.1],
            [0.1, 0.3, 0.6],
            [0.1, 0.0, 0.9],
            [0.1, 0.0, 0.9],
            [0.1, 0.0, 0.9],
        ]
    )
    cj_with1 = count.compute_confident_joint(labels, pred_probs)
    issues_with1 = filter.find_label_issues(labels, pred_probs)

    labels_no1 = labels = np.array([0, 0, 0, 0, 2, 2, 2, 2])  # change 1 to 2 (class 1 is missing!)
    cj_no1 = count.compute_confident_joint(labels, pred_probs)
    issues_no1 = filter.find_label_issues(labels, pred_probs)

    assert np.all(issues_with1 == issues_no1)
    assert np.all(
        cj_with1
        == [
            [3, 0, 1],
            [0, 1, 0],
            [0, 0, 3],
        ]
    )
    # Check that the 1, 1 entry goes away and moves to 2, 2 (since we changed label 1 to 2)
    assert np.all(
        cj_no1
        == [
            [3, 0, 1],
            [0, 0, 0],
            [0, 0, 4],
        ]
    )


def test_estimate_py_and_noise_matrices_missing_classes():
    labels = np.array([0, 0, 2, 2])
    pred_probs = np.array(
        [[0.9, 0.0, 0.1, 0.0], [0.8, 0.0, 0.2, 0.0], [0.1, 0.0, 0.9, 0.0], [0.95, 0.0, 0.05, 0.0]]
    )
    (
        py,
        noise_matrix,
        inverse_noise_matrix,
        confident_joint,
    ) = estimate_py_and_noise_matrices_from_probabilities(labels, pred_probs)
    # check results with pred-prob on missing classes = 0 match results without these missing classes in pred_probs
    present_classes = list(sorted(np.unique(labels)))
    pred_probs2 = pred_probs[:, present_classes]
    labels2 = np.array([0, 0, 1, 1])
    (
        py2,
        noise_matrix2,
        inverse_noise_matrix2,
        confident_joint2,
    ) = estimate_py_and_noise_matrices_from_probabilities(labels2, pred_probs2)
    # These may be slightly off due to clipping to prevent division by 0:
    assert (np.isclose(py[present_classes], py2, atol=1e-5)).all()
    assert (
        np.isclose(confident_joint[np.ix_(present_classes, present_classes)], confident_joint2)
    ).all()
    assert (np.isclose(noise_matrix[np.ix_(present_classes, present_classes)], noise_matrix2)).all()
    assert (
        np.isclose(
            inverse_noise_matrix[np.ix_(present_classes, present_classes)], inverse_noise_matrix2
        )
    ).all()

    # check this still works with nonzero pred_prob on missing class
    pred_probs3 = np.array(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.1, 0.1, 0.0],
            [0.1, 0.0, 0.9, 0.0],
            [0.9, 0.025, 0.025, 0.05],
        ]
    )
    _ = estimate_py_and_noise_matrices_from_probabilities(labels, pred_probs3)


def test_low_filter_by_methods():
    dataset = data
    num_issues = count.num_label_issues(dataset["labels"], dataset["pred_probs"])

    # test filter by low_normalized_margin, check num issues is same as using count.num_label_issues
    label_issues_nm = filter.find_label_issues(
        dataset["labels"], dataset["pred_probs"], filter_by="low_normalized_margin"
    )
    assert sum(label_issues_nm) == num_issues

    # test filter by low_self_confidence, check num issues is same as using count.num_label_issues
    label_issues_sc = filter.find_label_issues(
        dataset["labels"],
        dataset["pred_probs"],
        filter_by="low_self_confidence",
        return_indices_ranked_by="normalized_margin",
    )
    assert len(label_issues_sc) == num_issues

    label_issues_sc_sort = filter.find_label_issues(
        dataset["labels"],
        dataset["pred_probs"],
        filter_by="low_self_confidence",
        return_indices_ranked_by="confidence_weighted_entropy",
    )
    assert set(label_issues_sc) == set(label_issues_sc_sort)


def test_low_filter_by_methods_multilabel():
    dataset = multilabel_data
    num_issues = count.num_label_issues(dataset["labels"], dataset["pred_probs"], multi_label=True)

    # test filter by low_normalized_margin, check num issues is same as using count.num_label_issues
    label_issues_nm = filter.find_label_issues(
        dataset["labels"],
        dataset["pred_probs"],
        filter_by="low_normalized_margin",
        multi_label=True,
    )
    assert sum(label_issues_nm) == num_issues

    # test filter by low_self_confidence, check num issues is same as using count.num_label_issues
    label_issues_sc = filter.find_label_issues(
        dataset["labels"],
        dataset["pred_probs"],
        filter_by="low_self_confidence",
        multi_label=True,
        return_indices_ranked_by="confidence_weighted_entropy",
    )
    assert len(label_issues_sc) == num_issues

    label_issues_sc_sort = filter.find_label_issues(
        dataset["labels"],
        dataset["pred_probs"],
        filter_by="low_self_confidence",
        multi_label=True,
        return_indices_ranked_by="self_confidence",
    )
    assert set(label_issues_sc) == set(label_issues_sc_sort)


@pytest.mark.parametrize(
    "filter_by",
    ["prune_by_noise_rate", "prune_by_class", "both", "confident_learning", "predicted_neq_given"],
)
def test_does_not_flag_correct_examples(filter_by):
    labels = data["labels"]
    pred_probs = data["pred_probs"]
    pred_labels = np.argmax(pred_probs, axis=1)

    label_issues_mask = filter.find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        filter_by=filter_by,
    )

    matching_mask = labels == pred_labels  # mask specifying whether label == prediction
    assert (
        any(label_issues_mask[matching_mask]) == False
    )  # make sure none of these are flagged as label error
