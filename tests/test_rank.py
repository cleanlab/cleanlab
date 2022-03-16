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

from cleanlab import rank
import numpy as np
from cleanlab.noise_generation import generate_noise_matrix_from_trace
from cleanlab.noise_generation import generate_noisy_labels
from cleanlab import count
import pytest


def make_data(
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
        local_data.append(np.random.multivariate_normal(
            mean=means[idx], cov=covs[idx], size=sizes[idx]))
        test_data.append(np.random.multivariate_normal(
            mean=means[idx], cov=covs[idx], size=sizes[idx]))
        labels.append(np.array([idx for i in range(sizes[idx])]))
        test_labels.append(np.array([idx for i in range(sizes[idx])]))
    X_train = np.vstack(local_data)
    true_labels_train = np.hstack(labels)
    X_test = np.vstack(test_data)
    true_labels_test = np.hstack(test_labels)

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
    s = generate_noisy_labels(true_labels_train, noise_matrix)
    ps = np.bincount(s) / float(len(s))

    # Compute inverse noise matrix
    inv = count.compute_inv_noise_matrix(py, noise_matrix, ps)

    # Estimate pred_probs
    latent = count.estimate_py_noise_matrices_and_cv_pred_proba(
        X=X_train,
        labels=s,
        cv_n_folds=3,
    )

    label_errors_mask = s != true_labels_train

    return {
        "X_train": X_train,
        "true_labels_train": true_labels_train,
        "X_test": X_test,
        "true_labels_test": true_labels_test,
        "labels": s,
        "label_errors_mask": label_errors_mask,
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


# Global to be used by all test methods. Only compute this once for speed.
data = make_data()


def test_get_normalized_margin_for_each_label():
    scores = rank.get_normalized_margin_for_each_label(data['labels'], data['pred_probs'])
    label_errors = np.arange(len(data['labels']))[data['label_errors_mask']]
    least_confident_label = np.argmin(scores)
    most_confident_label = np.argmax(scores)
    assert (least_confident_label in label_errors)
    assert (most_confident_label not in label_errors)


def test_get_self_confidence_for_each_label():
    scores = rank.get_self_confidence_for_each_label(data['labels'], data['pred_probs'])
    label_errors = np.arange(len(data['labels']))[data['label_errors_mask']]
    least_confident_label = np.argmin(scores)
    most_confident_label = np.argmax(scores)
    assert (least_confident_label in label_errors)
    assert (most_confident_label not in label_errors)


def test_bad_rank_by_parameter_error():
    with pytest.raises(ValueError) as e:
        _ = rank.order_label_issues(
            label_issues_mask=data['label_errors_mask'],
            labels=data['labels'],
            pred_probs=data['pred_probs'],
            rank_by="not_a_real_method"
        )


def test_order_label_issues_using_margin_ranking():
    label_issues_indices = rank.order_label_issues(
        label_issues_mask=data['label_errors_mask'],
        labels=data['labels'],
        pred_probs=data['pred_probs'],
        rank_by="normalized_margin"
    )
    scores = rank.get_normalized_margin_for_each_label(data['labels'], data['pred_probs'])
    indices = np.arange(len(scores))[data['label_errors_mask']]
    scores = scores[data['label_errors_mask']]
    score_idx = sorted(list(zip(scores, indices)), key=lambda y: y[0])  # sort indices by score
    label_issues_indices2 = [z[1] for z in score_idx]
    assert(all(label_issues_indices == label_issues_indices2))


def test_order_label_issues_using_self_confidence_ranking():
    label_issues_indices = rank.order_label_issues(
        label_issues_mask=data['label_errors_mask'],
        labels=data['labels'],
        pred_probs=data['pred_probs'],
        rank_by="self_confidence"
    )
    scores = rank.get_self_confidence_for_each_label(data['labels'], data['pred_probs'])
    indices = np.arange(len(scores))[data['label_errors_mask']]
    scores = scores[data['label_errors_mask']]
    score_idx = sorted(list(zip(scores, indices)), key=lambda y: y[0])  # sort indices by score
    label_issues_indices2 = [z[1] for z in score_idx]
    assert(all(label_issues_indices == label_issues_indices2))
