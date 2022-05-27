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

import numpy as np
import pytest
from cleanlab import rank
from cleanlab.internal.label_quality_utils import _subtract_confident_thresholds
from cleanlab.benchmarking.noise_generation import generate_noise_matrix_from_trace
from cleanlab.benchmarking.noise_generation import generate_noisy_labels
from cleanlab import count
from sklearn.neighbors import NearestNeighbors


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
    inv = count.compute_inv_noise_matrix(py, noise_matrix, ps=ps)

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
    scores = rank.get_normalized_margin_for_each_label(data["labels"], data["pred_probs"])
    label_errors = np.arange(len(data["labels"]))[data["label_errors_mask"]]
    least_confident_label = np.argmin(scores)
    most_confident_label = np.argmax(scores)
    assert least_confident_label in label_errors
    assert most_confident_label not in label_errors


def test_get_self_confidence_for_each_label():
    scores = rank.get_self_confidence_for_each_label(data["labels"], data["pred_probs"])
    label_errors = np.arange(len(data["labels"]))[data["label_errors_mask"]]
    least_confident_label = np.argmin(scores)
    most_confident_label = np.argmax(scores)
    assert least_confident_label in label_errors
    assert most_confident_label not in label_errors


def test_bad_rank_by_parameter_error():
    with pytest.raises(ValueError) as e:
        _ = rank.order_label_issues(
            label_issues_mask=data["label_errors_mask"],
            labels=data["labels"],
            pred_probs=data["pred_probs"],
            rank_by="not_a_real_method",
        )


@pytest.mark.parametrize(
    "scoring_method_func",
    [
        ("self_confidence", rank.get_self_confidence_for_each_label),
        ("normalized_margin", rank.get_normalized_margin_for_each_label),
        ("confidence_weighted_entropy", rank.get_confidence_weighted_entropy_for_each_label),
    ],
)
@pytest.mark.parametrize("adjust_pred_probs", [False, True])
def test_order_label_issues_using_scoring_func_ranking(scoring_method_func, adjust_pred_probs):

    # test all scoring methods with the scoring function

    method, scoring_func = scoring_method_func

    # check if method supports adjust_pred_probs
    # do not run the test below if the method does not support adjust_pred_probs
    # confidence_weighted_entropy scoring method does not support adjust_pred_probs
    if not (adjust_pred_probs == True and method == "confidence_weighted_entropy"):

        indices = np.arange(len(data["label_errors_mask"]))[
            data["label_errors_mask"]
        ]  # indices of label issues

        label_issues_indices = rank.order_label_issues(
            label_issues_mask=data["label_errors_mask"],
            labels=data["labels"],
            pred_probs=data["pred_probs"],
            rank_by=method,
            rank_by_kwargs={"adjust_pred_probs": adjust_pred_probs},
        )

        # test scoring function with scoring method passed as arg
        scores = rank.get_label_quality_scores(
            data["labels"],
            data["pred_probs"],
            method=method,
            adjust_pred_probs=adjust_pred_probs,
        )
        scores = scores[data["label_errors_mask"]]
        score_idx = sorted(list(zip(scores, indices)), key=lambda y: y[0])  # sort indices by score
        label_issues_indices2 = [z[1] for z in score_idx]
        assert all(
            label_issues_indices == label_issues_indices2
        ), f"Test failed with scoring method: {method}"

        # test individual scoring function
        # only test if adjust_pred_probs=False because the individual scoring functions do not adjust pred_probs
        if not adjust_pred_probs:
            scores = scoring_func(data["labels"], data["pred_probs"])
            scores = scores[data["label_errors_mask"]]
            score_idx = sorted(
                list(zip(scores, indices)), key=lambda y: y[0]
            )  # sort indices by score
            label_issues_indices3 = [z[1] for z in score_idx]
            assert all(
                label_issues_indices == label_issues_indices3
            ), f"Test failed with scoring method: {method}"


def test__subtract_confident_thresholds():
    labels = data["labels"]
    pred_probs = data["pred_probs"]

    # subtract confident class thresholds and renormalize
    pred_probs_adj = _subtract_confident_thresholds(labels, pred_probs)

    assert (pred_probs_adj > 0).all()  # all pred_prob are positive numbers
    assert (
        abs(1 - pred_probs_adj.sum(axis=1)) < 1e-6
    ).all()  # all pred_prob sum to 1 with some small precision error


@pytest.mark.parametrize(
    "method",
    [
        "self_confidence",
        "normalized_margin",
        "confidence_weighted_entropy",
    ],
)
@pytest.mark.parametrize("adjust_pred_probs", [False, True])
@pytest.mark.parametrize("weight_ensemble_members_by", ["uniform", "accuracy", "log_loss_search"])
def test_ensemble_scoring_func(method, adjust_pred_probs, weight_ensemble_members_by):

    labels = data["labels"]
    pred_probs = data["pred_probs"]

    # check if method supports adjust_pred_probs
    # do not run the test below if the method does not support adjust_pred_probs
    # confidence_weighted_entropy scoring method does not support adjust_pred_probs
    if not (adjust_pred_probs == True and method == "confidence_weighted_entropy"):

        # baseline scenario where all the pred_probs are the same in the ensemble list
        num_repeat = 3
        pred_probs_list = list(np.repeat([pred_probs], num_repeat, axis=0))

        # get label quality score with single pred_probs
        label_quality_scores = rank.get_label_quality_scores(
            labels, pred_probs, method=method, adjust_pred_probs=adjust_pred_probs
        )

        # get ensemble label quality score
        label_quality_scores_ensemble = rank.get_label_quality_ensemble_scores(
            labels,
            pred_probs_list,
            method=method,
            adjust_pred_probs=adjust_pred_probs,
            weight_ensemble_members_by=weight_ensemble_members_by,
        )

        # if all pred_probs in the list are the same, then ensemble score should be the same as the regular score
        # account for small precision error due to averaging of scores
        assert (
            abs(label_quality_scores - label_quality_scores_ensemble) < 1e-6
        ).all(), f"Test failed with scoring method: {method}"


def test_bad_weight_ensemble_members_by_parameter_error():
    with pytest.raises(ValueError) as e:

        labels = data["labels"]
        pred_probs = data["pred_probs"]

        # baseline scenario where all the pred_probs are the same in the ensemble list
        num_repeat = 3
        pred_probs_list = list(np.repeat([pred_probs], num_repeat, axis=0))

        _ = rank.get_label_quality_ensemble_scores(
            labels,
            pred_probs_list,
            weight_ensemble_members_by="not_a_real_method",  # this should raise ValueError
        )


def test_custom_weights():
    with pytest.raises(AssertionError) as e:

        labels = data["labels"]
        pred_probs = data["pred_probs"]

        # baseline scenario where all the pred_probs are the same in the ensemble list
        num_repeat = 3
        pred_probs_list = list(np.repeat([pred_probs], num_repeat, axis=0))

        # baseline scenario where custom_weights are uniform
        custom_weights = np.ones(num_repeat) / 3

        scores_custom_weights = rank.get_label_quality_ensemble_scores(
            labels,
            pred_probs_list,
            weight_ensemble_members_by="custom",
            custom_weights=custom_weights,  # this should raise AssertionError
        )

        scores_uniform_weights = rank.get_label_quality_ensemble_scores(
            labels, pred_probs_list, weight_ensemble_members_by="uniform"
        )

        # if custom_weights are uniform, then it should be the same as using weight_ensemble_members_by="uniform"
        assert (scores_custom_weights == scores_uniform_weights).all()


def test_empty_custom_weights_error():

    labels = data["labels"]
    pred_probs = data["pred_probs"]

    # baseline scenario where all the pred_probs are the same in the ensemble list
    num_repeat = 3
    pred_probs_list = list(np.repeat([pred_probs], num_repeat, axis=0))

    with pytest.raises(AssertionError) as e:
        _ = rank.get_label_quality_ensemble_scores(
            labels,
            pred_probs_list,
            weight_ensemble_members_by="custom",
            custom_weights=None,  # this should raise AssertionError because custom_weights is None
        )


def test_wrong_length_custom_weights_error():

    labels = data["labels"]
    pred_probs = data["pred_probs"]

    # baseline scenario where all the pred_probs are the same in the ensemble list
    num_repeat = 3
    pred_probs_list = list(np.repeat([pred_probs], num_repeat, axis=0))

    # baseline scenario where custom_weights are uniform
    custom_weights = np.ones(num_repeat) / 3

    with pytest.raises(AssertionError) as e:
        _ = rank.get_label_quality_ensemble_scores(
            labels,
            pred_probs_list,
            weight_ensemble_members_by="custom",
            custom_weights=custom_weights[
                1:
            ],  # this should raise AssertionError because length of custom_weights don't match len(pred_probs_list)
        )


def test_wrong_weight_ensemble_members_by_for_custom_weights_error():

    labels = data["labels"]
    pred_probs = data["pred_probs"]

    # baseline scenario where all the pred_probs are the same in the ensemble list
    num_repeat = 3
    pred_probs_list = list(np.repeat([pred_probs], num_repeat, axis=0))

    # baseline scenario where custom_weights are uniform
    custom_weights = np.ones(num_repeat) / 3

    with pytest.raises(ValueError) as e:
        _ = rank.get_label_quality_ensemble_scores(
            labels,
            pred_probs_list,
            weight_ensemble_members_by="accuracy",  # this should raise ValueError because custom_weights array is provided
            custom_weights=custom_weights,
        )


def test_bad_pred_probs_list_parameter_error():
    with pytest.raises(AssertionError) as e:

        labels = data["labels"]
        pred_probs = data["pred_probs"]

        # baseline scenario where all the pred_probs are the same in the ensemble list
        num_repeat = 3
        pred_probs_list = np.repeat(
            [pred_probs], num_repeat, axis=0
        )  # this should be a list not an array

        # AssertionError because pred_probs_list is an array
        _ = rank.get_label_quality_ensemble_scores(labels, pred_probs_list)

        # AssertionError because pred_probs_list is empty
        _ = rank.get_label_quality_ensemble_scores(labels=labels, pred_probs_list=[])


def test_unsupported_method_for_adjust_pred_probs():
    with pytest.raises(ValueError) as e:

        labels = data["labels"]
        pred_probs = data["pred_probs"]

        # method that do not support adjust_pred_probs
        # note: use a list of methods if there are multiple methods that do not support adjust_pred_probs
        method = "confidence_weighted_entropy"

        _ = rank.get_label_quality_scores(labels, pred_probs, adjust_pred_probs=True, method=method)


def test_get_knn_distance_ood_scores():

    X_train = data["X_train"]
    X_test = data["X_test"]

    # Create OOD datapoint
    X_ood = np.array([[999999999.0, 999999999.0]])

    # Add OOD datapoint to X_test
    X_test_with_ood = np.vstack([X_test, X_ood])

    # Fit nearest neighbors on X_train
    nbrs = NearestNeighbors(n_neighbors=5).fit(X_train)

    # Get KNN distance as OOD score
    k = 5
    knn_distance_ood_score = rank.get_knn_distance_ood_scores(
        features=X_test_with_ood, nbrs=nbrs, k=k
    )

    # Index of the datapoint with max ood score should correspond to the last element of X_test_with_ood
    idx_max_score = np.argsort(knn_distance_ood_score)[-1]
    assert idx_max_score == (X_test_with_ood.shape[0] - 1)

    # Get KNN distance as OOD score without passing k
    # By default k=min(10, max_k) is used where max_k is extracted from the given nbrs
    knn_distance_ood_score = rank.get_knn_distance_ood_scores(features=X_test_with_ood, nbrs=nbrs)

    # Index of the datapoint with max ood score should correspond to the last element of X_test_with_ood
    idx_max_score = np.argsort(knn_distance_ood_score)[-1]
    assert idx_max_score == (X_test_with_ood.shape[0] - 1)


def test_bad_k_for_get_knn_distance_ood_scores():

    X_train = data["X_train"]
    X_test = data["X_test"]

    # Create OOD datapoint
    X_ood = np.array([[999999999.0, 999999999.0]])

    # Add OOD datapoint to X_test
    X_test_with_ood = np.vstack([X_test, X_ood])

    # Fit nearest neighbors on X_train
    nbrs = NearestNeighbors(n_neighbors=5).fit(X_train)

    with pytest.raises(AssertionError) as e:
        _ = rank.get_knn_distance_ood_scores(
            features=X_test_with_ood, nbrs=nbrs, k=10  # this should throw an assertion error
        )
