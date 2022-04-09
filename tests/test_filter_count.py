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

from cleanlab import count, filter
from cleanlab.internal.latent_algebra import compute_inv_noise_matrix
from cleanlab.benchmarking.noise_generation import generate_noise_matrix_from_trace
from cleanlab.benchmarking.noise_generation import generate_noisy_labels
from cleanlab.internal.util import value_counts
import numpy as np
import scipy
import pytest


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


# Global to be used by all test methods.
# Only compute this once for speed.
seed = 1
data = make_data(sparse=False, seed=1)

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
    cj = count.compute_confident_joint(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        calibrate=False,
    )
    calibrated_cj = count.calibrate_confident_joint(
        confident_joint=cj,
        labels=data["labels"],
    )
    label_counts = np.bincount(data["labels"])

    # Check calibration
    assert all(calibrated_cj.sum(axis=1).round().astype(int) == label_counts)
    assert len(data["labels"]) == int(round(np.sum(calibrated_cj)))

    calibrated_cj2 = count.compute_confident_joint(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        calibrate=True,
    )

    # Check equivalency
    assert np.all(calibrated_cj == calibrated_cj2)


@pytest.mark.parametrize("use_confident_joint", [True, False])
def test_estimate_joint(use_confident_joint):
    joint = count.estimate_joint(
        labels=data["labels"],
        pred_probs=data["pred_probs"],
        confident_joint=data["cj"] if use_confident_joint else None,
    )

    # Check that joint sums to 1.
    assert abs(np.sum(joint) - 1.0) < 1e-6


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
@pytest.mark.parametrize(
    "filter_by", ["prune_by_noise_rate", "prune_by_class", "both", "confident_learning"]
)
def test_find_label_issues_multi_label(multi_label, filter_by):
    """Note: argmax_not_equal method is not compatible with multi_label == True"""

    s_ml = [[z, data["true_labels_train"][i]] for i, z in enumerate(data["labels"])]
    noise_idx = filter.find_label_issues(
        labels=s_ml if multi_label else data["labels"],
        pred_probs=data["pred_probs"],
        filter_by=filter_by,
        multi_label=multi_label,
    )
    acc = np.mean((data["labels"] != data["true_labels_train"]) == noise_idx)
    # Make sure cleanlab does reasonably well finding the errors.
    # acc is the accuracy of detecting a label error.
    assert acc > 0.85


@pytest.mark.parametrize("return_indices_of_off_diagonals", [True, False])
def test_confident_learning_filter(return_indices_of_off_diagonals):
    if return_indices_of_off_diagonals:
        cj, indices = count.compute_confident_joint(
            labels=data["labels"],
            pred_probs=data["pred_probs"],
            calibrate=False,
            return_indices_of_off_diagonals=True,
        )
        # Check that the number of 'label issues' found in off diagonals
        # matches the off diagonals of the uncalibrated confident joint
        assert len(indices) == (np.sum(cj) - np.trace(cj))
    else:
        cj = count.compute_confident_joint(
            labels=data["labels"],
            pred_probs=data["pred_probs"],
            calibrate=False,
            return_indices_of_off_diagonals=False,
        )
        assert np.trace(cj) > -1


def test_predicted_neq_given_filter():
    pred_probs = np.array(
        [
            [0.9, 0.1, 0],
            [0.6, 0.2, 0.2],
            [0.3, 0.3, 4],
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


def test_num_label_issues():
    n = count.num_label_issues(
        labels=data["labels"], pred_probs=data["pred_probs"], confident_joint=data["cj"]
    )
    n2 = filter.find_label_issues(
        labels=data["labels"], pred_probs=data["pred_probs"], confident_joint=data["cj"]
    )
    # TODO: eventually these should be the same value
    assert n == 37
    assert sum(n2) == 35


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
