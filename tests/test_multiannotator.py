import numpy as np
import pytest
from copy import deepcopy
from cleanlab import dataset
from cleanlab.benchmarking.noise_generation import generate_noise_matrix_from_trace
from cleanlab.benchmarking.noise_generation import generate_noisy_labels
from cleanlab import count
from cleanlab.multiannotator import get_label_quality_multiannotator
import pandas as pd


def make_data(
    means=[[3, 2], [7, 7], [0, 8]],
    covs=[[[5, -1.5], [-1.5, 1]], [[1, 0.5], [0.5, 4]], [[5, 1], [1, 5]]],
    sizes=[80, 40, 40],
    avg_trace=0.8,
    num_annotators=50,
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

    # Generate our noisy labels using the noise_matrix for specified number of annotators.
    s = pd.DataFrame(
        np.vstack(
            [generate_noisy_labels(true_labels_train, noise_matrix) for _ in range(num_annotators)]
        ).transpose()
    )

    # column of labels without NaNs to test _get_worst_class
    complete_labels = deepcopy(s)

    # Each annotator only labels approximately 20% of the dataset
    # (unlabeled points represented with NaN)
    s = s.apply(lambda x: x.mask(np.random.random(n) < 0.8))
    s.dropna(axis=1, how="all", inplace=True)

    # Estimate pred_probs
    latent = count.estimate_py_noise_matrices_and_cv_pred_proba(
        X=X_train,
        labels=true_labels_train,
        cv_n_folds=3,
    )

    row_NA_check = pd.notna(s).any(axis=1)

    return {
        "X_train": X_train[row_NA_check],
        "true_labels_train": true_labels_train[row_NA_check],
        "X_test": X_test[row_NA_check],
        "true_labels_test": true_labels_test[row_NA_check],
        "labels": s[row_NA_check].reset_index(drop=True),
        "complete_labels": complete_labels,
        "pred_probs": latent[4][row_NA_check],
        "noise_matrix": noise_matrix,
    }


# Global to be used by all test methods. Only compute this once for speed.
data = make_data()


# TODO: update this test to work with NaN values - randomly deleting data will make assertion invalid
def test_get_worst_class():
    labels = data["complete_labels"][0]  # only testing on first column
    pred_probs = data["pred_probs"]

    # Assert that the worst class index should be the class with the highest noise
    assert dataset._get_worst_class(labels, pred_probs) == data["noise_matrix"].diagonal().argmax()


def test_label_quality_scores_multiannotator():
    labels = data["labels"]
    pred_probs = data["pred_probs"]

    lqs_multiannotator = get_label_quality_multiannotator(labels, pred_probs)
    assert isinstance(lqs_multiannotator, pd.DataFrame)

    # test verbose=False
    lqs_multiannotator = get_label_quality_multiannotator(labels, pred_probs, verbose=False)
    assert isinstance(lqs_multiannotator, pd.DataFrame)

    # test passing a list into consensus_method
    # TODO: change list items after adding more options
    lqs_multiannotator = get_label_quality_multiannotator(
        labels, pred_probs, consensus_method=["majority", "majority"]
    )

    # test different quality_methods
    lqs_multiannotator = get_label_quality_multiannotator(
        labels, pred_probs, quality_method="agreement"
    )

    # test returning annotator stats
    lqs_annotatorstats = get_label_quality_multiannotator(
        labels, pred_probs, return_annotator_stats=True
    )
    assert isinstance(lqs_annotatorstats, tuple)
    assert isinstance(lqs_annotatorstats[0], pd.DataFrame)
    assert isinstance(lqs_annotatorstats[1], pd.DataFrame)

    # test error catching when labels_multiannotator has NaN columns
    labels_NA = deepcopy(labels)
    labels_NA[0] = pd.NA

    try:
        lqs_annotatorstats = get_label_quality_multiannotator(
            labels_NA, pred_probs, return_annotator_stats=True
        )
    except ValueError as e:
        assert "cannot have columns with all NaN" in str(e)
