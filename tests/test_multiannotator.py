import numpy as np
import pytest
from cleanlab import dataset
from cleanlab.internal.label_quality_utils import _subtract_confident_thresholds
from cleanlab.benchmarking.noise_generation import generate_noise_matrix_from_trace
from cleanlab.benchmarking.noise_generation import generate_noisy_labels
from cleanlab import count
from cleanlab.multiannotator import get_label_quality_scores_multiannotator
import pandas as pd


def make_data(
    means=[[3, 2], [7, 7], [0, 8]],
    covs=[[[5, -1.5], [-1.5, 1]], [[1, 0.5], [0.5, 4]], [[5, 1], [1, 5]]],
    sizes=[80, 40, 40],
    avg_trace=0.8,
    num_annotators=5,
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

    # Each annotator only labels approximately 75% of the dataset
    # (unlabeled points represented with NaN)
    s = s.apply(lambda x: x.mask(np.random.random(n) < 0.25)).astype("Int32")

    # s = generate_noisy_labels(true_labels_train, noise_matrix) # one annotator
    # ps = np.bincount(s) / float(len(s))

    # Compute inverse noise matrix
    # inv = count.compute_inv_noise_matrix(py, noise_matrix, ps=ps)

    # Estimate pred_probs
    latent = count.estimate_py_noise_matrices_and_cv_pred_proba(
        X=X_train,
        labels=true_labels_train,
        cv_n_folds=3,
    )

    # label_errors_mask = s != true_labels_train

    return {
        "X_train": X_train,
        "true_labels_train": true_labels_train,
        "X_test": X_test,
        "true_labels_test": true_labels_test,
        "labels": s,
        # "label_errors_mask": label_errors_mask,
        # "ps": ps,
        # "py": py,
        "noise_matrix": noise_matrix,
        # "inverse_noise_matrix": inv,
        # "est_py": latent[0],
        # "est_nm": latent[1],
        # "est_inv": latent[2],
        # "cj": latent[3],
        "pred_probs": latent[4],
        # "m": m,
        # "n": n,
    }


# Global to be used by all test methods. Only compute this once for speed.
data = make_data()


# TODO: fix this test to work with NaN values - also unsure if this test should sit in this file
# logic of current test does not work - randomly deleting data will break it (put in another file?)
# def test_get_worst_class():

#     labels = data["labels"][0]  # only test on first column
#     pred_probs = data["pred_probs"]

#     # Assert that the worst class index should be the class with the highest noise
#     assert (
#         dataset._get_worst_class(
#             labels[pd.notna(labels)].astype("int32"), pred_probs[pd.notna(labels)]
#         )
#         == data["noise_matrix"].diagonal().argmax()
#     )


def test_label_quality_scores_multiannotator():

    labels = data["labels"]
    pred_probs = data["pred_probs"]

    lqs_multiannotator = get_label_quality_scores_multiannotator(labels, pred_probs)
    assert isinstance(lqs_multiannotator, pd.DataFrame)

    # test returning annotator stats
    lqs_annotatorstats = get_label_quality_scores_multiannotator(
        labels, pred_probs, return_annotator_stats=True
    )
    assert isinstance(lqs_annotatorstats, tuple)
