# coding: utf-8

from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
from cleanlab import latent_estimation
from cleanlab.noise_generation import generate_noise_matrix_from_trace, generate_noisy_labels
from cleanlab.latent_estimation import compute_inv_noise_matrix
from cleanlab import baseline_methods
from cleanlab.util import confusion_matrix
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
    data = []
    labels = []
    test_data = []
    test_labels = []

    for idx in range(m):
        data.append(np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx]))
        test_data.append(np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx]))
        labels.append(np.array([idx for i in range(sizes[idx])]))
        test_labels.append(np.array([idx for i in range(sizes[idx])]))
    X_train = np.vstack(data)
    y_train = np.hstack(labels)
    X_test = np.vstack(test_data)
    y_test = np.hstack(test_labels)

    if sparse:
        X_train = scipy.sparse.csr_matrix(X_train)
        X_test = scipy.sparse.csr_matrix(X_test)

    # Compute p(y=k)
    py = np.bincount(y_train) / float(len(y_train))

    noise_matrix = generate_noise_matrix_from_trace(
        m,
        trace=avg_trace * m,
        py=py,
        valid_noise_matrix=True,
        seed=seed,
    )

    # Generate our noisy labels using the noise_marix.
    s = generate_noisy_labels(y_train, noise_matrix)
    ps = np.bincount(s) / float(len(s))

    # Compute inverse noise matrix
    inv = compute_inv_noise_matrix(py, noise_matrix, ps)

    # Estimate psx
    latent = latent_estimation.estimate_py_noise_matrices_and_cv_pred_proba(
        X=X_train,
        s=s,
        cv_n_folds=3,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "s": s,
        "ps": ps,
        "py": py,
        "noise_matrix": noise_matrix,
        "inverse_noise_matrix": inv,
        "est_py": latent[0],
        "est_nm": latent[1],
        "est_inv": latent[2],
        "cj": latent[3],
        "psx": latent[4],
        "m": m,
        "n": n,
    }


# Global to be used by all test methods.
# Only compute this once for speed.
seed = 1
data = make_data(sparse=False, seed=seed)

# Create some simple data to test
psx_ = np.array([
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
])
s_ = np.array([0,0,1,1,1,1,1,1,1,2])


def test_confident_learning_baseline():
    cj, indices = latent_estimation.compute_confident_joint(
        s=data["s"],
        psx=data["psx"],
        calibrate=False,
        return_indices_of_off_diagonals=True,
    )
    # Check that the number of 'label errors' found in off diagonals
    # matches the off diagonals of the uncalibrated confident joint
    assert(len(indices) == (np.sum(cj) - np.trace(cj)))


def test_baseline_argmax():
    psx = np.array([
        [0.9, 0.1, 0],
        [0.6, 0.2, 0.2],
        [0.3, 0.3, 4],
        [0.1, 0.1, 0.8],
        [0.4, 0.5, 0.1],
    ])
    s = np.array([0,0,1,1,2])
    label_errors = baseline_methods.baseline_argmax(psx, s)
    assert(all(label_errors == [False, False, True, True, True]))
    
    label_errors = baseline_methods.baseline_argmax(psx_, s_)
    assert(all(label_errors == np.array([False, False, True, False, 
        False, False, False, False, False, False])))


@pytest.mark.parametrize("prune_method", ['prune_by_noise_rate',
                                          'prune_by_class', 'both'])
def test_baseline_argmax_confusion_matrix(prune_method):
    confident_joint = confusion_matrix(true=np.argmax(psx_, axis=1), pred=s_).T
    label_errors = baseline_methods.baseline_argmax_confusion_matrix(psx_, s_)
    assert(all(label_errors == np.array([False, False, True, False, 
        False, False, False, False, False, False])))


@pytest.mark.parametrize("prune_method", ['prune_by_noise_rate',
                                          'prune_by_class', 'both'])
def test_baseline_argmax_calibrated_confusion_matrix(prune_method):
    confident_joint = confusion_matrix(true=np.argmax(psx_, axis=1), pred=s_).T
    label_errors = baseline_methods.baseline_argmax_calibrated_confusion_matrix(
        psx_, s_)
    assert(all(label_errors == np.array([False, False, True, False, 
        False, False, False, False, False, False])))