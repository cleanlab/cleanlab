#!/usr/bin/env python
# coding: utf-8

from cleanlab.experimental.mnist_pytorch import (
    CNN,
    SKLEARN_DIGITS_TEST_SIZE,
    SKLEARN_DIGITS_TRAIN_SIZE,
)
import cleanlab
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
import pytest

X_train_idx = np.arange(SKLEARN_DIGITS_TRAIN_SIZE)
X_test_idx = np.arange(SKLEARN_DIGITS_TEST_SIZE)
# Get sklearn digits data labels
_, y_all = load_digits(return_X_y=True)
# PyTorch requires type long targets.
y_train = y_all[:-SKLEARN_DIGITS_TEST_SIZE].astype(np.int32)
true_labels_test = y_all[-SKLEARN_DIGITS_TEST_SIZE:].astype(np.int32)


@pytest.mark.slow
def test_loaders(
    seed=0,
):
    """This is going to OVERFIT - train and test on the SAME SET.
    The goal of this test is just to make sure the data loads correctly.
    And all the main functions work."""

    from cleanlab.count import estimate_confident_joint_and_cv_pred_proba, estimate_latent

    np.random.seed(seed)
    filter_by = "prune_by_noise_rate"
    # Pre-train for only 3 epochs (it maxes out around 8-12 epochs)
    cnn = CNN(epochs=3, log_interval=None, seed=seed, dataset="sklearn-digits")
    score = 0
    for loader in ["train", "test", None]:
        print("loader:", loader)
        prev_score = score
        X = X_test_idx if loader == "test" else X_train_idx
        y = true_labels_test if loader == "test" else y_train
        # Setting this overrides all future functions.
        cnn.loader = loader
        # pre-train (overfit, not out-of-sample) to entire dataset.
        cnn.fit(
            X,
            None,
        )
        # This next block of code checks if cleanlab works with the CNN
        # Out-of-sample cross-validated holdout predicted probabilities
        np.random.seed(seed)
        # Single epoch for cross-validation (already pre-trained)
        cnn.epochs = 1
        cj, pred_probs = estimate_confident_joint_and_cv_pred_proba(X, y, cnn, cv_n_folds=2)
        est_py, est_nm, est_inv = estimate_latent(cj, y)
        # algorithmic identification of label issues
        err_idx = cleanlab.filter.find_label_issues(
            y, pred_probs, confident_joint=cj, filter_by=filter_by
        )
        assert err_idx is not None

        # Get prediction on loader set.
        pred = cnn.predict(X)
        score = accuracy_score(y, pred)
        print("Acc Before: {:.2f}, After: {:.2f}".format(prev_score, score))
        assert score > prev_score  # Scores should increase


def test_throw_exception():
    cnn = CNN(epochs=1, log_interval=1000, seed=0)
    try:
        cnn.fit(train_idx=[0, 1], train_labels=[1])
    except Exception as e:
        assert "same length" in str(e)
        with pytest.raises(ValueError) as e:
            cnn.fit(train_idx=[0, 1], train_labels=[1])


def test_n_train_examples():
    cnn = CNN(
        epochs=4,
        log_interval=1000,
        loader="train",
        seed=1,
        dataset="sklearn-digits",
    )
    cnn.fit(
        train_idx=X_train_idx,
        train_labels=y_train,
        loader="train",
    )
    cnn.loader = "test"
    pred = cnn.predict(X_test_idx)
    print(accuracy_score(true_labels_test, pred))
    assert accuracy_score(true_labels_test, pred) > 0.1

    # Check that exception is raised when invalid name is given.
    cnn.loader = "INVALID"
    with pytest.raises(ValueError) as e:
        pred = cnn.predict(X_test_idx)

    # Check that pred_proba runs on all examples when None is passed in
    cnn.loader = "test"
    proba = cnn.predict_proba(idx=None, loader="test")
    assert proba is not None
    assert len(pred) == SKLEARN_DIGITS_TEST_SIZE
