#!/usr/bin/env python
# coding: utf-8

# Python 2 and 3 compatibility
from __future__ import (
    print_function, absolute_import, division,
    unicode_literals, with_statement,
)

# Make sure python version is compatible with pyTorch
from cleanlab.util import VersionWarning

python_version = VersionWarning(
    warning_str="pyTorch supports Python version 2.7, 3.5, 3.6, 3.7.",
    list_of_compatible_versions=[3.5, 3.6, 3.7],
)

if python_version.is_compatible():
    from cleanlab.models.mnist_pytorch import (
        CNN, MNIST_TEST_SIZE,
        MNIST_TRAIN_SIZE,
    )
    import cleanlab
    from os.path import expanduser
    import numpy as np
    from sklearn.metrics import accuracy_score
    from torchvision import datasets
    import pytest

    # Get home directory to store MNIST dataset
    home_dir = expanduser("~")
    data_dir = home_dir + "/data/"

    X_train = np.arange(MNIST_TRAIN_SIZE)
    X_test = np.arange(MNIST_TEST_SIZE)
    # X_train = X_train[X_train % 10 == 0]
    y_train = datasets.MNIST(data_dir, train=True,
                             download=True).train_labels.numpy()
    y_test = datasets.MNIST(data_dir, train=False,
                            download=True).test_labels.numpy()
    py_train = cleanlab.util.value_counts(y_train) / float(len(y_train))
    X_test_data = datasets.MNIST(data_dir, train=False,
                                 download=True).test_data.numpy()


def test_loaders(
        seed=0,
        n=300,  # Number of training examples to use
        pretrain_epochs=2,  # Increase to at least 10 for good results
):
    """This is going to OVERFIT - train and test on the SAME SET.
    The goal of this test is just to make sure the data loads correctly.
    And all the main functions work."""

    from cleanlab.latent_estimation import (
        estimate_confident_joint_and_cv_pred_proba, estimate_latent)

    if python_version.is_compatible():
        np.random.seed(seed)
        cnn = CNN(epochs=3, log_interval=1000, loader='train', seed=0)
        idx = np.random.choice(X_train, n,
                               replace=False)  # Grab n random examples.
        test_idx = np.random.choice(X_test, n,
                                    replace=False)  # Grab n random examples.

        prune_method = 'prune_by_noise_rate'

        # Pre-train
        cnn = CNN(epochs=1, log_interval=None, seed=seed)  # pre-train
        score = 0
        for loader in ['train', 'test', None]:
            print('loader:', loader)
            prev_score = score
            X = X_test[test_idx] if loader == 'test' else X_train[idx]
            y = y_test[test_idx] if loader == 'test' else y_train[idx]
            # Setting this overides all future functions.
            cnn.loader = loader
            # pre-train (overfit, not out-of-sample) to entire dataset.
            cnn.fit(X, None, loader='train')

            # Out-of-sample cross-validated holdout predicted probabilities
            np.random.seed(seed)
            # Single epoch for cross-validation (already pre-trained)
            cnn.epochs = 1
            cj, psx = estimate_confident_joint_and_cv_pred_proba(
                X, y, cnn, cv_n_folds=2)
            est_py, est_nm, est_inv = estimate_latent(cj, y)
            # algorithmic identification of label errors
            noise_idx = cleanlab.pruning.get_noise_indices(
                y, psx, est_inv, prune_method=prune_method)

            # Get prediction on loader set (in this case same as train set)
            pred = cnn.predict(X, loader='train')
            score = accuracy_score(y, pred)
            print(score)
            assert (score > prev_score)  # Scores should increase

    assert True


def test_throw_exception():
    if python_version.is_compatible():
        cnn = CNN(epochs=1, log_interval=1000, seed=0)
        try:
            cnn.fit(train_idx=[0, 1], train_labels=[1])
        except Exception as e:
            assert ('same length' in str(e))
            with pytest.raises(ValueError) as e:
                cnn.fit(train_idx=[0, 1], train_labels=[1])
    assert True


def test_n_train_examples(n=500):
    if python_version.is_compatible():
        cnn = CNN(epochs=3, log_interval=1000, loader='train', seed=0)
        idx = np.random.choice(X_train, n,
                               replace=False)  # Grab n random examples.
        cnn.fit(train_idx=X_train[idx], train_labels=y_train[idx],
                loader='train')
        cnn.loader = 'test'
        pred = cnn.predict(X_test[:n])
        print(accuracy_score(y_test[:n], pred))
        assert (accuracy_score(y_test[:n], pred) > 0.1)

        # Check that dataset defaults to test set when an invalid name is given.
        cnn.loader = 'INVALID'
        pred = cnn.predict(X_test[:n])
        assert (len(pred) == MNIST_TEST_SIZE)

        # Check that pred_proba runs on all examples when None is passed in
        cnn.loader = 'test'
        proba = cnn.predict_proba(idx=None, loader='test')
        assert proba is not None
        assert (len(pred) == MNIST_TEST_SIZE)

    assert True
