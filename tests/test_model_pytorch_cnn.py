
# coding: utf-8

# In[ ]:


# Python 2 and 3 compatibility
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement


# In[ ]:


# Make sure python version is compatible with pyTorch
from cleanlab.util import VersionWarning
python_version = VersionWarning(
    warning_str = "pyTorch supports Python version 2.7, 3.5, 3.6, 3.7.",
    list_of_compatible_versions = [2.7, 3.5, 3.6, 3.7],
)


# In[ ]:


if python_version.is_compatible():
    from cleanlab.models.mnist_pytorch import CNN, MNIST_TEST_SIZE, MNIST_TRAIN_SIZE
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
    y_train = datasets.MNIST(data_dir, train=True, download = True).train_labels.numpy()
    y_test = datasets.MNIST(data_dir, train=False, download = True).test_labels.numpy()
    py_train = cleanlab.util.value_counts(y_train) / float(len(y_train))
    X_test_data = datasets.MNIST(data_dir, train=False, download = True).test_data.numpy()


# In[ ]:


def test_loaders(
    seed = 0,
    pretrain_epochs = 2, # Increase to at least 10 for good results
):
    if python_version.is_compatible():
        np.random.seed(seed)

        prune_method = 'prune_by_noise_rate'
        
        # Pre-train
        cnn = CNN(epochs=2, log_interval=1000, seed = seed) #pre-train
        for loader in ['train', 'test', None]:
            cnn.loader = loader
            cnn.fit(X_test, y_test, loader=loader) # pre-train (overfit, not out-of-sample) to entire dataset.

            # Out-of-sample cross-validated holdout predicted probabilities
            np.random.seed(seed)
            # Single epoch for cross-validation (already pre-trained)
            cnn.epochs = 1 
            cj, psx = cleanlab.latent_estimation.estimate_confident_joint_and_cv_pred_proba(X_test, y_test, cnn, cv_n_folds=2)
            est_py, est_nm, est_inv = cleanlab.latent_estimation.estimate_latent(cj, y_test)
            # algorithmic identification of label errors
            noise_idx = cleanlab.pruning.get_noise_indices(y_test, psx, est_inv, prune_method=prune_method) 

            # Get prediction on test set
            pred = cnn.predict(loader = loader)
            if loader == 'test':
                score = accuracy_score(y_test, pred)
            else:
                score = accuracy_score(y_train, pred)
            print(score)
        
    assert(True)


# In[ ]:


def test_throw_exception():
    if python_version.is_compatible():
        cnn = CNN(epochs=1, log_interval=1000, seed = 0)
        try:
            cnn.fit(train_idx = [0,1], train_labels = [1])
        except Exception as e:
            assert('same length' in str(e))
            with pytest.raises(ValueError) as e:
                cnn.fit(train_idx = [0,1], train_labels = [1])
    assert(True)


# In[ ]:


def test_n_train_examples(n = 3000):
    if python_version.is_compatible():
        cnn = CNN(epochs=3, log_interval=1000, loader = 'train', seed = 0)
        idx = np.random.choice(X_train, n, replace = False) # Grab n random examples.
        cnn.fit(train_idx = X_train[idx], train_labels = y_train[idx], loader = 'train')
        cnn.loader = 'test'
        pred = cnn.predict(X_test[:n])
        assert(accuracy_score(y_test[:n], pred) > 0.5)

        # Check that dataset defaults to test set when an invalid name is given.
        cnn.loader = 'INVALID'
        pred = cnn.predict(X_test[:n])
        assert(len(pred) == MNIST_TEST_SIZE)
    assert(True)

