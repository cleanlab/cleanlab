
# coding: utf-8

# In[1]:


from __future__ import print_function, absolute_import, division, unicode_literals, with_statement

import numpy as np
from cleanlab.classification import RankPruning
from cleanlab.noise_generation import generate_noise_matrix_from_trace
from cleanlab.noise_generation import generate_noisy_labels
from sklearn.linear_model import LogisticRegression
from numpy.random import multivariate_normal

import pytest


# In[2]:


def make_data(
    means = [ [3, 2], [7, 7], [0, 8] ],
    covs = [ [[5, -1.5],[-1.5, 1]] , [[1, 0.5],[0.5, 4]], [[5, 1],[1, 5]] ],
    sizes = [ 800, 400, 400 ],
    avg_trace = 0.5,
    seed = 0, # set to None for non-reproducible randomness
):
    
    np.random.seed(seed = seed)

    K = len(means) # number of classes
    data = []
    labels = []
    test_data = []
    test_labels = []

    for idx in range(len(means)):
        data.append(np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx]))
        test_data.append(np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx]))
        labels.append(np.array([idx for i in range(sizes[idx])]))
        test_labels.append(np.array([idx for i in range(sizes[idx])]))
    X_train = np.vstack(data)
    y_train = np.hstack(labels)
    X_test = np.vstack(test_data)
    y_test = np.hstack(test_labels) 

    # Compute p(y=k)
    py = np.bincount(y_train) / float(len(y_train))

    noise_matrix = generate_noise_matrix_from_trace(
      K, 
      trace = avg_trace * K,
      py = py,
      valid_noise_matrix = True,
    )

    # Generate our noisy labels using the noise_marix.
    s = generate_noisy_labels(y_train, noise_matrix)
    ps = np.bincount(s) / float(len(s))
    
    return {
        "X_train" : X_train,
        "y_train" : y_train,
        "X_test" : X_test,
        "y_test" : y_test,
        "s" : s,
        "ps" : ps,
        "py" : py,
        "noise_matrix" : noise_matrix,
    }


# In[3]:


def test_rp():
    seed = 46
    data = make_data(seed = seed)
    rp = RankPruning(clf = LogisticRegression(multi_class='auto', solver='lbfgs', random_state=seed))
    rp.fit(data["X_train"], data["s"])
    score = rp.score(data["X_test"], data["y_test"])
    assert(abs(score - 0.88) < 0.01)

