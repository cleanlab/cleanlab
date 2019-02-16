#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Python 2 and 3 compatibility
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement


# In[ ]:


import numpy as np
from cleanlab import classification
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.noise_generation import generate_noise_matrix_from_trace
from cleanlab.noise_generation import generate_noisy_labels
from cleanlab.latent_algebra import compute_inv_noise_matrix
from sklearn.linear_model import LogisticRegression
from numpy.random import multivariate_normal

import pytest


# In[ ]:


seed = 1


# In[ ]:


def make_data(
    means = [ [3, 2], [7, 7], [0, 8] ],
    covs = [ [[5, -1.5],[-1.5, 1]] , [[1, 0.5],[0.5, 4]], [[5, 1],[1, 5]] ],
    sizes = [ 800, 400, 400 ],
    avg_trace = 0.8,
    seed = 0, # set to None for non-reproducible randomness
):
    
    np.random.seed(seed = seed)

    K = len(means) # number of classes
    data = []
    labels = []
    test_data = []
    test_labels = []

    for idx in range(K):
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
      seed = seed,
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


# In[ ]:


data = make_data(seed = seed)


# In[ ]:


def test_rp():
    rp = LearningWithNoisyLabels(clf = LogisticRegression(multi_class='auto', solver='lbfgs', random_state=seed))
    rp.fit(data["X_train"], data["s"])
    score = rp.score(data["X_test"], data["y_test"])
    print(score)
    # Check that this runs without error.
    assert(True)


# In[ ]:


def test_raise_error_no_clf_fit():
    class struct(object):
        def predict(self):
            pass
        def predict_proba(self):
            pass
    try:
        LearningWithNoisyLabels(clf = struct())
    except Exception as e:
        assert('fit' in str(e))
        with pytest.raises(ValueError) as e:
            LearningWithNoisyLabels(clf = struct())


# In[ ]:


def test_raise_error_no_clf_predict_proba():
    class struct(object):
        def fit(self):
            pass
        def predict(self):
            pass
    try:
        LearningWithNoisyLabels(clf = struct())
    except Exception as e:
        assert('predict_proba' in str(e))
        with pytest.raises(ValueError) as e:
            LearningWithNoisyLabels(clf = struct())


# In[ ]:


def test_raise_error_no_clf_predict():
    class struct(object):
        def fit(self):
            pass
        def predict_proba(self):
            pass
    try:
        LearningWithNoisyLabels(clf = struct())
    except Exception as e:
        assert('predict' in str(e))
        with pytest.raises(ValueError) as e:
            LearningWithNoisyLabels(clf = struct())


# In[ ]:


def test_seed():
    lnl = LearningWithNoisyLabels(seed = 0)
    assert(lnl.seed is not None)


# In[ ]:


def test_default_clf():
    lnl = LearningWithNoisyLabels()
    return lnl.clf is not None and         hasattr(lnl.clf, 'fit') and         hasattr(lnl.clf, 'predict') and         hasattr(lnl.clf, 'predict_proba')


# In[ ]:


def test_clf_fit_nm():
    lnl = LearningWithNoisyLabels()
    # Example of a bad noise matrix (impossible to learn from)
    nm = np.array([[0, 1],[1, 0]])
    try:
        lnl.fit(X = np.arange(3), s = np.array([0,0,1]), noise_matrix = nm)
    except Exception as e:
        assert('Trace(noise_matrix)' in str(e))
        with pytest.raises(ValueError) as e:
            lnl.fit(X = np.arange(3), s = np.array([0,0,1]), noise_matrix = nm)
    


# In[ ]:


def test_clf_fit_inm():
    lnl = LearningWithNoisyLabels()
    # Example of a bad noise matrix (impossible to learn from)
    inm = np.array([[.1, .9],[.9, .1]])
    try:
        lnl.fit(X = np.arange(3), s = np.array([0,0,1]), inverse_noise_matrix = inm)
    except Exception as e:
        assert('Trace(inverse_noise_matrix)' in str(e))
        with pytest.raises(ValueError) as e:
            lnl.fit(X = np.arange(3), s = np.array([0,0,1]), inverse_noise_matrix = inm)


# In[ ]:


def test_fit_with_nm(
    seed = 0, 
    used_by_another_test = False,
):
    lnl = LearningWithNoisyLabels(
        seed = seed,
    )
    nm = data['noise_matrix']
    # Learn with noisy labels with noise matrix given
    lnl.fit(data['X_train'], data['s'], noise_matrix = nm)
    score_nm = lnl.score(data['X_test'], data['y_test'])
    # Learn with noisy labels and estimate the noise matrix.
    lnl2 = LearningWithNoisyLabels(
        seed = seed,
    )
    lnl2.fit(data['X_train'], data['s'],)
    score = lnl2.score(data['X_test'], data['y_test'])
    if used_by_another_test:
        return score, score_nm
    else:
        assert(score < score_nm + 1e-4)


# In[ ]:


def test_fit_with_inm(
    seed = 0, 
    used_by_another_test = False,
):
    lnl = LearningWithNoisyLabels(
        seed = seed, 
    )
    inm = compute_inv_noise_matrix(
        data["py"],
        data["noise_matrix"],
        data["ps"],
    )
    # Learn with noisy labels with inverse noise matrix given
    lnl.fit(data['X_train'], data['s'], inverse_noise_matrix = inm)
    score_inm = lnl.score(data['X_test'], data['y_test'])
    # Learn with noisy labels and estimate the inv noise matrix.
    lnl2 = LearningWithNoisyLabels(
        seed = seed,
    )
    lnl2.fit(data['X_train'], data['s'],)
    score = lnl2.score(data['X_test'], data['y_test'])
    if used_by_another_test:
        return score, score_inm
    else:
        assert(score < score_inm + 1e-4)


# In[ ]:


def test_clf_fit_nm_inm():
    lnl = LearningWithNoisyLabels(seed = seed)
    nm = data['noise_matrix']
    inm = compute_inv_noise_matrix(
        data["py"],
        nm,
        data["ps"],
    )
    lnl.fit(
        X = data['X_train'], 
        s = data['s'],
        noise_matrix = nm,
        inverse_noise_matrix = inm,
    )
    score_nm_inm = lnl.score(data['X_test'], data['y_test'])
    
    # Learn with noisy labels and estimate the inv noise matrix.
    lnl2 = LearningWithNoisyLabels(seed = seed)
    lnl2.fit(data['X_train'], data['s'],)
    score = lnl2.score(data['X_test'], data['y_test'])
    assert(score < score_nm_inm + 1e-4)


# In[ ]:


def test_pred_and_pred_proba():
    lnl = LearningWithNoisyLabels()
    lnl.fit(data['X_train'], data['s'])
    n = np.shape(data['y_test'])[0]
    m = len(np.unique(data['y_test']))
    pred = lnl.predict(data['X_test'])
    probs = lnl.predict_proba(data['X_test'])
    # Just check that this functions return what we expect
    assert(np.shape(pred)[0] == n)
    assert(np.shape(probs) == (n, m))


# In[ ]:


def test_score():
    phrase = 'cleanlab is dope'
    class Struct():
        def fit(self):
            pass
        def predict_proba(self):
            pass
        def predict(self):
            pass
        def score(self, X, y):
            return phrase
    lnl = LearningWithNoisyLabels(clf = Struct())
    score = lnl.score(data['X_test'], data['y_test'])
    assert(score == phrase)


# In[ ]:


def test_no_score():
    class Struct():
        def fit(self):
            pass
        def predict_proba(self):
            pass
        def predict(self, X):
            return data['y_test']
    lnl = LearningWithNoisyLabels(clf = Struct())
    score = lnl.score(data['X_test'], data['y_test'])
    assert(abs(score - 1) < 1e-6)


# In[ ]:


def test_no_fit_sample_weight():
    class Struct():
        def fit(self, X, y):
            pass
        def predict_proba(self):
            pass
        def predict(self, X):
            return data['y_test']
    n = np.shape(data['y_test'])[0]
    m = len(np.unique(data['y_test']))
    psx = np.zeros(shape = (n, m))
    lnl = LearningWithNoisyLabels(clf = Struct())
    lnl.fit(data['X_train'], data['y_train'], psx = psx, noise_matrix = data['noise_matrix'])
    # If we make it here, without any error:
    assert(True)


# In[ ]:


def test_fit_psx():
    from cleanlab.latent_estimation import estimate_cv_predicted_probabilities
    lnl = LearningWithNoisyLabels()
    psx = estimate_cv_predicted_probabilities(
        X = data['X_train'],
        labels = data['y_train'],
    )
    lnl.fit(
        X = data['X_train'],
        s = data['y_train'],
        psx = psx
    )
    score_with_psx = lnl.score(data['X_test'], data['y_test'])
    lnl = LearningWithNoisyLabels()
    lnl.fit(
        X = data['X_train'],
        s = data['y_train'],
    )
    score_no_psx = lnl.score(data['X_test'], data['y_test'])
    assert(abs(score_with_psx - score_no_psx) < 1e-6)

