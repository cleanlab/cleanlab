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
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.noise_generation import generate_noise_matrix_from_trace
from cleanlab.noise_generation import generate_noisy_labels
from cleanlab.internal.latent_algebra import compute_inv_noise_matrix
from sklearn.linear_model import LogisticRegression
from numpy.random import multivariate_normal
import scipy
import warnings
import pytest


SEED = 1


def make_data(
    sparse,
    means=[[3, 2], [7, 7], [0, 8]],
    covs=[[[5, -1.5], [-1.5, 1]], [[1, 0.5], [0.5, 4]], [[5, 1], [1, 5]]],
    sizes=[100, 50, 50],
    avg_trace=0.8,
    seed=SEED,  # set to None for non-reproducible randomness
):
    np.random.seed(seed=SEED)

    K = len(means)  # number of classes
    data = []
    labels = []
    test_data = []
    test_labels = []

    for idx in range(K):
        data.append(np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx]))
        test_data.append(
            np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx])
        )
        labels.append(np.array([idx for i in range(sizes[idx])]))
        test_labels.append(np.array([idx for i in range(sizes[idx])]))
    X_train = np.vstack(data)
    true_labels_train = np.hstack(labels)
    X_test = np.vstack(test_data)
    true_labels_test = np.hstack(test_labels)

    if sparse:
        X_train = scipy.sparse.csr_matrix(X_train)
        X_test = scipy.sparse.csr_matrix(X_test)

    # Compute p(true_label=k)
    py = np.bincount(true_labels_train) / float(len(true_labels_train))

    noise_matrix = generate_noise_matrix_from_trace(
        K,
        trace=avg_trace * K,
        py=py,
        valid_noise_matrix=True,
        seed=SEED,
    )

    # Generate our noisy labels using the noise_marix.
    s = generate_noisy_labels(true_labels_train, noise_matrix)
    ps = np.bincount(s) / float(len(s))

    return {
        "X_train": X_train,
        "true_labels_train": true_labels_train,
        "X_test": X_test,
        "true_labels_test": true_labels_test,
        "labels": s,
        "ps": ps,
        "py": py,
        "noise_matrix": noise_matrix,
    }


DATA = make_data(sparse=False, seed=SEED)
SPARSE_DATA = make_data(sparse=False, seed=SEED)


@pytest.mark.parametrize("sparse", [True, False])
def test_rp(sparse):
    data = SPARSE_DATA if sparse else DATA
    rp = LearningWithNoisyLabels(
        clf=LogisticRegression(multi_class="auto", solver="lbfgs", random_state=SEED)
    )
    rp.fit(data["X_train"], data["labels"])
    score = rp.score(data["X_test"], data["true_labels_test"])
    print(score)
    # Check that this runs without error.
    assert True


def test_raise_error_no_clf_fit():
    class struct(object):
        def predict(self):
            pass

        def predict_proba(self):
            pass

    try:
        LearningWithNoisyLabels(clf=struct())
    except Exception as e:
        assert "fit" in str(e)
        with pytest.raises(ValueError) as e:
            LearningWithNoisyLabels(clf=struct())


def test_raise_error_no_clf_predict_proba():
    class struct(object):
        def fit(self):
            pass

        def predict(self):
            pass

    try:
        LearningWithNoisyLabels(clf=struct())
    except Exception as e:
        assert "predict_proba" in str(e)
        with pytest.raises(ValueError) as e:
            LearningWithNoisyLabels(clf=struct())


def test_raise_error_no_clf_predict():
    class struct(object):
        def fit(self):
            pass

        def predict_proba(self):
            pass

    try:
        LearningWithNoisyLabels(clf=struct())
    except Exception as e:
        assert "predict" in str(e)
        with pytest.raises(ValueError) as e:
            LearningWithNoisyLabels(clf=struct())


def test_seed():
    lnl = LearningWithNoisyLabels(seed=SEED)
    assert lnl.seed is not None


def test_default_clf():
    lnl = LearningWithNoisyLabels()
    check1 = lnl.clf is not None and hasattr(lnl.clf, "fit")
    check2 = hasattr(lnl.clf, "predict") and hasattr(lnl.clf, "predict_proba")
    assert check1 and check2


def test_clf_fit_nm():
    lnl = LearningWithNoisyLabels()
    # Example of a bad noise matrix (impossible to learn from)
    nm = np.array([[0, 1], [1, 0]])
    try:
        lnl.fit(X=np.arange(3), labels=np.array([0, 0, 1]), noise_matrix=nm)
    except Exception as e:
        assert "Trace(noise_matrix)" in str(e)
        with pytest.raises(ValueError) as e:
            lnl.fit(X=np.arange(3), labels=np.array([0, 0, 1]), noise_matrix=nm)


def test_clf_fit_inm():
    lnl = LearningWithNoisyLabels()
    # Example of a bad noise matrix (impossible to learn from)
    inm = np.array([[0.1, 0.9], [0.9, 0.1]])
    try:
        lnl.fit(X=np.arange(3), labels=np.array([0, 0, 1]), inverse_noise_matrix=inm)
    except Exception as e:
        assert "Trace(inverse_noise_matrix)" in str(e)
        with pytest.raises(ValueError) as e:
            lnl.fit(X=np.arange(3), labels=np.array([0, 0, 1]), inverse_noise_matrix=inm)


@pytest.mark.parametrize("sparse", [True, False])
def test_fit_with_nm(
    sparse,
    seed=SEED,
    used_by_another_test=False,
):
    data = SPARSE_DATA if sparse else DATA
    lnl = LearningWithNoisyLabels(
        seed=seed,
    )
    nm = data["noise_matrix"]
    # Learn with noisy labels with noise matrix given
    lnl.fit(data["X_train"], data["labels"], noise_matrix=nm)
    score_nm = lnl.score(data["X_test"], data["true_labels_test"])
    # Learn with noisy labels and estimate the noise matrix.
    lnl2 = LearningWithNoisyLabels(
        seed=seed,
    )
    lnl2.fit(
        data["X_train"],
        data["labels"],
    )
    score = lnl2.score(data["X_test"], data["true_labels_test"])
    if used_by_another_test:
        return score, score_nm
    else:
        assert score < score_nm + 1e-4


@pytest.mark.parametrize("sparse", [True, False])
def test_fit_with_inm(
    sparse,
    seed=SEED,
    used_by_another_test=False,
):
    data = SPARSE_DATA if sparse else DATA
    lnl = LearningWithNoisyLabels(
        seed=seed,
    )
    inm = compute_inv_noise_matrix(
        py=data["py"],
        noise_matrix=data["noise_matrix"],
        ps=data["ps"],
    )
    # Learn with noisy labels with inverse noise matrix given
    lnl.fit(data["X_train"], data["labels"], inverse_noise_matrix=inm)
    score_inm = lnl.score(data["X_test"], data["true_labels_test"])
    # Learn with noisy labels and estimate the inv noise matrix.
    lnl2 = LearningWithNoisyLabels(
        seed=seed,
    )
    lnl2.fit(
        data["X_train"],
        data["labels"],
    )
    score = lnl2.score(data["X_test"], data["true_labels_test"])
    if used_by_another_test:
        return score, score_inm
    else:
        assert score < score_inm + 1e-4


@pytest.mark.parametrize("sparse", [True, False])
def test_clf_fit_nm_inm(sparse):
    data = SPARSE_DATA if sparse else DATA
    lnl = LearningWithNoisyLabels(seed=SEED)
    nm = data["noise_matrix"]
    inm = compute_inv_noise_matrix(
        py=data["py"],
        noise_matrix=nm,
        ps=data["ps"],
    )
    lnl.fit(
        X=data["X_train"],
        labels=data["labels"],
        noise_matrix=nm,
        inverse_noise_matrix=inm,
    )
    score_nm_inm = lnl.score(data["X_test"], data["true_labels_test"])

    # Learn with noisy labels and estimate the inv noise matrix.
    lnl2 = LearningWithNoisyLabels(seed=SEED)
    lnl2.fit(
        data["X_train"],
        data["labels"],
    )
    score = lnl2.score(data["X_test"], data["true_labels_test"])
    assert score < score_nm_inm + 1e-4


@pytest.mark.parametrize("sparse", [True, False])
def test_pred_and_pred_proba(sparse):
    data = SPARSE_DATA if sparse else DATA
    lnl = LearningWithNoisyLabels()
    lnl.fit(data["X_train"], data["labels"])
    n = np.shape(data["true_labels_test"])[0]
    m = len(np.unique(data["true_labels_test"]))
    pred = lnl.predict(data["X_test"])
    probs = lnl.predict_proba(data["X_test"])
    # Just check that this functions return what we expect
    assert np.shape(pred)[0] == n
    assert np.shape(probs) == (n, m)


@pytest.mark.parametrize("sparse", [True, False])
def test_score(sparse):
    data = SPARSE_DATA if sparse else DATA
    phrase = "cleanlab is dope"

    class Struct:
        def fit(self):
            pass

        def predict_proba(self):
            pass

        def predict(self):
            pass

        def score(self, X, y):
            return phrase

    lnl = LearningWithNoisyLabels(clf=Struct())
    score = lnl.score(data["X_test"], data["true_labels_test"])
    assert score == phrase


@pytest.mark.parametrize("sparse", [True, False])
def test_no_score(sparse):
    data = SPARSE_DATA if sparse else DATA

    class Struct:
        def fit(self):
            pass

        def predict_proba(self):
            pass

        def predict(self, X):
            return data["true_labels_test"]

    lnl = LearningWithNoisyLabels(clf=Struct())
    score = lnl.score(data["X_test"], data["true_labels_test"])
    assert abs(score - 1) < 1e-6


@pytest.mark.parametrize("sparse", [True, False])
def test_no_fit_sample_weight(sparse):
    data = SPARSE_DATA if sparse else DATA

    class Struct:
        def fit(self, X, y):
            pass

        def predict_proba(self):
            pass

        def predict(self, X):
            return data["true_labels_test"]

    n = np.shape(data["true_labels_test"])[0]
    m = len(np.unique(data["true_labels_test"]))
    pred_probs = np.zeros(shape=(n, m))
    lnl = LearningWithNoisyLabels(clf=Struct())
    lnl.fit(
        data["X_train"],
        data["true_labels_train"],
        pred_probs=pred_probs,
        noise_matrix=data["noise_matrix"],
    )
    # If we make it here, without any error:
    assert True


@pytest.mark.parametrize("sparse", [True, False])
def test_fit_pred_probs(sparse):
    data = SPARSE_DATA if sparse else DATA
    from cleanlab.count import estimate_cv_predicted_probabilities

    lnl = LearningWithNoisyLabels()
    pred_probs = estimate_cv_predicted_probabilities(
        X=data["X_train"],
        labels=data["true_labels_train"],
    )
    lnl.fit(X=data["X_train"], labels=data["true_labels_train"], pred_probs=pred_probs)
    score_with_pred_probs = lnl.score(data["X_test"], data["true_labels_test"])
    lnl = LearningWithNoisyLabels()
    lnl.fit(
        X=data["X_train"],
        labels=data["true_labels_train"],
    )
    score_no_pred_probs = lnl.score(data["X_test"], data["true_labels_test"])
    assert abs(score_with_pred_probs - score_no_pred_probs) < 0.01


@pytest.mark.parametrize("sparse", [True, False])
def test_get_label_issues(sparse):
    data = SPARSE_DATA if sparse else DATA
    lnl = LearningWithNoisyLabels()
    lnl.fit(
        X=data["X_train"],
        labels=data["true_labels_train"],
        find_label_issues_kwargs = {"n_jobs":1, "min_examples_per_class":5}
    )
    assert all((lnl.get_label_issues() == lnl.label_issues_mask))
