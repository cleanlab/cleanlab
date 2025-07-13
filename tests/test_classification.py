# Copyright (C) 2017-2023  Cleanlab Inc.
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

from copy import deepcopy
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
import sklearn
import scipy
import pytest
import numpy as np
import pandas as pd
from cleanlab.classification import CleanLearning
from cleanlab.benchmarking.noise_generation import generate_noise_matrix_from_trace
from cleanlab.benchmarking.noise_generation import generate_noisy_labels
from cleanlab.internal.latent_algebra import compute_inv_noise_matrix
from cleanlab.count import (
    compute_confident_joint,
    estimate_cv_predicted_probabilities,
    get_confident_thresholds,
)
from cleanlab.filter import find_label_issues

SEED = 1


def make_data(
    format="numpy",
    means=[[3, 2], [7, 7], [0, 8]],
    covs=[[[5, -1.5], [-1.5, 1]], [[1, 0.5], [0.5, 4]], [[5, 1], [1, 5]]],
    sizes=[100, 50, 50],
    avg_trace=0.8,
    seed=SEED,  # set to None for non-reproducible randomness
):
    """format specifies what X (and y) looks like, one of:
    'numpy', 'sparse', 'dataframe', or 'series'.
    """
    np.random.seed(seed=seed)

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

    if format == "sparse":
        X_train = scipy.sparse.csr_matrix(X_train)
        X_test = scipy.sparse.csr_matrix(X_test)
    elif format == "dataframe":
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        # true_labels_train = list(true_labels_train)
        # true_labels_test = list(true_labels_test)
    elif format == "series":
        X_train = pd.Series(X_train[:, 0])
        X_test = pd.Series(X_test[:, 0])
        # true_labels_train = pd.Series(true_labels_train)
        # true_labels_test = pd.Series(true_labels_test)
    elif format != "numpy":
        raise ValueError("invalid value specified for: `format`.")

    # Compute p(true_label=k)
    py = np.bincount(true_labels_train) / float(len(true_labels_train))

    noise_matrix = generate_noise_matrix_from_trace(
        K,
        trace=avg_trace * K,
        py=py,
        valid_noise_matrix=True,
        seed=seed,
    )

    # Generate our noisy labels using the noise_matrix.
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


def make_rare_label(data):
    """Makes one label really rare in the dataset."""
    data = deepcopy(data)
    y = data["labels"]
    class0_inds = np.where(y == 0)[0]
    if len(class0_inds) < 1:
        raise ValueError("Class 0 too rare already")
    class0_inds_remove = class0_inds[1:]
    if len(class0_inds_remove) > 0:
        y[class0_inds_remove] = 1
    data["labels"] = y
    return data


def make_high_dim_data(seed=SEED):
    np.random.seed(seed=seed)
    X_train = np.random.randint(0, 255, (200, 28, 28))
    label_train = np.random.randint(0, 10, 200)
    X_test = np.random.randint(0, 255, (50, 28, 28))
    label_test = np.random.randint(0, 10, 50)
    X_train, X_test = X_train / 255.0, X_test / 255.0

    return {
        "X_train": X_train,
        "labels_train": label_train,
        "X_test": X_test,
        "labels_test": label_test,
    }


DATA = make_data(format="numpy", seed=SEED)
SPARSE_DATA = make_data(format="sparse", seed=SEED)
DATAFRAME_DATA = make_data(format="dataframe", seed=SEED)
SERIES_DATA = make_data(format="series", seed=SEED)  # special case not checked in most tests
HIGH_DIM_DATA = make_high_dim_data(seed=SEED)
DATA_FORMATS = {
    "numpy": DATA,
    "sparse": SPARSE_DATA,
    "dataframe": DATAFRAME_DATA,
}


@pytest.mark.parametrize("data", list(DATA_FORMATS.values()))
def test_cl(data):
    cl = CleanLearning(clf=LogisticRegression(solver="lbfgs", random_state=SEED))
    X_train_og = deepcopy(data["X_train"])
    cl.fit(data["X_train"], data["labels"])
    score = cl.score(data["X_test"], data["true_labels_test"])
    print(score)
    # ensure data has not been altered:
    if isinstance(X_train_og, np.ndarray):
        assert (data["X_train"] == X_train_og).all()
    elif isinstance(X_train_og, pd.DataFrame):
        assert data["X_train"].equals(X_train_og)


def test_cl_default_clf():
    cl = CleanLearning()  # default clf is LogisticRegression
    X_train_og = deepcopy(HIGH_DIM_DATA["X_train"])
    cl.fit(HIGH_DIM_DATA["X_train"], HIGH_DIM_DATA["labels_train"])

    # assert result has the correct length
    result = cl.predict(HIGH_DIM_DATA["X_test"])
    assert len(result) == len(HIGH_DIM_DATA["X_test"])

    result = cl.predict(X=HIGH_DIM_DATA["X_test"])
    assert len(result) == len(HIGH_DIM_DATA["X_test"])

    # assert pred_proba has the right dimensions (N x K),
    # where K = 10 (number of classes) as specified in make_high_dim_data()
    pred_proba = cl.predict_proba(HIGH_DIM_DATA["X_test"])
    assert pred_proba.shape == (len(HIGH_DIM_DATA["X_test"]), 10)

    pred_proba = cl.predict_proba(X=HIGH_DIM_DATA["X_test"])
    assert pred_proba.shape == (len(HIGH_DIM_DATA["X_test"]), 10)

    score = cl.score(HIGH_DIM_DATA["X_test"], HIGH_DIM_DATA["labels_test"])

    cl.find_label_issues(HIGH_DIM_DATA["X_train"], HIGH_DIM_DATA["labels_train"])

    # ensure data has not been altered:
    assert (HIGH_DIM_DATA["X_train"] == X_train_og).all()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("data", list(DATA_FORMATS.values()))
def test_rare_label(data):
    data = make_rare_label(data)
    test_cl(data)


def test_invalid_inputs():
    data = make_data(sizes=[1, 1, 1])
    try:
        test_cl(data)
    except Exception as e:
        assert "Need more data" in str(e)
    else:
        raise Exception("expected test to raise Exception")
    try:
        cl = CleanLearning(
            clf=LogisticRegression(solver="lbfgs", random_state=SEED),
            find_label_issues_kwargs={"return_indices_ranked_by": "self_confidence"},
        )
        cl.fit(
            data["X_train"],
            data["labels"],
        )
    except Exception as e:
        assert "not supported" in str(e) or "Need more data from each class" in str(e)
    else:
        raise Exception("expected test to raise Exception")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_aux_inputs():
    data = DATA
    K = len(np.unique(data["labels"]))
    confident_joint = np.ones(shape=(K, K))
    np.fill_diagonal(confident_joint, 10)
    find_label_issues_kwargs = {
        "confident_joint": confident_joint,
        "min_examples_per_class": 2,
    }
    cl = CleanLearning(
        clf=LogisticRegression(solver="lbfgs", random_state=SEED),
        find_label_issues_kwargs=find_label_issues_kwargs,
        verbose=1,
    )
    label_issues_df = cl.find_label_issues(data["X_train"], data["labels"], clf_kwargs={})
    assert isinstance(label_issues_df, pd.DataFrame)
    FIND_OUTPUT_COLUMNS = ["is_label_issue", "label_quality", "given_label", "predicted_label"]
    assert list(label_issues_df.columns) == FIND_OUTPUT_COLUMNS
    assert label_issues_df.equals(cl.get_label_issues())
    cl.fit(
        data["X_train"],
        data["labels"],
        label_issues=label_issues_df,
        clf_kwargs={},
        clf_final_kwargs={},
    )
    label_issues_df = cl.get_label_issues()
    assert isinstance(label_issues_df, pd.DataFrame)
    assert list(label_issues_df.columns) == (FIND_OUTPUT_COLUMNS + ["sample_weight"])
    score = cl.score(data["X_test"], data["true_labels_test"])

    # Test a second fit
    cl.fit(data["X_train"], data["labels"])

    # Test cl.find_label_issues with pred_prob input
    pred_probs_test = cl.predict_proba(data["X_test"])
    label_issues_df = cl.find_label_issues(
        X=None, labels=data["true_labels_test"], pred_probs=pred_probs_test
    )
    assert isinstance(label_issues_df, pd.DataFrame)
    assert list(label_issues_df.columns) == FIND_OUTPUT_COLUMNS
    assert label_issues_df.equals(cl.get_label_issues())
    cl.save_space()
    assert cl.label_issues_df is None

    # Verbose off
    cl = CleanLearning(clf=LogisticRegression(solver="lbfgs", random_state=SEED), verbose=0)
    cl.save_space()  # dummy call test

    cl = CleanLearning(clf=LogisticRegression(solver="lbfgs", random_state=SEED), verbose=0)
    cl.find_label_issues(
        labels=data["true_labels_test"], pred_probs=pred_probs_test, save_space=True
    )

    cl = CleanLearning(clf=LogisticRegression(solver="lbfgs", random_state=SEED), verbose=1)

    # Test with label_issues_mask input
    label_issues_mask = find_label_issues(
        labels=data["true_labels_test"],
        pred_probs=pred_probs_test,
    )
    cl.fit(data["X_test"], data["true_labels_test"], label_issues=label_issues_mask)
    label_issues_df = cl.get_label_issues()
    assert isinstance(label_issues_df, pd.DataFrame)
    assert set(label_issues_df.columns).issubset(FIND_OUTPUT_COLUMNS)

    # Test with label_issues_indices input
    label_issues_indices = find_label_issues(
        labels=data["true_labels_test"],
        pred_probs=pred_probs_test,
        return_indices_ranked_by="confidence_weighted_entropy",
    )
    cl.fit(data["X_test"], data["true_labels_test"], label_issues=label_issues_indices)
    label_issues_df2 = cl.get_label_issues().copy()
    assert isinstance(label_issues_df2, pd.DataFrame)
    assert set(label_issues_df2.columns).issubset(FIND_OUTPUT_COLUMNS)
    assert label_issues_df2["is_label_issue"].equals(label_issues_df["is_label_issue"])

    # Test fit() with pred_prob input:
    cl.fit(
        data["X_test"],
        data["true_labels_test"],
        pred_probs=pred_probs_test,
        label_issues=label_issues_mask,
    )
    label_issues_df = cl.get_label_issues()
    assert isinstance(label_issues_df, pd.DataFrame)
    assert set(label_issues_df.columns).issubset(FIND_OUTPUT_COLUMNS)
    assert "label_quality" in label_issues_df.columns

    # Test with sample_weight input:
    cl = CleanLearning(clf=LogisticRegression(solver="lbfgs", random_state=SEED), verbose=1)
    cl.fit(
        data["X_test"],
        data["true_labels_test"],
        sample_weight=np.random.randn(len(data["true_labels_test"])),
    )
    cl.fit(
        data["X_test"],
        data["true_labels_test"],
        label_issues=cl.get_label_issues(),
        sample_weight=np.random.randn(len(data["true_labels_test"])),
    )


class LogisticRegressionWithValidationData(LogisticRegression):
    def fit(self, X, y, X_val=None, y_val=None):
        super().fit(X, y)

        # Final fit() call does not use validation data
        # Checks to prevent arg missing error
        if X_val is not None or y_val is not None:
            print(self.score(X_val, y_val))


def val_func(X_val, y_val):
    return {"X_val": X_val, "y_val": y_val}


def test_validation_data():
    data = DATA
    cl = CleanLearning(clf=LogisticRegressionWithValidationData())
    cl.fit(
        data["X_train"],
        data["labels"],
        validation_func=val_func,
    )


def test_raise_error_no_clf_fit():
    class struct(object):
        def predict(self):
            pass

        def predict_proba(self):
            pass

    try:
        CleanLearning(clf=struct())
    except Exception as e:
        assert "fit" in str(e)
        with pytest.raises(ValueError) as e:
            CleanLearning(clf=struct())


def test_raise_error_no_clf_predict_proba():
    class struct(object):
        def fit(self):
            pass

        def predict(self):
            pass

    try:
        CleanLearning(clf=struct())
    except Exception as e:
        assert "predict_proba" in str(e)
        with pytest.raises(ValueError) as e:
            CleanLearning(clf=struct())


def test_raise_error_no_clf_predict():
    class struct(object):
        def fit(self):
            pass

        def predict_proba(self):
            pass

    try:
        CleanLearning(clf=struct())
    except Exception as e:
        assert "predict" in str(e)
        with pytest.raises(ValueError) as e:
            CleanLearning(clf=struct())


def test_seed():
    cl = CleanLearning(seed=SEED)
    assert cl.seed is not None


def test_default_clf():
    cl = CleanLearning()
    check1 = cl.clf is not None and hasattr(cl.clf, "fit")
    check2 = hasattr(cl.clf, "predict") and hasattr(cl.clf, "predict_proba")
    assert check1 and check2


def test_clf_fit_nm():
    cl = CleanLearning()
    # Example of a bad noise matrix (impossible to learn from)
    nm = np.array([[0, 1], [1, 0]])
    try:
        cl.fit(X=np.arange(3), labels=np.array([0, 0, 1]), noise_matrix=nm)
    except Exception as e:
        assert "Trace(noise_matrix)" in str(e)
        with pytest.raises(ValueError) as e:
            cl.fit(X=np.arange(3), labels=np.array([0, 0, 1]), noise_matrix=nm)


def test_clf_fit_inm():
    cl = CleanLearning()
    # Example of a bad noise matrix (impossible to learn from)
    inm = np.array([[0.1, 0.9], [0.9, 0.1]])
    try:
        cl.fit(X=np.arange(3), labels=np.array([0, 0, 1]), inverse_noise_matrix=inm)
    except Exception as e:
        assert "Trace(inverse_noise_matrix)" in str(e)
        with pytest.raises(ValueError) as e:
            cl.fit(X=np.arange(3), labels=np.array([0, 0, 1]), inverse_noise_matrix=inm)


@pytest.mark.parametrize("format", list(DATA_FORMATS.keys()))
def test_fit_with_nm(
    format,
    seed=SEED,
    used_by_another_test=False,
):
    data = DATA_FORMATS[format]
    cl = CleanLearning(
        seed=seed,
    )
    nm = data["noise_matrix"]
    # Learn with noisy labels with noise matrix given
    cl.fit(data["X_train"], data["labels"], noise_matrix=nm)
    score_nm = cl.score(data["X_test"], data["true_labels_test"])
    # Learn with noisy labels and estimate the noise matrix.
    cl2 = CleanLearning(
        seed=seed,
    )
    cl2.fit(
        data["X_train"],
        data["labels"],
    )
    score = cl2.score(data["X_test"], data["true_labels_test"])
    if used_by_another_test:
        return score, score_nm
    else:
        assert score < score_nm + 1e-4


@pytest.mark.parametrize("format", list(DATA_FORMATS.keys()))
def test_fit_with_inm(
    format,
    seed=SEED,
    used_by_another_test=False,
):
    data = DATA_FORMATS[format]
    cl = CleanLearning(
        seed=seed,
    )
    inm = compute_inv_noise_matrix(
        py=data["py"],
        noise_matrix=data["noise_matrix"],
        ps=data["ps"],
    )
    # Learn with noisy labels with inverse noise matrix given
    cl.fit(data["X_train"], data["labels"], inverse_noise_matrix=inm)
    score_inm = cl.score(data["X_test"], data["true_labels_test"])
    # Learn with noisy labels and estimate the inv noise matrix.
    cl2 = CleanLearning(
        seed=seed,
    )
    cl2.fit(
        data["X_train"],
        data["labels"],
    )
    score = cl2.score(data["X_test"], data["true_labels_test"])
    if used_by_another_test:
        return score, score_inm
    else:
        assert score < score_inm + 1e-4


@pytest.mark.parametrize("format", list(DATA_FORMATS.keys()))
def test_clf_fit_nm_inm(format):
    data = DATA_FORMATS[format]
    cl = CleanLearning(seed=SEED)
    nm = data["noise_matrix"]
    inm = compute_inv_noise_matrix(
        py=data["py"],
        noise_matrix=nm,
        ps=data["ps"],
    )
    cl.fit(
        X=data["X_train"],
        labels=data["labels"],
        noise_matrix=nm,
        inverse_noise_matrix=inm,
    )
    score_nm_inm = cl.score(data["X_test"], data["true_labels_test"])

    # Learn with noisy labels and estimate the inv noise matrix.
    cl2 = CleanLearning(seed=SEED)
    cl2.fit(
        data["X_train"],
        data["labels"],
    )
    score = cl2.score(data["X_test"], data["true_labels_test"])
    assert score < score_nm_inm + 1e-4


@pytest.mark.parametrize("format", list(DATA_FORMATS.keys()))
def test_clf_fit_y_alias(format):
    data = DATA_FORMATS[format]
    cl = CleanLearning(seed=SEED)

    # Valid signature
    cl.fit(data["X_train"], data["labels"])

    # Valid signature for labels/y alias
    cl.fit(data["X_train"], labels=data["labels"])
    cl.fit(data["X_train"], y=data["labels"])
    cl.fit(X=data["X_train"], labels=data["labels"])
    cl.fit(X=data["X_train"], y=data["labels"])

    # Invalid signatures
    with pytest.raises(ValueError):
        cl.fit(data["X_train"])
    with pytest.raises(ValueError):
        cl.fit(data["X_train"], data["labels"], y=data["labels"])
    with pytest.raises(ValueError):
        cl.fit(X=data["X_train"], labels=data["labels"], y=data["labels"])


@pytest.mark.parametrize("format", list(DATA_FORMATS.keys()))
def test_pred_and_pred_proba(format):
    data = DATA_FORMATS[format]
    cl = CleanLearning()
    cl.fit(data["X_train"], data["labels"])
    n = np.shape(data["true_labels_test"])[0]
    m = len(np.unique(data["true_labels_test"]))
    pred = cl.predict(data["X_test"])
    probs = cl.predict_proba(data["X_test"])
    # Just check that this functions return what we expect
    assert np.shape(pred)[0] == n
    assert np.shape(probs) == (n, m)


@pytest.mark.parametrize("format", list(DATA_FORMATS.keys()))
def test_score(format):
    data = DATA_FORMATS[format]
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

    cl = CleanLearning(clf=Struct())
    score = cl.score(data["X_test"], data["true_labels_test"])
    assert score == phrase


@pytest.mark.parametrize("format", list(DATA_FORMATS.keys()))
def test_no_score(format):
    data = DATA_FORMATS[format]

    class Struct:
        def fit(self):
            pass

        def predict_proba(self):
            pass

        def predict(self, X):
            return data["true_labels_test"]

    cl = CleanLearning(clf=Struct())
    score = cl.score(data["X_test"], data["true_labels_test"])
    assert abs(score - 1) < 1e-6


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("format", list(DATA_FORMATS.keys()))
def test_no_fit_sample_weight(format):
    data = DATA_FORMATS[format]

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
    cl = CleanLearning(clf=Struct())
    cl.fit(
        data["X_train"],
        data["true_labels_train"],
        pred_probs=pred_probs,
        noise_matrix=data["noise_matrix"],
    )
    # If we make it here, without any error:


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("format", list(DATA_FORMATS.keys()))
def test_fit_pred_probs(format):
    data = DATA_FORMATS[format]

    cl = CleanLearning()
    pred_probs = estimate_cv_predicted_probabilities(
        X=data["X_train"],
        labels=data["true_labels_train"],
    )
    cl.fit(X=data["X_train"], labels=data["true_labels_train"], pred_probs=pred_probs)
    score_with_pred_probs = cl.score(data["X_test"], data["true_labels_test"])
    cl = CleanLearning()
    cl.fit(
        X=data["X_train"],
        labels=data["true_labels_train"],
    )
    score_no_pred_probs = cl.score(data["X_test"], data["true_labels_test"])
    assert abs(score_with_pred_probs - score_no_pred_probs) < 0.01


def make_2d(X):
    X = np.asarray(X)
    return X.reshape(X.shape[0], -1)


class ReshapingLogisticRegression(BaseEstimator):
    def __init__(self):
        self.clf = LogisticRegression()

    def fit(self, X, y):
        y = np.asarray(y).flatten()
        self.clf.fit(make_2d(X), y)

    def predict(self, X):
        return self.clf.predict(make_2d(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(make_2d(X))

    def score(self, X, y, sample_weight=None):
        return self.clf.score(make_2d(X), y, sample_weight=sample_weight)


def dimN_data(N):
    size = [100] + [3 for _ in range(N - 1)]
    X = np.random.normal(size=size)
    labels = np.random.randint(0, 4, size=100)
    # ensure that every class is represented
    labels[0:10] = 0
    labels[11:20] = 1
    labels[21:30] = 2
    labels[31:40] = 3
    return X, labels


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("N", [1, 3, 4])
def test_dimN(N):
    X, labels = dimN_data(N)
    cl = CleanLearning(clf=ReshapingLogisticRegression())
    # just make sure we don't crash...
    cl.fit(X, labels)
    cl.predict(X)
    cl.predict_proba(X)
    cl.score(X, labels)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_1D_formats():
    X, labels = dimN_data(1)
    X_series = pd.Series(X)
    labels_series = pd.Series(labels)
    idx = list(np.random.choice(len(labels), size=len(labels), replace=False))
    X_series.index = idx
    labels_series.index = idx
    cl = CleanLearning(clf=ReshapingLogisticRegression())
    # just make sure we don't crash...
    cl.fit(X_series, labels_series)
    cl.predict(X_series)
    cl.predict_proba(X_series)
    cl.score(X_series, labels)
    # Repeat with rare labels:
    labels_rare = deepcopy(labels)
    class0_inds = np.where(labels_rare == 0)[0]
    class0_inds_remove = class0_inds[1:]
    labels_rare[class0_inds_remove] = 1
    cl = CleanLearning(clf=ReshapingLogisticRegression())
    cl.fit(X_series, labels_rare)
    cl.predict(X_series)
    cl.predict_proba(X_series)
    cl.score(X_series, labels)
    # Repeat with DataFrame labels:
    labels_df = pd.DataFrame({"colname": labels})
    cl = CleanLearning(clf=ReshapingLogisticRegression())
    cl.fit(X, labels_df)
    cl.predict(X)
    pred_probs = cl.predict_proba(X)
    cl.score(X, labels)
    # Repeat with DataFrame labels and pred_probs
    cl = CleanLearning(clf=ReshapingLogisticRegression())
    cl.fit(X, labels_df, pred_probs=pred_probs)
    # Repeat with list labels:
    labels_list = list(labels)
    cl = CleanLearning(clf=ReshapingLogisticRegression())
    cl.fit(X, labels_list)
    cl.predict(X)
    cl.predict_proba(X)
    cl.score(X, labels)


# Check if the current Python version is 3.11
is_python_311 = sys.version_info.major == 3 and sys.version_info.minor == 11

# This warning should be ignored as in Python 3.11, the sre_constants module has been deprecated.
# At the time of writing this, cleanlab supports Python 3.8-3.11. This warning is raised by
# tensorflow <2.14.0, which imports sre_constants. This warning is not relevant to cleanlab.
# Once Python 3.8 reaches EOL, we may remove this warning filter as we can set the tensorflow
# dev-dependency to a version that does not raise this warning (2.14 or higher).
if is_python_311:
    sre_deprecation_pytestmark = pytest.mark.filterwarnings(
        "ignore:module 'sre_constants' is deprecated"
    )
else:
    sre_deprecation_pytestmark = pytest.mark.filterwarnings("default")

# Check if the installed version of sklearn is 1.5.0.
# The test_sklearn_gridsearchcv test fails due to a regression introduced in 1.5.0.
# This issue will be fixed in sklearn version 1.5.1.
uses_sklearn_1_5_0 = sklearn.__version__ == "1.5.0"


@sre_deprecation_pytestmark  # Allow sre_constants deprecation warning for Python 3.11
@pytest.mark.filterwarnings("error")  # All other warnings are treated as errors
@pytest.mark.skipif(
    uses_sklearn_1_5_0,
    reason="Test is skipped because sklearn 1.5.0 is installed, which has a regression for GridSearchCV.",
)  # TODO: Remove this line once sklearn 1.5.1 is released
def test_sklearn_gridsearchcv():
    # hyper-parameters for grid search
    param_grid = {
        "find_label_issues_kwargs": [
            {"filter_by": "prune_by_noise_rate"},
            {"filter_by": "prune_by_class"},
            {"filter_by": "both"},
            {"filter_by": "confident_learning"},
            {"filter_by": "predicted_neq_given"},
        ],
        "converge_latent_estimates": [True, False],
    }

    clf = LogisticRegression(random_state=0, solver="newton-cg")

    cv = GridSearchCV(
        estimator=CleanLearning(clf),
        param_grid=param_grid,
        cv=3,
    )

    # cv.fit() raises a warning if some fits fail (including raising
    # exceptions); we don't expect any fits to fail, so ensure that the code
    # doesn't raise any warnings
    cv.fit(X=DATA["X_train"], y=DATA["labels"])


@pytest.mark.parametrize("filter_by", ["both", "confident_learning"])
@pytest.mark.parametrize("seed", [0, 6, 2])
def test_cj_in_find_label_issues_kwargs(filter_by, seed):
    labels = DATA["labels"]
    num_issues = []
    for provide_confident_joint in [True, False]:
        print(f"\nfilter_by: {filter_by} | seed: {seed} | cj_provided: {provide_confident_joint}")
        np.random.seed(seed=seed)
        if provide_confident_joint:
            pred_probs = estimate_cv_predicted_probabilities(
                X=DATA["X_train"], labels=labels, seed=seed
            )
            confident_joint = compute_confident_joint(labels=labels, pred_probs=pred_probs)
            cl = CleanLearning(
                find_label_issues_kwargs={
                    "confident_joint": confident_joint,
                    "filter_by": "both",
                    "min_examples_per_class": 1,
                },
                verbose=1,
            )
        else:
            cl = CleanLearning(
                clf=LogisticRegression(random_state=seed),
                find_label_issues_kwargs={
                    "filter_by": "both",
                    "min_examples_per_class": 1,
                },
                verbose=0,
            )
        label_issues_df = cl.find_label_issues(DATA["X_train"], labels=labels)
        label_issues_mask = label_issues_df["is_label_issue"].values
        # Check if the noise matrix was computed based on the passed in confident joint
        cj_reconstruct = (cl.inverse_noise_matrix * np.bincount(DATA["labels"])).T.astype(int)
        np.all(cl.confident_joint == cj_reconstruct)
        num_issues.append(sum(label_issues_mask))

    # Chceck that the same exact number of issues are found regardless if the confident joint
    # is computed during find_label_issues or precomputed and provided as a kwargs parameter.
    assert num_issues[0] == num_issues[1]


def test_find_label_issues_uses_thresholds():
    X = DATA["X_train"]
    labels = DATA["labels"]
    pred_probs = estimate_cv_predicted_probabilities(X=X, labels=labels)

    confident_thresholds = get_confident_thresholds(labels=labels, pred_probs=pred_probs)
    confident_joint = compute_confident_joint(labels=labels, pred_probs=pred_probs)

    # regular find label issues with no args
    cl = CleanLearning()
    label_issues_reg = cl.find_label_issues(labels=labels, pred_probs=pred_probs)

    # find label issues with specified confident thresholds
    cl = CleanLearning()
    label_issues_thres = cl.find_label_issues(
        labels=labels, pred_probs=pred_probs, thresholds=confident_thresholds
    )

    # find label issues with specified confident joint
    cl = CleanLearning(
        find_label_issues_kwargs={
            "confident_joint": confident_joint,
        }
    )
    label_issues_cj = cl.find_label_issues(labels=labels, pred_probs=pred_probs)

    # the labels issues in above three calls should be the same
    assert np.sum(label_issues_reg["is_label_issue"]) == np.sum(
        label_issues_thres["is_label_issue"]
    )
    assert np.sum(label_issues_reg["is_label_issue"]) == np.sum(label_issues_cj["is_label_issue"])

    # find label issues with different specified confident thresholds
    confident_thresholds_alt = np.full(pred_probs.shape[1], 0.25)
    cl = CleanLearning()
    label_issues_thres_alt = cl.find_label_issues(
        labels=labels, pred_probs=pred_probs, thresholds=confident_thresholds_alt
    )

    # find label issues with different specified confident joint
    confident_joint_alt = compute_confident_joint(
        labels=labels, pred_probs=pred_probs, thresholds=confident_thresholds_alt
    )
    cl = CleanLearning(
        find_label_issues_kwargs={
            "confident_joint": confident_joint_alt,
        }
    )
    label_issues_cj_alt = cl.find_label_issues(labels=labels, pred_probs=pred_probs)

    # the number of issues for these 2 alt calls should be same as one another, but different from above 3
    assert np.sum(label_issues_thres_alt["is_label_issue"]) == np.sum(
        label_issues_cj_alt["is_label_issue"]
    )
    assert np.sum(label_issues_thres_alt["is_label_issue"]) != np.sum(
        label_issues_reg["is_label_issue"]
    )


def test_find_issues_missing_classes():
    labels = np.array([0, 0, 2, 2])
    pred_probs = np.array(
        [[0.9, 0.0, 0.1, 0.0], [0.8, 0.0, 0.2, 0.0], [0.1, 0.0, 0.9, 0.0], [0.95, 0.0, 0.05, 0.0]]
    )
    issues_df = CleanLearning(
        find_label_issues_kwargs={"min_examples_per_class": 0}
    ).find_label_issues(labels=labels, pred_probs=pred_probs)
    issues = issues_df["is_label_issue"].values
    assert np.all(issues == np.array([False, False, False, True]))
    # Check results match without these missing classes present in pred_probs:
    pred_probs2 = pred_probs[:, list(sorted(np.unique(labels)))]
    labels2 = np.array([0, 0, 1, 1])
    issues_df2 = CleanLearning(
        find_label_issues_kwargs={"min_examples_per_class": 0}
    ).find_label_issues(labels=labels2, pred_probs=pred_probs2)
    assert all(issues_df2["is_label_issue"].values == issues)


def test_find_issues_low_memory():
    X = DATA["X_train"]
    labels = DATA["labels"]
    pred_probs = estimate_cv_predicted_probabilities(X=X, labels=labels, seed=SEED)
    issues_df = CleanLearning().find_label_issues(labels=labels, pred_probs=pred_probs)
    issues_df_lm = CleanLearning(low_memory=True).find_label_issues(
        labels=labels, pred_probs=pred_probs
    )
    # check jaccard similarity:
    intersection = len(list(set(issues_df).intersection(set(issues_df_lm))))
    union = len(set(issues_df)) + len(set(issues_df_lm)) - intersection
    assert float(intersection) / union > 0.95
    # Without pred_probs
    issues_df = CleanLearning(low_memory=True, verbose=True, seed=SEED).find_label_issues(
        X=X, labels=labels
    )
    assert issues_df.equals(issues_df_lm)
    # With unused arguments find_label_issues_kwargs and noise_matrix
    find_label_issues_kwargs = {"min_examples_per_class": 2}
    issues_df = CleanLearning(
        low_memory=True, find_label_issues_kwargs=find_label_issues_kwargs, seed=SEED
    ).find_label_issues(X=X, labels=labels, noise_matrix=DATA["noise_matrix"])
    assert issues_df.equals(issues_df_lm)


def test_confident_joint_setting_in_find_label_issues_kwargs():
    """
    This test ensures that the 'confident_joint' is correctly set in the
    'find_label_issues_kwargs' of the 'CleanLearning' class when calling find_label_issues().

    This test was added to cover the lines of code that were previously
    missed due to the removal of another test.
    """
    # Load training data and labels
    X = DATA["X_train"]
    labels = DATA["labels"]

    # Estimate predicted probabilities using cross-validation
    pred_probs = estimate_cv_predicted_probabilities(X=X, labels=labels, seed=SEED)

    # Initialize CleanLearning instance
    cl = CleanLearning()

    # Test that the confident joint is not set initially
    cj = cl.find_label_issues_kwargs.get("confident_joint")
    assert cj is None, "Initial confident_joint should be None"

    # Call find_label_issues to set the confident joint
    cl.find_label_issues(labels=labels, pred_probs=pred_probs)
    cj = cl.find_label_issues_kwargs.get("confident_joint")

    # Compute expected confident joint
    expected_cj = compute_confident_joint(labels=labels, pred_probs=pred_probs)

    # Assert that the confident joint is set correctly
    np.testing.assert_array_equal(
        cj, expected_cj, "Confident joint not set correctly after find_label_issues"
    )

    # Pass a precomputed confident_joint to the CleanLearning instance
    cj_as_input = np.random.rand(3, 3)
    cl = CleanLearning(
        find_label_issues_kwargs={
            "confident_joint": cj_as_input,
        }
    )

    # Ensure the precomputed confident joint is used
    cj = cl.find_label_issues_kwargs.get("confident_joint")
    np.testing.assert_array_equal(
        cj, cj_as_input, "Confident joint not set correctly when passed as input"
    )

    # Calling find_label_issues should not change the precomputed confident_joint
    cl.find_label_issues(labels=labels, pred_probs=pred_probs)
    cj = cl.find_label_issues_kwargs.get("confident_joint")
    np.testing.assert_array_equal(
        cj,
        cj_as_input,
        "Confident joint should not change after find_label_issues call when precomputed joint is provided",
    )
