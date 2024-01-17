import pytest
import random
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from cleanlab.regression.rank import (
    get_label_quality_scores,
    _get_residual_score_for_each_label,
    _get_outre_score_for_each_label,
)
from cleanlab.regression.learn import CleanLearning

# set seed for reproducability
SEED = 1
np.random.seed(SEED)
random.seed(SEED)


def make_data(num_examples=200, num_features=3, noise=0.2, error_frac=0.1, error_noise=5):
    X = np.random.random(size=(num_examples, num_features))
    coefficients = np.random.uniform(-1, 1, size=num_features)
    label_noise = np.random.normal(scale=noise, size=num_examples)

    true_y = np.dot(X, coefficients)
    y = np.dot(X, coefficients) + label_noise

    # add extra noisy examples
    num_errors = int(num_examples * error_frac)
    extra_noise = np.random.normal(scale=error_noise, size=num_errors)
    random_idx = np.random.choice(num_examples, num_errors)
    y[random_idx] += extra_noise
    error_idx = np.argsort(abs(y - true_y))[-num_errors:]  # get the noisiest examples idx

    # create test set
    X_test = np.random.random(size=(num_examples, num_features))
    label_noise = np.random.normal(scale=noise, size=num_examples)
    y_test = np.dot(X_test, coefficients) + label_noise

    return {
        "X": X,
        "y": y,
        "true_y": true_y,
        "X_test": X_test,
        "y_test": y_test,
        "error_idx": error_idx,
    }


# To be used for most tests
data = make_data()
X, labels, predictions = data["X"], data["y"], data["true_y"]
error_idx = data["error_idx"]
X_test, y_test = data["X_test"], data["y_test"]
y = labels  # for ease

# Used for characterization tests
small_labels = np.array([1, 2, 3, 4])
small_predictions = np.array([2, 2, 5, 4.1])
expected_score_outre = np.array([0.2162406, 0.62585509, 0.20275104, 0.62585509])
expected_score_residual = np.array([0.36787944, 1.0, 0.13533528, 0.90483742])
expected_scores = {"outre": expected_score_outre, "residual": expected_score_residual}

# Inputs that are not array like
aConstant = 1
aString = "predictions_non_array"
aDict = {"labels": [1, 2], "predictions": [2, 3]}
aSet = {1, 2, 3, 4}
aBool = True


@pytest.fixture
def non_array_input():
    return [aConstant, aString, aDict, aSet, aBool]


# test with deafault parameters
def test_output_shape_type():
    scores = get_label_quality_scores(labels=labels, predictions=predictions)
    assert labels.shape == scores.shape
    assert isinstance(scores, np.ndarray)


def test_labels_are_arraylike(non_array_input):
    for new_input in non_array_input:
        with pytest.raises(ValueError) as error:
            get_label_quality_scores(labels=new_input, predictions=predictions)
            assert error.type == ValueError


def test_predictionns_are_arraylike(non_array_input):
    for new_input in non_array_input:
        with pytest.raises(ValueError) as error:
            get_label_quality_scores(labels=labels, predictions=new_input)
            assert error.type == ValueError


# test for input shapes
def test_input_shape_labels():
    with pytest.raises(AssertionError) as error:
        get_label_quality_scores(labels=labels[:-1], predictions=predictions)
    assert (
        str(error.value)
        == f"Number of examples in labels {labels[:-1].shape} and predictions {predictions.shape} are not same."
    )


def test_input_shape_predictions():
    with pytest.raises(AssertionError) as error:
        get_label_quality_scores(labels=labels, predictions=predictions[:-1])
    assert (
        str(error.value)
        == f"Number of examples in labels {labels.shape} and predictions {predictions[:-1].shape} are not same."
    )


# test individual scoring functions
@pytest.mark.parametrize(
    "scoring_funcs",
    [_get_residual_score_for_each_label, _get_outre_score_for_each_label],
)
def test_individual_scoring_functions(scoring_funcs):
    scores = scoring_funcs(labels=labels, predictions=predictions)
    assert labels.shape == scores.shape
    assert isinstance(scores, np.ndarray)


# test for method argument
@pytest.mark.parametrize(
    "method",
    [
        "residual",
        "outre",
    ],
)
def test_method_pass_get_label_quality_scores(method):
    scores = get_label_quality_scores(labels=labels, predictions=predictions, method=method)
    assert labels.shape == scores.shape
    assert isinstance(scores, np.ndarray)


@pytest.mark.parametrize(
    "method",
    [
        "residual",
        "outre",
    ],
)
def test_expected_scores(method):
    # characterization test
    scores = get_label_quality_scores(
        labels=small_labels, predictions=small_predictions, method=method
    )
    assert np.allclose(scores, expected_scores[method], atol=1e-08)


def test_cleanlearning():
    # test fit and predict
    cl = CleanLearning()
    cl.fit(X, y)
    preds = cl.predict(X)
    cl_r2_score = cl.score(X, y)
    manual_r2_score = r2_score(y, preds)
    assert len(preds) == len(y)
    assert isinstance(cl_r2_score, float)
    assert cl_r2_score == manual_r2_score

    # check if label issues were identified
    label_issues = cl.get_label_issues()
    identified_label_issues = label_issues[label_issues["is_label_issue"] == True].index
    frac_errors_identified = np.mean([e in identified_label_issues for e in error_idx])
    assert frac_errors_identified >= 0.9  # assert most errors were detected

    # compare perf to base LinearRegression model
    cl_score = cl.score(X_test, y_test)
    lr = LinearRegression()
    lr.fit(X, y)
    lr_score = lr.score(X_test, y_test)
    assert cl_score > lr_score

    # test passing in label issues in various forms
    # also test different regression model
    cl = CleanLearning(model=SVR())
    label_issues = cl.find_label_issues(X, y)
    assert isinstance(label_issues, pd.DataFrame)

    cl.fit(X, y, label_issues=label_issues)
    cl.fit(X, pd.Series(y), label_issues=label_issues["is_label_issue"])
    cl.fit(X, list(y), label_issues=label_issues["is_label_issue"].values)


def test_optional_inputs():
    # test with sample_weight input
    cl = CleanLearning(verbose=1)
    cl.fit(X, y, sample_weight=np.random.random(size=len(y)))
    cl.fit(X, y, label_issues=cl.get_label_issues(), sample_weight=np.random.random(size=len(y)))

    # test with uncertainty input
    cl = CleanLearning()
    cl.find_label_issues(X, y, uncertainty=5)  # constant uncertainty
    cl.find_label_issues(X, y, uncertainty=np.random.random(size=len(y)))  # per-example uncertainty

    # test with not calculating uncertainty
    cl = CleanLearning(n_boot=0, include_aleatoric_uncertainty=False)
    cl.find_label_issues(X, y)

    # test with odd grid search sizes
    cl = CleanLearning()
    cl.find_label_issues(X, y, coarse_search_range=[0.2])
    cl.find_label_issues(X, y, fine_search_size=0)
    cl.fit(
        X, y, find_label_issues_kwargs={"coarse_search_range": [0.2, 0.1], "fine_search_size": 2}
    )


def test_low_example_count():
    data_tiny = make_data(num_examples=3)
    X_tiny, y_tiny = data_tiny["X"], data_tiny["y"]

    try:
        cl = CleanLearning()
        cl.find_label_issues(X_tiny, y_tiny)
    except ValueError as e:
        assert "There are too few examples" in str(e)

    cl = CleanLearning(cv_n_folds=3)
    cl.find_label_issues(X_tiny, y_tiny)
    assert isinstance(cl.get_label_issues(), pd.DataFrame)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_save_space():
    # test label issues df does not save
    cl = CleanLearning()
    cl.find_label_issues(X, y, save_space=True)
    assert cl.get_label_issues() is None

    # test label issues df deletes properly
    cl = CleanLearning()
    cl.find_label_issues(X, y)
    assert isinstance(cl.get_label_issues(), pd.DataFrame)

    cl.save_space()
    assert cl.get_label_issues() is None
