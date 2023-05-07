import pytest
import numpy as np
from cleanlab.regression import rank

# To be used for all the tests
labels = np.array([1, 2, 3, 4])
predictions = np.array([2, 2, 5, 4.1])

# Used for characterization tests
expected_score_outre = np.array([0.04536998, 0.38809391, 0.03983538, 0.38809391])
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
    scores = rank.get_label_quality_scores(labels=labels, predictions=predictions)
    assert labels.shape == scores.shape
    assert isinstance(scores, np.ndarray)


def test_labels_are_arraylike(non_array_input):
    for new_input in non_array_input:
        with pytest.raises(ValueError) as error:
            rank.get_label_quality_scores(labels=new_input, predictions=predictions)
            assert error.type == ValueError


def test_predictionns_are_arraylike(non_array_input):
    for new_input in non_array_input:
        with pytest.raises(ValueError) as error:
            rank.get_label_quality_scores(labels=labels, predictions=new_input)
            assert error.type == ValueError


# test for input shapes
def test_input_shape_labels():
    with pytest.raises(AssertionError) as error:
        rank.get_label_quality_scores(labels=labels[:-1], predictions=predictions)
    assert (
        str(error.value)
        == f"Number of examples in labels {labels[:-1].shape} and predictions {predictions.shape} are not same."
    )


def test_input_shape_predictions():
    with pytest.raises(AssertionError) as error:
        rank.get_label_quality_scores(labels=labels, predictions=predictions[:-1])
    assert (
        str(error.value)
        == f"Number of examples in labels {labels.shape} and predictions {predictions[:-1].shape} are not same."
    )


# test individual scoring functions
@pytest.mark.parametrize(
    "scoring_funcs",
    [rank.get_residual_score_for_each_label, rank.get_outre_score_for_each_label],
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
    scores = rank.get_label_quality_scores(labels=labels, predictions=predictions, method=method)
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
    scores = rank.get_label_quality_scores(labels=labels, predictions=predictions, method=method)
    assert np.allclose(scores, expected_scores[method], atol=1e-08)
