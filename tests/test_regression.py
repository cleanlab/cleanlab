import numpy as np

# import pandas as pd
import pytest
from typing import Union, Sequence

from cleanlab.regression import rank

ArrayLike = Union[np.ndarray, Sequence]

# To be used for all the tests
labels = np.array([1, 2, 3, 4])
predictions = np.array([1, 3, 4, 5])

# Inputs that are not array like
aConstant = 1
aString = "predictions_non_array"
aDict = {"labels": [1, 2], "predictions": [2, 3]}
aSet = {1, 2, 3, 4}
aBool = True


# test with deafault parameters
def test_output_shape_type():
    scores = rank.get_label_quality_scores(labels=labels, predictions=predictions)
    assert labels.shape == scores.shape
    assert isinstance(scores, np.ndarray)


@pytest.mark.parametrize(
    "aInput",
    [aConstant, aString, aDict, aSet, aBool],
)
def test_labels_are_arraylike(aInput):
    with pytest.raises(ValueError) as error:
        rank.get_label_quality_scores(labels=aInput, predictions=predictions)
        assert error.type == ValueError


@pytest.mark.parametrize(
    "aInput",
    [aConstant, aString, aDict, aSet, aBool],
)
def test_predictionns_are_arraylike(aInput):
    with pytest.raises(ValueError) as error:
        rank.get_label_quality_scores(labels=labels, predictions=aInput)
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
