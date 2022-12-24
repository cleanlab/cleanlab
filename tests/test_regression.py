import numpy as np
import pandas as pd
import pytest

from cleanlab.regression import rank

# To be used for all the tests
labels = np.array([1, 2, 3, 4])
predictions = np.array([1, 3, 4, 5])

# test with deafault parameters
def test_output_shape_type():
    scores = rank.get_label_quality_scores(labels=labels, predictions=predictions)
    assert labels.shape == scores.shape
    assert isinstance(scores, np.ndarray)


# test for acceptable datatypes
@pytest.mark.parametrize("format", [pd.Series, pd.DataFrame, list])
def test_type_error_for_input_types(format):
    with pytest.raises(TypeError) as error:
        _ = rank.get_label_quality_scores(labels=format(labels), predictions=format(predictions))


# test for input shapes
def test_assertion_error_for_input_shape():
    with pytest.raises(AssertionError) as error:
        _ = rank.get_label_quality_scores(labels=labels[:-1], predictions=predictions)
        _ = rank.get_label_quality_scores(labels=labels, predictions=predictions[:-1])


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
