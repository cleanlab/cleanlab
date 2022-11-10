import numpy as np
import pandas as pd
import pytest

from cleanlab.regression.rank import get_label_quality_scores

# To be used for all the tests
labels = np.array([1, 2, 3, 4])
pred_labels = np.array([1, 3, 4, 5])


def test_output_shape_type():
    scores = get_label_quality_scores(labels=labels, pred_labels=pred_labels)
    assert labels.shape == scores.shape
    assert isinstance(scores, np.ndarray)


@pytest.mark.parametrize("format", [pd.Series, pd.DataFrame, list])
def test_type_error_for_input_types(format):
    with pytest.raises(TypeError) as error:
        _ = get_label_quality_scores(labels=format(labels), pred_labels=format(pred_labels))


def test_assertion_error_for_input_shape():
    with pytest.raises(AssertionError) as error:
        _ = get_label_quality_scores(labels=labels[:-1], pred_labels=pred_labels)
        _ = get_label_quality_scores(labels=labels, pred_labels=pred_labels[:-1])
