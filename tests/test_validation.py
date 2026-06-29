# coding: utf-8

from cleanlab.internal import validation
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("y_list", [["a", "b", "a"], [0, 1, 2]])
@pytest.mark.parametrize("format", [list, np.array, pd.Series, pd.DataFrame])
def test_labels_to_array_return_types(y_list, format):
    y = format(y_list)
    labels = validation.labels_to_array(y)
    assert isinstance(labels, np.ndarray)


@pytest.mark.parametrize("y_list", [["a", "b", "a"], [0, 1, 2]])
@pytest.mark.parametrize("format", [list, np.array, pd.Series])
def test_labels_to_array_return_values(y_list, format):
    y = format(y_list)
    labels = validation.labels_to_array(y)
    assert np.array_equal(y, labels)


def test_label_to_array_raises_error():
    # Pandas DataFrame should have only one column
    y = pd.DataFrame({"a": [0, 1], "b": [2, 3]})
    with pytest.raises(ValueError):
        validation.labels_to_array(y)


def test_assert_valid_inputs_accepts_valid_inputs_and_warns():
    X = np.zeros((3, 2))
    y = np.array([0, 1, 0])
    pred_probs = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])

    with pytest.warns(UserWarning, match="may be ignored"):
        validation.assert_valid_inputs(X, y, pred_probs=pred_probs)


def test_assert_valid_inputs_ignores_x_when_pred_probs_present():
    X = np.zeros((2, 2))
    y = np.array([0, 1, 0])
    pred_probs = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])

    with pytest.warns(UserWarning, match="may be ignored"):
        validation.assert_valid_inputs(X, y, pred_probs=pred_probs)


def test_assert_valid_inputs_rejects_invalid_labels_type():
    with pytest.raises(TypeError, match="labels should be a numpy array or pandas Series"):
        validation.assert_valid_inputs(np.zeros((2, 2)), y="invalid")


def test_assert_valid_inputs_rejects_length_mismatch():
    X = np.zeros((2, 2))
    y = np.array([0, 1, 0])

    with pytest.raises(ValueError, match="X and labels must be same length"):
        validation.assert_valid_inputs(X, y)


def test_assert_valid_inputs_rejects_invalid_pred_probs_shape():
    X = np.zeros((2, 2))
    y = np.array([0, 1])
    pred_probs = np.array([0.5, 0.5])

    with pytest.raises(ValueError, match="shape: num_examples x num_classes"):
        validation.assert_valid_inputs(X, y, pred_probs=pred_probs)
