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


def test_labels_to_list_multilabel_returns_nested_list():
    labels = [[0, 1], [1], []]

    assert validation.labels_to_list_multilabel(labels) == labels


@pytest.mark.parametrize("labels", [np.array([[0], [1]]), [0, 1]])
def test_labels_to_list_multilabel_rejects_unsupported_formats(labels):
    with pytest.raises(ValueError):
        validation.labels_to_list_multilabel(labels)
