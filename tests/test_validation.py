# coding: utf-8

import numpy as np
import pandas as pd
import pytest

from cleanlab.internal import validation


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


def test_assert_valid_class_labels_fails_with_str_labels():
    str_labels = np.array(["b", "b", "a", "c", "a"])
    with pytest.raises(ValueError, match="string"):
        validation.assert_valid_class_labels(str_labels)
