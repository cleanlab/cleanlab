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


def test_labels_to_list_multilabel_from_numpy_object_array() -> None:
    """Tests conversion of a jagged numpy object array to a list of lists.

    This test directly covers the `isinstance(y, np.ndarray)` branch in
    the `labels_to_list_multilabel` function.
    """
    # Multi-label data often results in jagged arrays, which NumPy handles
    # using dtype=object. This is the exact use case the code is for.
    y_numpy: "np.ndarray[np.ndarray[int]]" = np.array([[0], [1, 2], []], dtype=object)
    result: list[list[int]] = validation.labels_to_list_multilabel(y_numpy)
    expected: list[list[int]] = [[0], [1, 2], []]
    assert result == expected, "The converted list does not match the expected output."
    assert isinstance(result, list), f"Expected result to be a list, but got {type(result)}."
    assert all(
        isinstance(i, list) for i in result
    ), "Expected all elements in the result to be lists."


def test_labels_to_list_multilabel_from_numpy_regular_array() -> None:
    """Tests conversion of a standard rectangular numpy array."""
    y_numpy: np.ndarray = np.array([[1, 2], [3, 4]])
    result: list[list[int]] = validation.labels_to_list_multilabel(y_numpy)
    expected: list[list[int]] = [[1, 2], [3, 4]]
    assert result == expected, "The converted list does not match the expected output."
    assert isinstance(result, list), f"Expected result to be a list, but got {type(result)}."


def test_labels_to_list_multilabel_error_on_bad_type() -> None:
    """Tests that a non-list/non-numpy input raises a ValueError."""
    y_tuple: tuple[list[int], ...] = ([[0], [1]],)  # A tuple of lists
    with pytest.raises(ValueError, match="Unsupported Label format"):
        validation.labels_to_list_multilabel(y_tuple)


def test_labels_to_list_multilabel_error_on_non_nested_list() -> None:
    """Tests that a flat list (not a list of lists) raises a ValueError."""
    y_flat_list: list[int] = [0, 1, 2]
    with pytest.raises(ValueError, match="Each element in list of labels must be a list"):
        validation.labels_to_list_multilabel(y_flat_list)
