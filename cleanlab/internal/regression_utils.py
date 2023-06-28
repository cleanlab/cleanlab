# Copyright (C) 2017-2022  Cleanlab Inc.
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


"""
Helper functions internally used in cleanlab.regression.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from typing import Tuple, Union


def assert_valid_prediction_inputs(
    labels: ArrayLike,
    predictions: ArrayLike,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Checks that ``labels``, ``predictions``, ``method`` are correctly formatted."""

    # Load array_like input as numpy array. If not raise error.
    try:
        labels = np.asarray(labels)
    except:
        raise ValueError(f"labels must be array_like.")

    try:
        predictions = np.asarray(predictions)
    except:
        raise ValueError(f"predictions must be array_like.")

    # Check if labels and predictions are 1-D and numeric
    valid_labels = check_dimension_and_datatype(check_input=labels, text="labels")
    valid_predictions = check_dimension_and_datatype(check_input=predictions, text="predictions")

    # Check if number of examples are same.
    assert (
        valid_labels.shape == valid_predictions.shape
    ), f"Number of examples in labels {labels.shape} and predictions {predictions.shape} are not same."

    # Check if inputs have missing values
    check_missing_values(valid_labels, text="labels")
    check_missing_values(valid_predictions, text="predictions")

    # Check if method is among allowed scoring method
    scoring_methods = ["residual", "outre"]
    if method not in scoring_methods:
        raise ValueError(f"Specified method '{method}' must be one of: {scoring_methods}.")

    # return 1-D numpy array
    return valid_labels, valid_predictions


def assert_valid_regression_inputs(
    X: Union[np.ndarray, pd.DataFrame],
    y: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Checks that regression inputs are properly formatted and returns the inputs in numpy array format.
    """
    try:
        X = np.asarray(X)
    except:
        raise ValueError(f"X must be array_like.")

    y = check_dimension_and_datatype(y, "y")
    check_missing_values(y, text="y")

    if len(X) != len(y):
        raise ValueError("X and y must have same length.")

    return X, y


def check_dimension_and_datatype(check_input: ArrayLike, text: str) -> np.ndarray:
    """
    Raises errors related to:
    1. If input is empty
    2. If input is not 1-D
    3. If input is not numeric

    If all the checks are passed, it returns the squeezed 1-D array required by the main algorithm.
    """

    try:
        check_input = np.asarray(check_input)
    except:
        raise ValueError(f"{text} could not be converted to numpy array, check input.")

    # Check if input is empty
    if not check_input.size:
        raise ValueError(f"{text} cannot be empty array.")

    # Remove axis with length one
    check_input = np.squeeze(check_input)

    # Check if input is 1-D
    if check_input.ndim != 1:
        raise ValueError(
            f"Expected 1-Dimensional inputs for {text}, got {check_input.ndim} dimensions."
        )

    # Check if datatype is numeric
    if not np.issubdtype(check_input.dtype, np.number):
        raise ValueError(
            f"Expected {text} to contain numeric values, got values of type {check_input.dtype}."
        )

    return check_input


def check_missing_values(check_input: np.ndarray, text: str):
    """Raise error if there are any missing values in Numpy array."""

    if np.isnan(check_input).any():
        raise ValueError(f"{text} cannot contain missing values.")
