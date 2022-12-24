"""
Helper function internally used in cleanlab.regression
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from cleanlab.typing import LabelLike
from typing import Optional


def assert_valid_inputs(
    labels: Optional[LabelLike],
    predictions: Optional[LabelLike],
    method: str,
) -> None:
    """Checks that ``labels``, ``predictions``, ``method`` are correctly formatted."""

    supported_types = (list, np.ndarray, pd.Series, pd.DataFrame)

    # Check if labels and predictions are of supported types
    if not isinstance(labels, supported_types) and not isinstance(predictions, supported_types):
        raise TypeError(
            f"Expected labels and predictions to be either of {supported_types}, Got labels of type {type(labels)}, and predictions of type {type(predictions)}",
        )

    # check if labels and predictions are 1-D and numeric
    check_dimension_and_datatype(check_input=labels, text = "labels")
    check_dimension_and_datatype(check_input=predictions, text = "predictions")

    # check if number of examples are same.
    assert len(labels) == len(
        predictions
    ), f"Length of labels {len(labels)} and predictions {len(predictions)} are not same."

    # Check if method is among allowed scoring method
    scoring_methods = ["residual", "outre"]
    if method not in scoring_methods:
        raise ValueError(
            f"Passed method is not among allowed method. Expected either of {scoring_methods}, got {method}"
        )


def check_dimension_and_datatype(check_input: Optional[LabelLike], text : str):
    # check if input is empty
    if not len(check_input):
        raise ValueError(
            f"{text} is Empty, check input."
        )

    if isinstance(check_input, list):
        if isinstance(check_input[0], list):
            raise ValueError(f"{text} must be 1-D. List of List is not supported.")
        elif not all(isinstance(x, (int, float)) for x in check_input):
            raise ValueError(
                f"All element of {text} must be of type numeric i.e., integer or float"
            )

    elif isinstance(check_input, pd.DataFrame):
        if check_input.shape[1] != 1:
            raise ValueError(
                f"{text} must be 1-D. For DataFrame, second dimension must be 1, got {check_input.shape}."
            )
        elif check_input.shape[1] == 1:
            if not is_numeric_dtype(check_input):
                raise ValueError(f"{text} must be 1-D and numeric type. got {check_input.dtype}.")
    elif isinstance(check_input, (np.ndarray, pd.Series)):
        if len(check_input.shape) != 1:
            raise ValueError(f"{text} must be 1-D {type(check_input)}, got {check_input.shape}")
        elif len(check_input.shape) == 1:
            if isinstance(check_input, pd.Series) and not is_numeric_dtype(check_input):
                raise ValueError(f"{text} must be 1-D and numeric type. got {check_input.dtype}.")
            elif isinstance(check_input, np.ndarray):
                if not all(isinstance(x, (int, float)) for x in check_input.tolist()):
                    raise ValueError(f"{text} must be 1-d and numeric type i.e., integer or float.")


def check_dimensions(labels: np.ndarray, predictions: np.ndarray) -> None:
    if labels.ndim != 1:
        raise ValueError(
            f"labels have dimensions {labels.ndim}, Expected 1-D array as input for labels"
        )
    if predictions.ndim != 1:
        raise ValueError(
            f"predictions have dimensions {labels.ndim}, Expected 1-D array as input for predictions"
        )
