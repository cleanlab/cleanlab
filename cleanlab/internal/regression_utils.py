"""
Helper function internally used in cleanlab.regression
"""

import numpy as np


def assert_valid_inputs(
    labels: np.ndarray,
    predictions: np.ndarray,
    method: str,
) -> None:
    """Checks that ``labels``, ``predictions``, ``method`` are correctly formatted."""

    # Check if labels and pred_labels are np.ndarray
    if not isinstance(labels, np.ndarray) or not isinstance(predictions, np.ndarray):
        raise TypeError("labels and predictions must be of type np.ndarray")

    # Check if labels and predictions are of same shape
    assert (
        labels.shape == predictions.shape
    ), f"shape of label {labels.shape} and predicted labels {predictions.shape} are not same."

    # Check if method is among allowed scoring method
    scoring_methods = ["residual", "outre"]
    if method not in scoring_methods:
        raise ValueError(
            f"Passed method is not among allowed method. Expected either of {scoring_methods}, got {method}"
        )


def check_dimensions(labels: np.ndarray, predictions: np.ndarray) -> None:
    if labels.ndim != 1:
        raise ValueError(
            f"labels have dimensions {labels.ndim}, Expected 1-D array as input for labels"
        )
    if predictions.ndim != 1:
        raise ValueError(
            f"predictions have dimensions {labels.ndim}, Expected 1-D array as input for predictions"
        )
