"""
Checks to ensure valid inputs for various methods.
"""

from cleanlab.typing import LabelLike, DatasetLike
from cleanlab.internal.constants import FLOATING_POINT_COMPARISON
from typing import Any, List, Optional, Union
import warnings
import numpy as np
import pandas as pd


def assert_valid_inputs(
    X: DatasetLike,
    y: LabelLike,
    pred_probs: Optional[np.ndarray] = None,
    multi_label: bool = False,
    allow_missing_classes: bool = True,
    allow_one_class: bool = False,
) -> None:
    """Checks that ``X``, ``labels``, ``pred_probs`` are correctly formatted."""
    y = _validate_labels(y, multi_label, allow_missing_classes, allow_one_class)

    if pred_probs is None:
        _validate_features_if_needed(X, y)
    else:
        _validate_pred_probs(pred_probs, y, multi_label, X)


def _validate_labels(
    y: LabelLike,
    multi_label: bool,
    allow_missing_classes: bool,
    allow_one_class: bool,
):
    if not isinstance(y, (list, np.ndarray, np.generic, pd.Series, pd.DataFrame)):
        raise TypeError("labels should be a numpy array or pandas Series.")
    if not multi_label:
        y = labels_to_array(y)
        assert_valid_class_labels(
            y=y, allow_missing_classes=allow_missing_classes, allow_one_class=allow_one_class
        )
    return y


def _get_num_examples(X: DatasetLike) -> int:
    try:
        return len(X)
    except Exception:
        try:
            return X.shape[0]
        except Exception as exc:
            raise TypeError("Data features X must support either: len(X) or X.shape[0]") from exc


def _validate_features_if_needed(X: DatasetLike, y) -> None:
    assert_nonempty_input(X)
    num_examples = _get_num_examples(X)
    if num_examples != len(y):
        raise ValueError(
            f"X and labels must be same length, but X is length {num_examples} and labels is length {len(y)}."
        )

    assert_indexing_works(X, length_X=num_examples)


def _validate_pred_probs(pred_probs, y, multi_label: bool, X: DatasetLike) -> None:
    if not isinstance(pred_probs, (np.ndarray, np.generic)):
        raise TypeError("pred_probs must be a numpy array.")
    if len(pred_probs) != len(y):
        raise ValueError("pred_probs and labels must have same length.")
    if len(pred_probs.shape) != 2:
        raise ValueError("pred_probs array must have shape: num_examples x num_classes.")
    if not multi_label:
        assert isinstance(y, np.ndarray)
        highest_class = max(y) + 1
    else:
        assert isinstance(y, list)
        assert all(isinstance(y_i, list) for y_i in y)
        highest_class = max([max(y_i) for y_i in y if len(y_i) != 0]) + 1
    if pred_probs.shape[1] < highest_class:
        raise ValueError(
            f"pred_probs must have at least {highest_class} columns, based on the largest class index which appears in labels."
        )
    # Check for valid probabilities.
    if (np.min(pred_probs) < 0 - FLOATING_POINT_COMPARISON) or (
        np.max(pred_probs) > 1 + FLOATING_POINT_COMPARISON
    ):
        raise ValueError("Values in pred_probs must be between 0 and 1.")
    if X is not None:
        warnings.warn("When X and pred_probs are both provided, the former may be ignored.")


def assert_valid_class_labels(
    y: np.ndarray,
    allow_missing_classes: bool = True,
    allow_one_class: bool = False,
) -> None:
    """Checks that ``labels`` is properly formatted, i.e. a 1D numpy array where labels are zero-based
    integers (not multi-label).
    """
    if y.ndim != 1:
        raise ValueError("Labels must be 1D numpy array.")
    if any([isinstance(label, str) for label in y]):
        raise ValueError(
            "Labels cannot be strings, they must be zero-indexed integers corresponding to class indices."
        )
    if not np.equal(np.mod(y, 1), 0).all():  # check that labels are integers
        raise ValueError("Labels must be zero-indexed integers corresponding to class indices.")
    if min(y) < 0:
        raise ValueError("Labels must be positive integers corresponding to class indices.")

    unique_classes = np.unique(y)
    if (not allow_one_class) and (len(unique_classes) < 2):
        raise ValueError("Labels must contain at least 2 classes.")

    if not allow_missing_classes:
        if (unique_classes != np.arange(len(unique_classes))).any():
            msg = "cleanlab requires zero-indexed integer labels (0,1,2,..,K-1), but in "
            msg += "your case: np.unique(labels) = {}. ".format(str(unique_classes))
            msg += "Every class in (0,1,2,..,K-1) must be present in labels as well."
            raise TypeError(msg)


def assert_nonempty_input(X: Any) -> None:
    """Ensures input is not None."""
    if X is None:
        raise ValueError("Data features X cannot be None. Currently X is None.")


def assert_indexing_works(
    X: DatasetLike, idx: Optional[List[int]] = None, length_X: Optional[int] = None
) -> None:
    """Ensures we can do list-based indexing into ``X`` and ``y``.
    ``length_X`` is an optional argument since sparse matrix ``X``
    does not support: ``len(X)`` and we want this method to work for sparse ``X``
    (in addition to many other types of ``X``).
    """
    if idx is None:
        if length_X is None:
            length_X = 2  # pragma: no cover

        idx = [0, length_X - 1]

    is_indexed = False
    try:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            _ = X.iloc[idx]  # type: ignore[call-overload]
            is_indexed = True
    except Exception:
        pass
    if not is_indexed:
        try:  # check if X is pytorch Dataset object using lazy import
            import torch

            if isinstance(X, torch.utils.data.Dataset):  # indexing for pytorch Dataset
                _ = torch.utils.data.Subset(X, idx)  # type: ignore[call-overload]
                is_indexed = True
        except Exception:
            pass
    if not is_indexed:
        try:
            _ = X[idx]  # type: ignore[call-overload]
        except Exception:
            msg = (
                "Data features X must support list-based indexing; i.e. one of these must work: \n"
            )
            msg += "1)  X[index_list] where say index_list = [0,1,3,10], or \n"
            msg += "2)  X.iloc[index_list] if X is pandas DataFrame."
            raise TypeError(msg)


def labels_to_array(y: Union[LabelLike, np.generic]) -> np.ndarray:
    """Converts different types of label objects to 1D numpy array and checks their validity.

    Parameters
    ----------
    y : Union[LabelLike, np.generic]
        Labels to convert to 1D numpy array. Can be a list, numpy array, pandas Series, or pandas DataFrame.

    Returns
    -------
    labels_array : np.ndarray
        1D numpy array of labels.
    """
    if isinstance(y, pd.Series):
        y_series: np.ndarray = y.to_numpy()
        return y_series
    elif isinstance(y, pd.DataFrame):
        y_arr = y.values
        assert isinstance(y_arr, np.ndarray)
        if y_arr.shape[1] != 1:
            raise ValueError("labels must be one dimensional.")
        return y_arr.flatten()
    else:  # y is list, np.ndarray, or some other tuple-like object
        try:
            return np.asarray(y)
        except:
            raise ValueError(
                "List of labels must be convertable to 1D numpy array via: np.ndarray(labels)."
            )


def labels_to_list_multilabel(y: List) -> List[List[int]]:
    """Converts different types of label objects to nested list and checks their validity.

    Parameters
    ----------
    y : List
        Labels to convert to nested list. Supports only list type.

    Returns
    -------
    labels_list : List[List[int]]
        Nested list of labels.
    """
    if not isinstance(y, list):
        raise ValueError("Unsupported Label format")
    if not all(isinstance(x, list) for x in y):
        raise ValueError("Each element in list of labels must be a list.")

    return y
