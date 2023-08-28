# Copyright (C) 2017-2023  Cleanlab Inc.
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
Helper methods used internally in cleanlab.multiannotator
"""

from cleanlab.typing import LabelLike
from typing import Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from cleanlab.internal.validation import assert_valid_class_labels
from cleanlab.internal.numerics import softmax
from cleanlab.internal.util import get_num_classes, value_counts

SMALL_CONST = 1e-30


def assert_valid_inputs_multiannotator(
    labels_multiannotator: np.ndarray,
    pred_probs: Optional[np.ndarray] = None,
    ensemble: bool = False,
    allow_single_label: bool = False,
    annotator_ids: Optional[pd.Index] = None,
) -> None:
    """Validate format of multi-annotator labels"""
    # Check that labels_multiannotator is a 2D array
    if labels_multiannotator.ndim != 2:
        raise ValueError(
            "labels_multiannotator must be a 2D array or dataframe, "
            "each row represents an example and each column represents an annotator."
        )

    # Raise error if labels are not formatted properly
    if any([isinstance(label, str) for label in labels_multiannotator.ravel()]):
        raise ValueError(
            "Labels cannot be strings, they must be zero-indexed integers corresponding to class indices."
        )

    # Raise error if labels_multiannotator has NaN rows
    nan_row_mask = np.isnan(labels_multiannotator).all(axis=1)
    if nan_row_mask.any():
        nan_rows = list(np.where(nan_row_mask)[0])
        raise ValueError(
            "labels_multiannotator cannot have rows with all NaN, each example must have at least one label.\n"
            f"Examples {nan_rows} do not have any labels."
        )

    # Raise error if labels_multiannotator has NaN columns
    nan_col_mask = np.isnan(labels_multiannotator).all(axis=0)
    if nan_col_mask.any():
        if annotator_ids is not None:
            nan_columns = list(annotator_ids[np.where(nan_col_mask)[0]])
        else:
            nan_columns = list(np.where(nan_col_mask)[0])
        raise ValueError(
            "labels_multiannotator cannot have columns with all NaN, each annotator must annotator at least one example.\n"
            f"Annotators {nan_columns} did not label any examples."
        )

    if not allow_single_label:
        # Raise error if labels_multiannotator has <= 1 column
        if labels_multiannotator.shape[1] <= 1:
            raise ValueError(
                "labels_multiannotator must have more than one column.\n"
                "If there is only one annotator, use cleanlab.rank.get_label_quality_scores instead"
            )

        # Raise error if labels_multiannotator only has 1 label per example
        if (np.sum(~np.isnan(labels_multiannotator), axis=1) == 1).all():
            raise ValueError(
                "Each example only has one label, collapse the labels into a 1-D array and use "
                "cleanlab.rank.get_label_quality_scores instead"
            )

        # Raise warning if no examples with 2 or more annotators agree
        # TODO: might shift this later in the code to avoid extra compute
        if np.apply_along_axis(
            lambda s: np.array_equal(np.unique(s[~np.isnan(s)]), s[~np.isnan(s)]),
            axis=1,
            arr=labels_multiannotator,
        ).all():
            warnings.warn("Annotators do not agree on any example. Check input data.")

    # Check labels
    all_labels_flatten = labels_multiannotator.ravel()
    all_labels_flatten = all_labels_flatten[~np.isnan(all_labels_flatten)]
    assert_valid_class_labels(all_labels_flatten, allow_one_class=True)

    # Raise error if number of classes in labels_multiannoator does not match number of classes in pred_probs
    if pred_probs is not None:
        if not isinstance(pred_probs, np.ndarray):
            raise TypeError("pred_probs must be a numpy array.")

        if ensemble:
            if pred_probs.ndim != 3:
                error_message = "pred_probs must be a 3d array."
                if pred_probs.ndim == 2:
                    error_message += " If you have a 2d pred_probs array, use the non-ensemble version of this function."
                raise ValueError(error_message)

            if pred_probs.shape[1] != len(labels_multiannotator):
                raise ValueError("each pred_probs and labels_multiannotator must have same length.")

            num_classes = pred_probs.shape[2]
        else:
            if pred_probs.ndim != 2:
                error_message = "pred_probs must be a 2d array."
                if pred_probs.ndim == 3:
                    error_message += " If you have a 3d pred_probs array, use the ensemble version of this function."
                raise ValueError(error_message)

            if len(pred_probs) != len(labels_multiannotator):
                raise ValueError("pred_probs and labels_multiannotator must have same length.")

            num_classes = pred_probs.shape[1]

        highest_class = np.nanmax(labels_multiannotator) + 1

        # this allows for missing labels, but not missing columns in pred_probs
        if num_classes < highest_class:
            raise ValueError(
                f"pred_probs must have at least {int(highest_class)} columns based on the largest class label "
                "which appears in labels_multiannotator. Perhaps some rarely-annotated classes were lost while "
                "establishing consensus labels used to train your classifier."
            )


def assert_valid_pred_probs(
    pred_probs: Optional[np.ndarray] = None,
    pred_probs_unlabeled: Optional[np.ndarray] = None,
    ensemble: bool = False,
):
    """Validate format of pred_probs for multiannotator active learning functions"""
    if pred_probs is None and pred_probs_unlabeled is None:
        raise ValueError(
            "pred_probs and pred_probs_unlabeled cannot both be None, specify at least one of the two."
        )

    if ensemble:
        if pred_probs is not None:
            if not isinstance(pred_probs, np.ndarray):
                raise TypeError("pred_probs must be a numpy array.")
            if pred_probs.ndim != 3:
                error_message = "pred_probs must be a 3d array."
                if pred_probs.ndim == 2:  # pragma: no cover
                    error_message += " If you have a 2d pred_probs array (ie. only one predictor), use the non-ensemble version of this function."
                raise ValueError(error_message)

        if pred_probs_unlabeled is not None:
            if not isinstance(pred_probs_unlabeled, np.ndarray):
                raise TypeError("pred_probs_unlabeled must be a numpy array.")
            if pred_probs_unlabeled.ndim != 3:
                error_message = "pred_probs_unlabeled must be a 3d array."
                if pred_probs_unlabeled.ndim == 2:  # pragma: no cover
                    error_message += " If you have a 2d pred_probs_unlabeled array, use the non-ensemble version of this function."
                raise ValueError(error_message)

        if pred_probs is not None and pred_probs_unlabeled is not None:
            if pred_probs.shape[2] != pred_probs_unlabeled.shape[2]:
                raise ValueError(
                    "pred_probs and pred_probs_unlabeled must have the same number of classes"
                )

    else:
        if pred_probs is not None:
            if not isinstance(pred_probs, np.ndarray):
                raise TypeError("pred_probs must be a numpy array.")
            if pred_probs.ndim != 2:
                error_message = "pred_probs must be a 2d array."
                if pred_probs.ndim == 3:  # pragma: no cover
                    error_message += " If you have a 3d pred_probs array, use the ensemble version of this function."
                raise ValueError(error_message)

        if pred_probs_unlabeled is not None:
            if not isinstance(pred_probs_unlabeled, np.ndarray):
                raise TypeError("pred_probs_unlabeled must be a numpy array.")
            if pred_probs_unlabeled.ndim != 2:
                error_message = "pred_probs_unlabeled must be a 2d array."
                if pred_probs_unlabeled.ndim == 3:  # pragma: no cover
                    error_message += " If you have a 3d pred_probs_unlabeled array, use the non-ensemble version of this function."
                raise ValueError(error_message)

        if pred_probs is not None and pred_probs_unlabeled is not None:
            if pred_probs.shape[1] != pred_probs_unlabeled.shape[1]:
                raise ValueError(
                    "pred_probs and pred_probs_unlabeled must have the same number of classes"
                )


def format_multiannotator_labels(labels: LabelLike) -> Tuple[pd.DataFrame, dict]:
    """Takes an array of labels and formats it such that labels are in the set ``0, 1, ..., K-1``,
    where ``K`` is the number of classes. The labels are assigned based on lexicographic order.

    Returns
    -------
    formatted_labels
        Returns pd.DataFrame of shape ``(N,M)``. The return labels will be properly formatted and can be passed to
        cleanlab.multiannotator functions.

    mapping
        A dictionary showing the mapping of new to old labels, such that ``mapping[k]`` returns the name of the k-th class.
    """
    if isinstance(labels, pd.DataFrame):
        np_labels = labels.values
    elif isinstance(labels, np.ndarray):
        np_labels = labels
    else:
        raise TypeError("labels must be 2D numpy array or pandas DataFrame")

    unique_labels = pd.unique(np_labels.ravel())

    try:
        unique_labels = unique_labels[~np.isnan(unique_labels)]
        unique_labels.sort()
    except TypeError:  # np.unique / np.sort cannot handle string values or pd.NA types
        nan_mask = np.array([(l is np.NaN) or (l is pd.NA) or (l == "nan") for l in unique_labels])
        unique_labels = unique_labels[~nan_mask]
        unique_labels.sort()

    # convert float labels (that arose because np.nan is float type) to int
    if unique_labels.dtype == "float":
        unique_labels = unique_labels.astype("int")

    label_map = {label: i for i, label in enumerate(unique_labels)}
    inverse_map = {i: label for label, i in label_map.items()}

    if isinstance(labels, np.ndarray):
        labels = pd.DataFrame(labels)

    formatted_labels = labels.replace(label_map)

    return formatted_labels, inverse_map


def check_consensus_label_classes(
    labels_multiannotator: np.ndarray,
    consensus_label: np.ndarray,
    consensus_method: str,
) -> None:
    """Check if any classes no longer appear in the set of consensus labels (established using the consensus_method stated)"""
    unique_ma_labels = np.unique(labels_multiannotator)
    unique_ma_labels = unique_ma_labels[~np.isnan(unique_ma_labels)]
    labels_set_difference = set(unique_ma_labels) - set(consensus_label)

    if len(labels_set_difference) > 0:
        print(
            "CAUTION: Number of unique classes has been reduced from the original data when establishing consensus labels "
            f"using consensus method '{consensus_method}', likely due to some classes being rarely annotated. "
            "If training a classifier on these consensus labels, it will never see any of the omitted classes unless you "
            "manually replace some of the consensus labels.\n"
            f"Classes in the original data but not in consensus labels: {list(map(int, labels_set_difference))}"
        )


def compute_soft_cross_entropy(
    labels_multiannotator: np.ndarray,
    pred_probs: np.ndarray,
) -> float:
    """Compute soft cross entropy between the annotators' empirical label distribution and model pred_probs"""
    num_classes = get_num_classes(pred_probs=pred_probs)

    empirical_label_distribution = np.full((len(labels_multiannotator), num_classes), np.NaN)
    for i, labels in enumerate(labels_multiannotator):
        labels_subset = labels[~np.isnan(labels)]
        empirical_label_distribution[i, :] = value_counts(
            labels_subset, num_classes=num_classes
        ) / len(labels_subset)

    clipped_pred_probs = np.clip(pred_probs, a_min=SMALL_CONST, a_max=None)
    soft_cross_entropy = -np.sum(
        empirical_label_distribution * np.log(clipped_pred_probs), axis=1
    ) / np.log(num_classes)

    return soft_cross_entropy


def find_best_temp_scaler(
    labels_multiannotator: np.ndarray,
    pred_probs: np.ndarray,
    coarse_search_range: list = [0.1, 0.2, 0.5, 0.8, 1, 2, 3, 5, 8],
    fine_search_size: int = 4,
) -> float:
    """Find the best temperature scaling factor that minimizes the soft cross entropy between the annotators' empirical label distribution
    and model pred_probs"""

    soft_cross_entropy_coarse = np.full(len(coarse_search_range), np.NaN)
    log_pred_probs = np.log(
        pred_probs, where=pred_probs > 0, out=np.full(pred_probs.shape, -np.inf)
    )
    for i, curr_temp in enumerate(coarse_search_range):
        scaled_pred_probs = softmax(log_pred_probs, temperature=curr_temp, axis=1, shift=False)
        soft_cross_entropy_coarse[i] = np.mean(
            compute_soft_cross_entropy(labels_multiannotator, scaled_pred_probs)
        )

    min_entropy_ind = np.argmin(soft_cross_entropy_coarse)
    fine_search_range = _set_fine_search_range(
        coarse_search_range, fine_search_size, min_entropy_ind
    )
    soft_cross_entropy_fine = np.full(len(fine_search_range), np.NaN)
    for i, curr_temp in enumerate(fine_search_range):
        scaled_pred_probs = softmax(log_pred_probs, temperature=curr_temp, axis=1, shift=False)
        soft_cross_entropy_fine[i] = np.mean(
            compute_soft_cross_entropy(labels_multiannotator, scaled_pred_probs)
        )
    best_temp = fine_search_range[np.argmin(soft_cross_entropy_fine)]
    return best_temp


def _set_fine_search_range(
    coarse_search_range: list, fine_search_size: int, min_entropy_ind: np.intp
) -> np.ndarray:
    fine_search_range = np.array([])
    if min_entropy_ind != 0:
        fine_search_range = np.append(
            np.linspace(
                coarse_search_range[min_entropy_ind - 1],
                coarse_search_range[min_entropy_ind],
                fine_search_size,
                endpoint=False,
            ),
            fine_search_range,
        )
    if min_entropy_ind != len(coarse_search_range) - 1:
        fine_search_range = np.append(
            fine_search_range,
            np.linspace(
                coarse_search_range[min_entropy_ind],
                coarse_search_range[min_entropy_ind + 1],
                fine_search_size + 1,
                endpoint=True,
            ),
        )
    return fine_search_range


def temp_scale_pred_probs(
    pred_probs: np.ndarray,
    temp: float,
) -> np.ndarray:
    """Scales pred_probs by the given temperature factor. Temperature of <1 will sharpen the pred_probs while temperatures of >1 will smoothen it."""
    # clip pred_probs to prevent taking log of 0
    pred_probs = np.clip(pred_probs, a_min=SMALL_CONST, a_max=None)
    pred_probs = pred_probs / np.sum(pred_probs, axis=1)[:, np.newaxis]

    # apply temperate scale
    scaled_pred_probs = softmax(np.log(pred_probs), temperature=temp, axis=1, shift=False)
    scaled_pred_probs = (
        scaled_pred_probs / np.sum(scaled_pred_probs, axis=1)[:, np.newaxis]
    )  # normalize

    return scaled_pred_probs
