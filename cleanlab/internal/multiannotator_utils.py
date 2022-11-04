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
Helper methods used internally in cleanlab.multiannotator
"""

from cleanlab.typing import LabelLike
from typing import Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from cleanlab.internal.validation import assert_valid_class_labels


def assert_valid_inputs_multiannotator(
    labels_multiannotator: pd.DataFrame,
    pred_probs: Optional[np.ndarray] = None,
) -> None:
    """Validate format of multi-annotator labels"""
    # Raise error if labels are not formatted properly
    if any([isinstance(label, str) for label in labels_multiannotator.values.ravel()]):
        raise ValueError(
            "Labels cannot be strings, they must be zero-indexed integers corresponding to class indices."
        )

    all_labels_flatten = labels_multiannotator.replace({pd.NA: np.NaN}).astype(float).values.ravel()
    all_labels_flatten = all_labels_flatten[~np.isnan(all_labels_flatten)]
    assert_valid_class_labels(all_labels_flatten, allow_one_class=True)

    # Raise error if number of classes in labels_multiannoator does not match number of classes in pred_probs
    if pred_probs is not None:
        num_classes = pred_probs.shape[1]
        highest_class = (
            np.nanmax(labels_multiannotator.replace({pd.NA: np.NaN}).astype(float).values) + 1
        )

        # this allows for missing labels, but not missing columns in pred_probs
        if num_classes < highest_class:
            raise ValueError(
                f"""pred_probs must have at least {int(highest_class)} columns based on the largest class label which appears in labels_multiannotator.
            Perhaps some rarely-annotated classes were lost while establishing consensus labels used to train your classifier."""
            )

    # Raise error if labels_multiannotator has NaN rows
    if labels_multiannotator.isna().all(axis=1).any():
        raise ValueError("labels_multiannotator cannot have rows with all NaN.")

    # Raise error if labels_multiannotator has NaN columns
    if labels_multiannotator.isna().all().any():
        nan_columns = list(
            labels_multiannotator.columns[labels_multiannotator.isna().all() == True]
        )
        raise ValueError(
            f"""labels_multiannotator cannot have columns with all NaN.
        Annotators {nan_columns} did not label any examples."""
        )

    # Raise error if labels_multiannotator has <= 1 column
    if len(labels_multiannotator.columns) <= 1:
        raise ValueError(
            """labels_multiannotator must have more than one column. 
        If there is only one annotator, use cleanlab.rank.get_label_quality_scores instead"""
        )

    # Raise error if labels_multiannotator only has 1 label per example
    if labels_multiannotator.apply(lambda s: len(s.dropna()) == 1, axis=1).all():
        raise ValueError(
            """Each example only has one label, collapse the labels into a 1-D array and use
        cleanlab.rank.get_label_quality_scores instead"""
        )

    # Raise warning if no examples with 2 or more annotators agree
    # TODO: might shift this later in the code to avoid extra compute
    if labels_multiannotator.apply(
        lambda s: np.array_equal(s.dropna().unique(), s.dropna()), axis=1
    ).all():
        warnings.warn("Annotators do not agree on any example. Check input data.")


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
    except (TypeError):  # np.unique / np.sort cannot handle string values or pd.NA types
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
