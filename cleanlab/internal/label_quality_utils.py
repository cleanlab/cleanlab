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

"""Helper functions for computing label quality scores"""

import numpy as np
import pandas as pd
from cleanlab import rank
from cleanlab.count import get_confident_thresholds


def _subtract_confident_thresholds(
    labels: np.array,
    pred_probs: np.array,
    multi_label: bool = False,
) -> np.array:
    """Returns adjusted predicted probabilities by subtracting the class confident thresholds and renormalizing.

    The confident class threshold for a class j is the expected (average) "self-confidence" for class j.

    The purpose of this adjustment is to handle class imbalance.

    Parameters
    ----------
    labels : np.array
      Labels in the same format expected by the `cleanlab.count.get_confident_thresholds()` method.

    pred_probs : np.array (shape (N, K))
      Predicted-probabilities in the same format expected by the `cleanlab.count.get_confident_thresholds()` method.

    multi_label : bool, optional
      If ``True``, labels should be an iterable (e.g. list) of iterables, containing a
      list of labels for each example, instead of just a single label.
      The multi-label setting supports classification tasks where an example has 1 or more labels.
      Example of a multi-labeled `labels` input: ``[[0,1], [1], [0,2], [0,1,2], [0], [1], ...]``.
      The major difference in how this is calibrated versus single-label is that
      the total number of errors considered is based on the number of labels,
      not the number of examples. So, the calibrated `confident_joint` will sum
      to the number of total labels.

    Returns
    -------
    pred_probs_adj : np.array (float)
      Adjusted pred_probs.
    """

    # Get expected (average) self-confidence for each class
    # TODO: Test this for multi-label
    confident_thresholds = get_confident_thresholds(labels, pred_probs, multi_label=multi_label)

    # Subtract the class confident thresholds
    pred_probs_adj = pred_probs - confident_thresholds

    # Re-normalize by shifting data to take care of negative values from the subtraction
    pred_probs_adj += confident_thresholds.max()
    pred_probs_adj /= pred_probs_adj.sum(axis=1)[
        :, None
    ]  # The [:, None] adds a dimension to make the /= operator work for broadcasting.

    return pred_probs_adj


def get_normalized_entropy(pred_probs: np.array, min_allowed_prob=1e-6) -> np.array:
    """Returns the normalized entropy of pred_probs.

    Normalized entropy is between 0 and 1. Higher values of entropy indicate higher uncertainty in the model's prediction of the correct label.

    Read more about normalized entropy `on Wikipedia <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.

    Normalized entropy is used in active learning for uncertainty sampling: https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b

    Unlike label-quality scores, entropy only depends on the model's predictions, not the given label.

    Parameters
    ----------
    pred_probs : np.array (shape (N, K))
      P(label=k|x) is a matrix with K model-predicted probabilities.
      Each row of this matrix corresponds to an example x and contains the model-predicted
      probabilities that x belongs to each possible class.
      The columns must be ordered such that these probabilities correspond to class 0,1,2,...
      `pred_probs` should have been computed using 3 (or higher) fold cross-validation.

    min_allowed_prob : float, default=1e-6
      Minimum allowed probability value. Entries of `pred_probs` below this value will be clipped to this value.
      Ensures entropy remains well-behaved even when `pred_probs` contains zeros.
    Returns
    -------
    entropy : np.array (float)

    """

    num_classes = pred_probs.shape[1]

    # Note that dividing by log(num_classes) changes the base of the log which rescales entropy to 0-1 range
    clipped_pred_probs = np.clip(pred_probs, a_min=min_allowed_prob, a_max=None)
    return -np.sum(pred_probs * np.log(clipped_pred_probs), axis=1) / np.log(num_classes)


def _get_label_quality_scores_with_NA(
    labels: pd.Series,
    pred_probs: np.array,
    kwargs: dict = {},
) -> pd.Series:
    """Returns label quality scores for each datapoint.
    Same functionality as ``get_label_quality_scores`` with additional support for
    datasets with NaN values.

    For more info about parameters and returns, see the docstring of :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.
    """
    label_quality_scores_subset = rank.get_label_quality_scores(
        labels=labels[pd.notna(labels)].astype("int32"),
        pred_probs=pred_probs[pd.notna(labels)],
        **kwargs,
    )
    label_quality_scores = pd.Series(np.nan, index=np.arange(len(labels)))
    label_quality_scores[pd.notna(labels)] = label_quality_scores_subset
    return label_quality_scores
