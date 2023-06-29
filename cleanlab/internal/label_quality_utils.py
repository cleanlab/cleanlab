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

"""Helper methods used internally for computing label quality scores."""
import warnings
import numpy as np
from typing import Optional
from scipy.special import xlogy

from cleanlab.count import get_confident_thresholds


def _subtract_confident_thresholds(
    labels: Optional[np.ndarray],
    pred_probs: np.ndarray,
    multi_label: bool = False,
    confident_thresholds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return adjusted predicted probabilities by subtracting the class confident thresholds and renormalizing.

    The confident class threshold for a class j is the expected (average) "self-confidence" for class j.
    The purpose of this adjustment is to handle class imbalance.

    Parameters
    ----------
    labels : np.ndarray
      Labels in the same format expected by the `cleanlab.count.get_confident_thresholds()` method.
      If labels is None, confident_thresholds needs to be passed in as it will not be calculated.
    pred_probs : np.ndarray (shape (N, K))
      Predicted-probabilities in the same format expected by the `cleanlab.count.get_confident_thresholds()` method.
    confident_thresholds : np.ndarray (shape (K,))
      Pre-calculated confident thresholds. If passed in, function will subtract these thresholds instead of calculating
      confident_thresholds from the given labels and pred_probs.
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
    pred_probs_adj : np.ndarray (float)
      Adjusted pred_probs.
    """
    # Get expected (average) self-confidence for each class
    # TODO: Test this for multi-label
    if confident_thresholds is None:
        if labels is None:
            raise ValueError(
                "Cannot calculate confident_thresholds without labels. Pass in either labels or already calculated "
                "confident_thresholds parameter. "
            )
        confident_thresholds = get_confident_thresholds(labels, pred_probs, multi_label=multi_label)

    # Subtract the class confident thresholds
    pred_probs_adj = pred_probs - confident_thresholds

    # Re-normalize by shifting data to take care of negative values from the subtraction
    pred_probs_adj += confident_thresholds.max()
    pred_probs_adj /= pred_probs_adj.sum(axis=1, keepdims=True)

    return pred_probs_adj


def get_normalized_entropy(
    pred_probs: np.ndarray, min_allowed_prob: Optional[float] = None
) -> np.ndarray:
    """Return the normalized entropy of pred_probs.

    Normalized entropy is between 0 and 1. Higher values of entropy indicate higher uncertainty in the model's prediction of the correct label.

    Read more about normalized entropy `on Wikipedia <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.

    Normalized entropy is used in active learning for uncertainty sampling: https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b

    Unlike label-quality scores, entropy only depends on the model's predictions, not the given label.

    Parameters
    ----------
    pred_probs : np.ndarray (shape (N, K))
      Each row of this matrix corresponds to an example x and contains the model-predicted
      probabilities that x belongs to each possible class: P(label=k|x)

    min_allowed_prob : float, default: None, deprecated
      Minimum allowed probability value. If not `None` (default),
      entries of `pred_probs` below this value will be clipped to this value.

      .. deprecated:: 2.5.0
         This keyword is deprecated and should be left to the default.
         The entropy is well-behaved even if `pred_probs` contains zeros,
         clipping is unnecessary and (slightly) changes the results.

    Returns
    -------
    entropy : np.ndarray (shape (N, ))
      Each element is the normalized entropy of the corresponding row of ``pred_probs``.

    Raises
    ------
    ValueError
        An error is raised if any of the probabilities is not in the interval [0, 1].
    """
    if np.any(pred_probs < 0) or np.any(pred_probs > 1):
        raise ValueError("All probabilities are required to be in the interval [0, 1].")
    num_classes = pred_probs.shape[1]

    if min_allowed_prob is not None:
        warnings.warn(
            "Using `min_allowed_prob` is not necessary anymore and will be removed.",
            DeprecationWarning,
        )
        pred_probs = np.clip(pred_probs, a_min=min_allowed_prob, a_max=None)

    # Note that dividing by log(num_classes) changes the base of the log which rescales entropy to 0-1 range
    return -np.sum(xlogy(pred_probs, pred_probs), axis=1) / np.log(num_classes)
