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


def get_confident_thresholds(labels: np.array, pred_probs: np.array) -> np.array:
    """Returns expected (average) "self-confidence" for each class.

    The confident class threshold for a class j is the expected (average) "self-confidence" for class j.

    Parameters
    ----------
    labels : np.array
      A discrete vector of noisy labels, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    pred_probs : np.array (shape (N, K))
      P(label=k|x) is a matrix with K model-predicted probabilities.
      Each row of this matrix corresponds to an example x and contains the model-predicted
      probabilities that x belongs to each possible class.
      The columns must be ordered such that these probabilities correspond to class 0,1,2,...
      `pred_probs` should have been computed using 3 (or higher) fold cross-validation.

    Returns
    -------
    confident_thresholds : np.array (shape (K,))

    """
    confident_thresholds = np.array(
        [np.mean(pred_probs[:, k][labels == k]) for k in range(pred_probs.shape[1])]
    )
    return confident_thresholds


def subtract_confident_thresholds(labels: np.array, pred_probs: np.array) -> np.array:
    """Returns adjusted predicted probabilities by subtracting the class confident thresholds and renormalizing.

    The confident class threshold for a class j is the expected (average) "self-confidence" for class j.

    The purpose of this adjustment is to handle class imbalance.

    Parameters
    ----------
    labels : np.array
      Labels in the same format expected by the `get_confident_thresholds()` method.

    pred_probs : np.array (shape (N, K))
      Predicted-probabilities in the same format expected by the `get_confident_thresholds()` method.

    Returns
    -------
    pred_probs_adj : np.array (float)
      Adjusted pred_probs.
    """

    # Get expected (average) self-confidence for each class
    confident_thresholds = get_confident_thresholds(labels, pred_probs)

    # Subtract the class confident thresholds
    pred_probs_adj = pred_probs - confident_thresholds

    # Renormalize by shifting data to take care of negative values from the subtraction
    pred_probs_adj += 1
    pred_probs_adj /= pred_probs_adj.sum(axis=1)[:, None]

    return pred_probs_adj
