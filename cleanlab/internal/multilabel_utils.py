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
Helper functions used internally for multi-label classification tasks.
"""
from typing import Tuple, Optional, List

import numpy as np

from cleanlab.internal.util import get_num_classes


def _is_multilabel(y: np.ndarray) -> bool:
    """Checks whether `y` is in a multi-label indicator matrix format.

    Sparse matrices are not supported.
    """
    if not (isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] > 1):
        return False
    return np.array_equal(np.unique(y), [0, 1])


def stack_complement(pred_prob_slice: np.ndarray) -> np.ndarray:
    """
    Extends predicted probabilities of a single class to two columns.

    Parameters
    ----------
    pred_prob_slice:
        A 1D array with predicted probabilities for a single class.

    Example
    -------
    >>> pred_prob_slice = np.array([0.1, 0.9, 0.3, 0.8])
    >>> stack_complement(pred_prob_slice)
    array([[0.9, 0.1],
            [0.1, 0.9],
            [0.7, 0.3],
            [0.2, 0.8]])
    """
    return np.vstack((1 - pred_prob_slice, pred_prob_slice)).T


def get_onehot_num_classes(
    labels: list, pred_probs: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, int]:
    """Returns OneHot encoding of MultiLabel Data, and number of classes"""
    num_classes = get_num_classes(labels=labels, pred_probs=pred_probs)
    try:
        y_one = int2onehot(labels, K=num_classes)
    except TypeError:
        raise ValueError(
            "wrong format for labels, should be a list of list[indices], please check the documentation in find_label_issues for further information"
        )
    return y_one, num_classes


def int2onehot(labels: list, K: int) -> np.ndarray:
    """Convert multi-label classification `labels` from a ``List[List[int]]`` format to a onehot matrix.
    This returns a binarized format of the labels as a multi-hot vector for each example, where the entries in this vector are 1 for each class that applies to this example and 0 otherwise.

    Parameters
    ----------
    labels: list of lists of integers
      e.g. [[0,1], [3], [1,2,3], [1], [2]]
      All integers from 0,1,...,K-1 must be represented.
    K: int
      The number of classes."""

    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer(classes=range(K))
    return mlb.fit_transform(labels)


def onehot2int(onehot_matrix: np.ndarray) -> List[List[int]]:
    """Convert multi-label classification `labels` from a onehot matrix format to a ``List[List[int]]`` format that can be used with other cleanlab functions.

    Parameters
    ----------
    onehot_matrix: 2D np.ndarray of 0s and 1s
      A matrix representation of multi-label classification labels in a binarized format as a multi-hot vector for each example.
      The entries in this vector are 1 for each class that applies to this example and 0 otherwise.

    Returns
    -------
    labels: list of lists of integers
      e.g. [[0,1], [3], [1,2,3], [1], [2]]
      All integers from 0,1,...,K-1 must be represented."""

    return [list(np.where(row == 1)[0]) for row in onehot_matrix]
