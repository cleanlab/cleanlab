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
Helper functions used internally for multi-label classification tasks.
"""

import numpy as np

from cleanlab.internal.util import get_num_classes, int2onehot


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


def get_onehot_num_classes(labels, pred_probs=None):
    """Returns OneHot encoding of MultiLabel Data, and number of classes"""
    num_classes = get_num_classes(labels=labels, pred_probs=pred_probs)
    try:
        y_one = int2onehot(labels, K=num_classes)
    except TypeError:
        raise ValueError(
            "wrong format for labels, should be a list of list[indices], please check the documentation in find_label_issues for further information"
        )
    return y_one, num_classes
