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

# Baseline methods
# 
# Contains baseline methods for estimating label errors.
#
# These methods ONLY WORK FOR SINGLE_LABEL (not multi-label)

from __future__ import (
    print_function, absolute_import, division, unicode_literals, with_statement
)
from sklearn.metrics import confusion_matrix
from cleanlab.filter import get_noise_indices
from cleanlab.count import calibrate_confident_joint
import numpy as np


def baseline_argmax(psx, labels, multi_label=False):
    """This is the simplest baseline approach. Just consider
    anywhere argmax != s as a label error.

    Parameters
    ----------
    labels : np.array
        A discrete vector of noisy labels, i.e. some labels may be erroneous.

    psx : np.array (shape (N, K))
        P(label=k|x) is a matrix with K (noisy) probabilities for each of the
        N examples x. This is the probability distribution over all K classes,
        for each example, regarding whether the example has label s==k P(s=k|x).
        psx should have been computed using 3 (or higher) fold cross-validation.

    multi_label : bool
        Set to True if s is multi-label (list of lists, or np.array of np.array)

    Returns
    -------
        A boolean mask that is true if the example belong
        to that index is label error."""
    
    if multi_label:
        return np.array([np.argsort(psx[i]) == j for i, j in enumerate(labels)])
    return np.argmax(psx, axis=1) != np.asarray(labels)


def baseline_argmax_confusion_matrix(
    psx,
    labels,
    calibrate=True,
    prune_method='prune_by_noise_rate',
):
    """This is a baseline approach. That uses the confusion matrix
    of argmax(psx) and s as the confident joint and then uses cleanlab
    (confident learning) to find the label errors using this matrix.

    This method does not support multi-label labels.

    Parameters
    ----------

    labels : np.array
        A discrete vector of noisy labels, i.e. some labels may be erroneous.

    psx : np.array (shape (N, K))
        P(label=k|x) is a matrix with K (noisy) probabilities for each of the
        N examples x. This is the probability distribution over all K classes,
        for each example, regarding whether the example has label s==k P(s=k|x).
        psx should have been computed using 3 (or higher) fold cross-validation.

    calibrate : bool
        Set to True to calibrate the confusion matrix created by pred != given labels.
        This calibration just makes

    Returns
    -------
        A boolean mask that is true if the example belong
        to that index is label error."""

    confident_joint = confusion_matrix(np.argmax(psx, axis=1), labels).T
    if calibrate:
        confident_joint = calibrate_confident_joint(confident_joint, labels)
    return get_noise_indices(
        s=labels,
        psx=psx,
        confident_joint=confident_joint,
        prune_method=prune_method,
    )