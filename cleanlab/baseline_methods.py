
# coding: utf-8

# Copyright (c) 2017-2050 Curtis G. Northcutt
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License

# This agreement applies to this version and all previous versions of cleanlab.


# Baseline methods
# 
# Contains baseline methods for estimating label errors.
#
# These methods ONLY WORK FOR SINGLE_LABEL (not multi-label)

from __future__ import (
    print_function, absolute_import, division, unicode_literals, with_statement
)
from sklearn.metrics import confusion_matrix
from cleanlab.pruning import get_noise_indices
from cleanlab.latent_estimation import calibrate_confident_joint
import numpy as np


def baseline_argmax(psx, s):
    '''This is the simplest baseline approach. Just consider 
    anywhere argmax != s as a label error.

    Parameters
    ----------

    s : np.array
        A discrete vector of noisy labels, i.e. some labels may be erroneous.

    psx : np.array (shape (N, K))
        P(label=k|x) is a matrix with K (noisy) probabilities for each of the
        N examples x. This is the probability distribution over all K classes,
        for each example, regarding whether the example has label s==k P(s=k|x).
        psx should have been computed using 3 (or higher) fold cross-validation.

    Returns
    -------
        A boolean mask that is true if the example belong
        to that index is label error..'''
    
    return np.argmax(psx, axis=1) != np.asarray(s)


def baseline_argmax_confusion_matrix(
    psx,
    s,
    calibrate=False,
    prune_method='prune_by_noise_rate',
):
    '''This is a baseline approach. That uses the a confusion matrix
    of argmax(psx) and s as the confident joint and then uses cleanlab
    (confident learning) to find the label errors using this matrix.

    Parameters
    ----------

    s : np.array
        A discrete vector of noisy labels, i.e. some labels may be erroneous.

    psx : np.array (shape (N, K))
        P(label=k|x) is a matrix with K (noisy) probabilities for each of the
        N examples x. This is the probability distribution over all K classes,
        for each example, regarding whether the example has label s==k P(s=k|x).
        psx should have been computed using 3 (or higher) fold cross-validation.

    Returns
    -------
        A boolean mask that is true if the example belong
        to that index is label error..'''

    confident_joint = confusion_matrix(np.argmax(psx, axis=1), s).T
    if calibrate:
        confident_joint = calibrate_confident_joint(confident_joint, s)
    return get_noise_indices(
        s=s,
        psx=psx,
        confident_joint=confident_joint,
        prune_method=prune_method,
    )


def baseline_argmax_calibrated_confusion_matrix(
    psx,
    s,
    prune_method='prune_by_noise_rate',
):
    '''docstring is the same as baseline_argmax_confusion_matrix
    Except in this method, we calibrate the confident joint created using
    the confusion matrix before using cleanlab to find the label errors.'''

    return baseline_argmax_confusion_matrix(
        s=s,
        psx=psx,
        calibrate=True,
        prune_method=prune_method,
    )
    
