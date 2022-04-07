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


"""Dataset Module
Supports dataset-level and class-level automated quality, including finding which classes should
be merged with other classes in your dataset, which classes should be removed,
and which classes have the best label accuracy, and which have the worst labels.
"""

import numpy as np
from count import estimate_joint, compute_confident_joint


def find_similar_classes(
    joint=None,
    asymmetric=True,
    *,
    labels=None,
    pred_probs=None,
    confident_joint=None,
    multi_label=False,
):
    """Ranks all pairs of classes by their similarity and returns the ordered pairs as well as
    a similarity score between 0 and 1. Use this function to determine which classes in a dataset
    should be merged.

    The method uses the joint distribution of noisy and true labels to compute ontological issues
    via the approach published in (Northcutt, Jiang, Chuang (2021) JAIR).
    paper: https://arxiv.org/abs/1911.00068

    This method works by providing any one (and only one) of the following inputs:
    1. confident_joint
    2. joint
    3. labels and pred_probs
    but it is not necessary to provide a combination, e.g. the labels and pred_probs and joint.

    For parameter info, see the docstring of `count.estimate_joint`

    Parameters
    ----------
    See `count.estimate_joint` docstring.

    asymmetric : bool (default: True)
      If `asymmetric==True`, includes both pairs (class1, class2) and (class2, class1). Use this for
      finding "is a" relationships where for example "class1 is a class2".
      If `asymmetric==False`, the pair (class1, class2) will onyl be returned once and order is
      arbitrary (internally this is just summing score(class1, class2) + score(class2, class1)).

    Returns
    -------
        A tuple of 2 ordered lists: (list_of_pairs_of_classes, likelihood_they_should_be_merged)"""

    def _find_largest_pairs(matrix):
        """Helper function that returns the sorted (descending) (row, column) index pairs
        and their values for an arbitrary 2-d numpy matrix."""

        # Find indices of similar classes
        idx = np.array(list(zip(*np.unravel_index(np.argsort(matrix.ravel()), matrix.shape)))[::-1])
        # Bring max score between 0.1 and 1.0, otherwise score scales inversely with num_classes ^ 2
        scaling_factor = 10 ** np.floor(-np.log10(np.max(matrix)))
        scores = np.sort(matrix.ravel()) * scaling_factor
        return idx, scores

    if joint is None:
        joint = estimate_joint(
            labels=labels,
            pred_probs=pred_probs,
            confident_joint=confident_joint,
            multi_label=multi_label,
        )
    if asymmetric:
        # Remove diagonal elements
        joint_no_diagonal = joint - np.eye(len(joint)) * joint.diagonal()
        idx, scores = _find_largest_pairs(joint_no_diagonal)
        cutoff = len(joint) ** 2 - len(joint)  # Omit diagonal elements from end of ranking
        return idx[:cutoff], scores[:cutoff]
    else:  # symmetric
        # Sum the upper and lower triangles and remove the lower triangle and the diagonal
        # This provides values only in (the upper triangle) of the matrix.
        sym_joint = np.triu(joint) + np.tril(joint).T
        sym_joint = sym_joint - np.eye(len(sym_joint)) * sym_joint.diagonal()
        idx, scores = _find_largest_pairs(sym_joint)
        cutoff = (len(joint) ** 2 - len(joint)) / 2  # Omit lower triangle and diagonal elements
        return idx[:cutoff], scores[:cutoff]


def get_classes_ranked_by_label_noise(joint, descending=True, rank_by="given_label"):
    """Returns the ordered class indices and associated scores quantifying how
    noisy (prone to label issues) each class in the dataset is.

    Score values are unnormalized and may tend to be very small. What matters is their relative
    ranking across the classes."""

    if rank_by == "given_label":
        # Ranks the label quality of each class based on the given (potentially error-prone) labels
        # This score represents how many of the given labels in each class likely have issues.
        scores = joint.sum(axis=1) - joint.diagonal()
    elif rank_by == "true_label":
        # Ranks the label quality of each class based on the estimated true labels.
        # This score represents how likely each class was actually labeled something else.
        scores = joint.sum(axis=0) - joint.diagonal()
    elif rank_by == "total_noise":
        # p(label=k, true_label) + p(label, true_label=k) - 2 * P(label=k, true_label=k)
        # The error for the class (both in terms of how often the given label is wrong
        # and how often other classes should be this label.
        scores = joint.sum(axis=0) + joint.sum(axis=1) - 2 * joint.diagonal()
    else:
        raise ValueError(
            f"parameter 'rank_by' should be 'given_label', 'true_label', or"
            "'total_noise', but was {rank_by}"
        )
    class_indices, scores = np.argsort(scores), np.sort(scores)
    if descending:
        class_indices, scores = class_indices[::-1], scores[::-1]
    return class_indices, scores


def get_classes_ranked_by_label_quality(joint, descending=True):
    """Returns the ordered class indices and associated scores quantifying the quality of the labels
     in each class (larger number implies higher quality labels) for the joint of a dataset.

    Score values are unnormalized. What matters is relative ranking across the classes."""

    scores = joint.diagonal()
    class_indices, scores = np.argsort(scores), np.sort(scores)
    if descending:
        class_indices, scores = class_indices[::-1], scores[::-1]
    return class_indices, scores


def dataset_label_quality_score(joint):
    """Returns a single metric for the quality of the labels in a dataset. The score is between
    0 and 1. A higher score implies higher quality labels, with 1 implying labels that have no
    issues."""

    return joint.trace()
