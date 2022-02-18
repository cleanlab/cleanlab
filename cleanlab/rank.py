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


# ## Rank
#
# Methods for ordering data.
# This includes ranking data by most confusing to a model, most likely to be an
# error or label issue, and related types of sorting / ordering of data.
import numpy as np


def order_label_errors(
    label_errors_bool,
    psx,
    labels,
    sorted_index_method="normalized_margin",
):
    """Sorts label errors by normalized margin.
    See https://arxiv.org/pdf/1810.05369.pdf (eqn 2.2)
    eg. normalized_margin = prob_label - max_prob_not_label

    Parameters
    ----------
    label_errors_bool : np.array (bool)
      Contains True if the index of labels is an error, o.w. false

    psx : np.array (shape (N, K))
      P(s=k|x) is a matrix with K probabilities for all N examples x.
      This is the probability distribution over all K classes, for each
      example, regarding whether the example has label s==k P(s=k|x). psx
      should computed using 3 (or higher) fold cross-validation.

    labels : np.array
      A binary vector of labels, which may contain label errors.

    sorted_index_method : str ['normalized_margin', 'prob_given_label']
      Method to order label error indices (instead of a bool mask), either:
        'normalized_margin' := normalized margin (p(s = k) - max(p(s != k)))
        'prob_given_label' := [psx[i][labels[i]] for i in label_errors_idx]

    Returns
    -------
      label_errors_idx : np.array (int)
        Return the index integers of the label errors, ordered by
        the normalized margin."""

    # Convert bool mask to index mask
    label_errors_idx = np.arange(len(labels))[label_errors_bool]
    # self confidence is the holdout probability that an example
    # belongs to its given class label
    self_confidence = np.array(
        # np.mean is used so that this works for multi-labels (list of lists)
        [np.mean(psx[i][labels[i]]) for i in label_errors_idx]
    )
    if sorted_index_method == "prob_given_label":
        return label_errors_idx[np.argsort(self_confidence)]
    elif sorted_index_method == "normalized_margin":
        psx_er, labels_er = psx[label_errors_bool], labels[label_errors_bool]
        max_prob_not_label = np.array(
            [max(np.delete(psx_er[i], l, -1)) for i, l in enumerate(labels_er)]
        )
        margin = self_confidence - max_prob_not_label
        return label_errors_idx[np.argsort(margin)]
    else:
        raise ValueError(
            'sorted_index_method must be "prob_given_label" or "no'
            'rmalized_margin", but is "' + sorted_index_method + '".'
        )


def get_self_confidence_for_each_label(
    psx,
    labels,
):
    """Returns the "self-confidence" for every example associated psx and
    labels.
    The self-confidence is the holdout probability that an example belongs to
    its given class label.

    Score is between 0 and 1.
    1 - clean label (not an error).
    0 - dirty label (label error).

    Parameters
    ----------
    psx : np.array (shape (N, K))
      P(s=k|x) is a matrix with K probabilities for all N examples x.
      This is the probability distribution over all K classes, for each
      example, regarding whether the example has label s==k P(s=k|x). psx
      should computed using 3 (or higher) fold cross-validation.
    labels : np.array
      A binary vector of labels, which may contain label errors.

    Returns
    -------
      self_confidence : np.array (float)
        Return the holdout probability that each example in psx belongs to its
        label. Assumes psx is computed holdout/out-of-sample."""

    # np.mean is used so that this works for multi-labels (list of lists)
    return np.array([np.mean(psx[i, l]) for i, l in enumerate(labels)])


def get_normalized_margin_for_each_label(
    psx,
    labels,
):
    """Returns the "normalized margin" for every example associated psx and
    labels.
    The normalized margin is (p(s = k) - max(p(s != k))), i.e. the probability
    of the given label minus the probability of the argmax label that is not
    the given label. This gives you an idea of how likely an example is BOTH
    its given label AND not another label, and therefore, scores its likelihood
    of being a good label or a label error.

    Score is between 0 and 1.
    1 - clean label (not an error).
    0 - dirty label (label error).

    Parameters
    ----------
    psx : np.array (shape (N, K))
      P(s=k|x) is a matrix with K probabilities for all N examples x.
      This is the probability distribution over all K classes, for each
      example, regarding whether the example has label s==k P(s=k|x). psx
      should be computed using 3 (or higher) fold cross-validation.
    labels : np.array
      A binary vector of labels, which may contain label errors.

    Returns
    -------
      normalized_margin : np.array (float)
        Return a score (between 0 and 1) for each example of its likelihood of
        being correctly labeled. Assumes psx is computed holdout/out-of-sample.
        normalized_margin = prob_label - max_prob_not_label"""

    self_confidence = get_self_confidence_for_each_label(psx, labels)
    max_prob_not_label = np.array(
        [max(np.delete(psx[i], l, -1)) for i, l in enumerate(labels)]
    )
    margin = (self_confidence - max_prob_not_label + 1) / 2
    return margin
