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


def order_label_issues(
    label_issues_mask,
    labels,
    psx,
    rank_by="normalized_margin",
):
    """Sorts label issues by normalized margin.
    See https://arxiv.org/pdf/1810.05369.pdf (eqn 2.2)
    e.g. normalized_margin = prob_label - max_prob_not_label

    Parameters
    ----------
    label_issues_mask : np.array (bool)
      Contains True if the index of labels is an error, o.w. false

    labels : np.array
      A discrete vector of noisy labels, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    psx : np.array (shape (N, K))
      P(label=k|x) is a matrix with K model-predicted probabilities.
      Each row of this matrix corresponds to an example x and contains the model-predicted
      probabilities that x belongs to each possible class.
      The columns must be ordered such that these probabilities correspond to class 0,1,2,...
      `psx` should have been computed using 3 (or higher) fold cross-validation.

    rank_by : str ['normalized_margin', 'prob_given_label']
      Method to order label error indices (instead of a bool mask), either:
        'normalized_margin' := normalized margin (p(label = k) - max(p(label != k)))
        'prob_given_label' := [psx[i][labels[i]] for i in label_issues_idx]

    Returns
    -------
      label_issues_idx : np.array (int)
        Return the index integers of the label issues, ordered by
        the normalized margin."""

    assert (len(psx) == len(labels))
    # Convert bool mask to index mask
    label_issues_idx = np.arange(len(labels))[label_issues_mask]
    # self-confidence is the holdout probability that an example belongs to its given class label
    self_confidence = np.array(
        # np.mean is used so that this works for multi-labels (list of lists)
        [np.mean(psx[i][labels[i]]) for i in label_issues_idx]
    )
    if rank_by == "prob_given_label":
        return label_issues_idx[np.argsort(self_confidence)]
    elif rank_by == "normalized_margin":
        psx_er, labels_er = psx[label_issues_mask], labels[label_issues_mask]
        margin = get_normalized_margin_for_each_label(labels_er, psx_er)
        return label_issues_idx[np.argsort(margin)]
    else:
        raise ValueError(
            'rank_by must be "prob_given_label" or "normalized_margin", '
            'but is "' + rank_by + '".'
        )


def get_self_confidence_for_each_label(
    labels,
    psx,
):
    """Returns the "self-confidence" for every example in the dataset associated psx and labels.
    The self-confidence is the holdout probability that an example belongs to
    its given class label.

    Score is between 0 and 1.
    1 - clean label (not an error).
    0 - dirty label (label error).

    Parameters
    ----------

    labels : np.array
      A discrete vector of noisy labels, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    psx : np.array (shape (N, K))
      P(label=k|x) is a matrix with K model-predicted probabilities.
      Each row of this matrix corresponds to an example x and contains the model-predicted
      probabilities that x belongs to each possible class.
      The columns must be ordered such that these probabilities correspond to class 0,1,2,...
      `psx` should have been computed using 3 (or higher) fold cross-validation.

    Returns
    -------
      self_confidence : np.array (float)
        Return the holdout probability that each example in psx belongs to its
        label. Assumes psx is computed holdout/out-of-sample."""

    # np.mean is used so that this works for multi-labels (list of lists)
    return np.array([np.mean(psx[i, l]) for i, l in enumerate(labels)])


def get_normalized_margin_for_each_label(
    labels,
    psx,
):
    """Returns the "normalized margin" for every example associated psx and
    labels.
    The normalized margin is (p(label = k) - max(p(label != k))), i.e. the probability
    of the given label minus the probability of the argmax label that is not
    the given label. This gives you an idea of how likely an example is BOTH
    its given label AND not another label, and therefore, scores its likelihood
    of being a good label or a label error.

    Score is between 0 and 1.
    1 - clean label (not an error).
    0 - dirty label (label error).

    Parameters
    ----------

    labels : np.array
      A discrete vector of noisy labels, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    psx : np.array (shape (N, K))
      P(label=k|x) is a matrix with K model-predicted probabilities.
      Each row of this matrix corresponds to an example x and contains the model-predicted
      probabilities that x belongs to each possible class.
      The columns must be ordered such that these probabilities correspond to class 0,1,2,...
      `psx` should have been computed using 3 (or higher) fold cross-validation.

    Returns
    -------
      normalized_margin : np.array (float)
        Return a score (between 0 and 1) for each example of its likelihood of
        being correctly labeled. Assumes psx is computed holdout/out-of-sample.
        normalized_margin = prob_label - max_prob_not_label"""

    self_confidence = get_self_confidence_for_each_label(labels, psx)
    max_prob_not_label = np.array(
        [max(np.delete(psx[i], l, -1)) for i, l in enumerate(labels)]
    )
    margin = (self_confidence - max_prob_not_label + 1) / 2
    return margin
