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
from __future__ import division, absolute_import, print_function, \
    unicode_literals


def order_label_errors(
        label_errors_bool,
        psx,
        labels,
        sorted_index_method='normalized_margin',
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
        [np.mean(psx[i][labels[i]]) for i in label_errors_idx]
    )
    if sorted_index_method == 'prob_given_label':
        return label_errors_idx[np.argsort(self_confidence)]
    else:  # sorted_index_method == 'normalized_margin'
        margin = self_confidence - psx[label_errors_bool].max(axis=1)
        return label_errors_idx[np.argsort(margin)]