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
Methods to find label issues in token classification datasets (text data), where each token in a sentence receives its own class label.
"""

import numpy as np
from typing import List, Tuple

from cleanlab.filter import find_label_issues as find_label_issues_main


def find_label_issues(
    labels: list,
    pred_probs: list,
    return_indices_ranked_by: str = "self_confidence",
) -> List[Tuple[int, int]]:
    """Returns issues identified by cleanlab

    This is a function to find the label issues for token classification datasets

    Parameters
    ----------
    labels:
        noisy token labels in nested list format, such that `labels[i]` is a list of token labels of the i'th
        sentence. For datasets with `K` classes, each label must be in 0, 1, ..., K-1. All classes must be present.

    pred_probs:
        list of np.arrays, such that `pred_probs[i]` is the model-predicted probabilities for the tokens in
        the i'th sentence, and has shape `(N, K)`. Each row of the matrix corresponds to a token `t` and contains
        the model-predicted probabilities that `t` belongs to each possible class, for each of the K classes. The
        columns must be ordered such that the probabilities correspond to class 0, 1, ..., K-1.

    return_indices_ranked_by: {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
        Returned indicies are sorted by label quality score. See `cleanlab.filter.find_label_issues` for more details on
        each label quality score method.

    Returns
    ----------
    issues:
        a list containing all potential issues identified by cleanlab, such that each element is a tuple (i, j), which
        corresponds to the j'th token of the i'th sentence.
    """
    labels_flatten = [l for label in labels for l in label]
    pred_probs_flatten = np.array([pred for pred_prob in pred_probs for pred in pred_prob])

    issues_main = find_label_issues_main(
        labels_flatten, pred_probs_flatten, return_indices_ranked_by=return_indices_ranked_by
    )

    lengths = [len(label) for label in labels]
    mapping = [[(i, j) for j in range(length)] for i, length in enumerate(lengths)]
    mapping_flatten = [index for indicies in mapping for index in indicies]

    issues = [mapping_flatten[issue] for issue in issues_main]
    return issues
