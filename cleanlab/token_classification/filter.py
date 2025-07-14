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
Methods to find label issues in token classification datasets (text data), where each token in a sentence receives its own class label.

The underlying algorithms are described in `this paper <https://arxiv.org/abs/2210.03920>`_.
"""

import numpy as np
from typing import List, Tuple
import warnings

from cleanlab.filter import find_label_issues as find_label_issues_main
from cleanlab.experimental.label_issues_batched import find_label_issues_batched


def find_label_issues(
    labels: list,
    pred_probs: list,
    *,
    return_indices_ranked_by: str = "self_confidence",
    low_memory: bool = False,
    **kwargs,
) -> List[Tuple[int, int]]:
    """Identifies tokens with label issues in a token classification dataset.

    Tokens identified with issues will be ranked by their individual label quality score.

    Instead use :py:func:`token_classification.rank.get_label_quality_scores <cleanlab.token_classification.rank.get_label_quality_scores>`
    if you prefer to rank the sentences based on their overall label quality.

    Parameters
    ----------
    labels:
        Nested list of given labels for all tokens, such that `labels[i]` is a list of labels, one for each token in the `i`-th sentence.

        For a dataset with K classes, each class label must be integer in 0, 1, ..., K-1.

    pred_probs:
        List of np arrays, such that `pred_probs[i]` has shape ``(T, K)`` if the `i`-th sentence contains T tokens.

        Each row of `pred_probs[i]` corresponds to a token `t` in the `i`-th sentence,
        and contains model-predicted probabilities that `t` belongs to each of the K possible classes.

        Columns of each `pred_probs[i]` should be ordered such that the probabilities correspond to class 0, 1, ..., K-1.

    return_indices_ranked_by: {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
        Returned token-indices are sorted by their label quality score.

        See :py:func:`cleanlab.filter.find_label_issues <cleanlab.filter.find_label_issues>`
        documentation for more details on each label quality scoring method.

    kwargs:
        Additional keyword arguments to pass into :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`
        which is internally applied at the token level. Can include values like `n_jobs` to control parallel processing, `frac_noise`, etc.

    Returns
    -------
    issues:
        List of label issues identified by cleanlab, such that each element is a tuple ``(i, j)``, which
        indicates that the `j`-th token of the `i`-th sentence has a label issue.

        These tuples are ordered in `issues` list based on the likelihood that the corresponding token is mislabeled.

        Use :py:func:`token_classification.summary.display_issues <cleanlab.token_classification.summary.display_issues>`
        to view these issues within the original sentences.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.token_classification.filter import find_label_issues
    >>> labels = [[0, 0, 1], [0, 1]]
    >>> pred_probs = [
    ...     np.array([[0.9, 0.1], [0.7, 0.3], [0.05, 0.95]]),
    ...     np.array([[0.8, 0.2], [0.8, 0.2]]),
    ... ]
    >>> find_label_issues(labels, pred_probs)
    [(1, 1)]
    """
    labels_flatten = [l for label in labels for l in label]
    pred_probs_flatten = np.array([pred for pred_prob in pred_probs for pred in pred_prob])

    if low_memory:
        for arg_name, _ in kwargs.items():
            warnings.warn(f"`{arg_name}` is not used when `low_memory=True`.")
        quality_score_kwargs = {"method": return_indices_ranked_by}
        issues_main = find_label_issues_batched(
            labels_flatten, pred_probs_flatten, quality_score_kwargs=quality_score_kwargs
        )
    else:
        issues_main = find_label_issues_main(
            labels_flatten,
            pred_probs_flatten,
            return_indices_ranked_by=return_indices_ranked_by,
            **kwargs,
        )

    lengths = [len(label) for label in labels]
    mapping = [[(i, j) for j in range(length)] for i, length in enumerate(lengths)]
    mapping_flatten = [index for indicies in mapping for index in indicies]

    issues = [mapping_flatten[issue] for issue in issues_main]
    return issues
