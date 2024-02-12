"""
Methods to find label issues in span classification datasets (text data), each token in a sentence receives one or more class labels.

The underlying label error detection algorithms are in `cleanlab.token_classification`.
"""

import numpy as np
from typing import List, Tuple, Optional

from cleanlab.token_classification.filter import find_label_issues as find_label_issues_token
from cleanlab.token_classification.summary import display_issues as display_issues_token
from cleanlab.token_classification.rank import (
    get_label_quality_scores as get_label_quality_scores_token,
)


def find_label_issues(
    labels: list,
    pred_probs: list,
):
    """Identifies tokens with label issues in a span classification dataset.

    Tokens identified with issues will be ranked by their individual label quality score.

    To rank the sentences based on their overall label quality, use :py:func:`experimental.span_classification.get_label_quality_scores <cleanlab.experimental.span_classification.get_label_quality_scores>`

    Parameters
    ----------
    labels:
        Nested list of given labels for all tokens.
         Refer to documentation for this argument in :py:func:`token_classification.filter.find_label_issues <cleanlab.token_classification.filter.find_label_issues>` for further details.

      Note:  Currently, only a single span class is supported.

    pred_probs:
        An array of shape ``(T, K)`` of model-predicted class probabilities.
       Refer to documentation for this argument in :py:func:`token_classification.filter.find_label_issues <cleanlab.token_classification.filter.find_label_issues>` for further details.

    Returns
    -------
    issues:
        List of label issues identified by cleanlab, such that each element is a tuple ``(i, j)``, which
        indicates that the `j`-th token of the `i`-th sentence has a label issue.

        These tuples are ordered in `issues` list based on the likelihood that the corresponding token is mislabeled.

        Use :py:func:`experimental.span_classification.get_label_quality_scores <cleanlab.experimental.span_classification.get_label_quality_scores>`
        to view these issues within the original sentences.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.experimental.span_classification import find_label_issues
    >>> labels = [[0, 0, 1, 1], [1, 1, 0]]
    >>> pred_probs = [
    ...     np.array([0.9, 0.9, 0.9, 0.1]),
    ...     np.array([0.1, 0.1, 0.9]),
    ... ]
    >>> find_label_issues(labels, pred_probs)
    """
    pred_probs_token = _get_pred_prob_token(pred_probs)
    return find_label_issues_token(labels, pred_probs_token)


def display_issues(
    issues: list,
    tokens: List[List[str]],
    *,
    labels: Optional[list] = None,
    pred_probs: Optional[list] = None,
    exclude: List[Tuple[int, int]] = [],
    class_names: Optional[List[str]] = None,
    top: int = 20,
) -> None:
    """
    See documentation of :py:meth:`token_classification.summary.display_issues<cleanlab.token_classification.summary.display_issues>` for description.
    """
    display_issues_token(
        issues,
        tokens,
        labels=labels,
        pred_probs=pred_probs,
        exclude=exclude,
        class_names=class_names,
        top=top,
    )


def get_label_quality_scores(
    labels: list,
    pred_probs: list,
    **kwargs,
) -> Tuple[np.ndarray, list]:
    """
    See documentation of :py:meth:`token_classification.rank.get_label_quality_scores<cleanlab.token_classification.rank.get_label_quality_scores>` for description.
    """
    pred_probs_token = _get_pred_prob_token(pred_probs)
    return get_label_quality_scores_token(labels, pred_probs_token, **kwargs)


def _get_pred_prob_token(pred_probs: list) -> list:
    """Converts pred_probs for span classification to pred_probs for token classification."""
    pred_probs_token = []
    for probs in pred_probs:
        pred_probs_token.append(np.stack([1 - probs, probs], axis=1))
    return pred_probs_token
