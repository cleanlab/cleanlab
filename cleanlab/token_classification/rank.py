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
Methods to rank and score sentences in a token classification dataset (text data), based on how likely they are to contain label errors.

The underlying algorithms are described in `this paper <https://arxiv.org/abs/2210.03920>`_.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple

from cleanlab.rank import get_label_quality_scores as main_get_label_quality_scores
from cleanlab.internal.numerics import softmax


def get_label_quality_scores(
    labels: list,
    pred_probs: list,
    *,
    tokens: Optional[list] = None,
    token_score_method: str = "self_confidence",
    sentence_score_method: str = "min",
    sentence_score_kwargs: dict = {},
) -> Tuple[np.ndarray, list]:
    """
    Returns overall quality scores for the labels in each sentence, as well as for the individual tokens' labels in a token classification dataset.

    Each score is between 0 and 1.

    Lower scores indicate token labels that are less likely to be correct, or sentences that are more likely to contain a mislabeled token.

    Parameters
    ----------
    labels:
        Nested list of given labels for all tokens, such that `labels[i]` is a list of labels, one for each token in the `i`-th sentence.

        For a dataset with K classes, each label must be in 0, 1, ..., K-1.

    pred_probs:
        List of np arrays, such that `pred_probs[i]` has shape ``(T, K)`` if the `i`-th sentence contains T tokens.

        Each row of `pred_probs[i]` corresponds to a token `t` in the `i`-th sentence,
        and contains model-predicted probabilities that `t` belongs to each of the K possible classes.

        Columns of each `pred_probs[i]` should be ordered such that the probabilities correspond to class 0, 1, ..., K-1.

    tokens:
        Nested list such that `tokens[i]` is a list of tokens (strings/words) that comprise the `i`-th sentence.

        These strings are used to annotated the returned `token_scores` object, see its documentation for more information.

    sentence_score_method: {"min", "softmin"}, default="min"
        Method to aggregate individual token label quality scores into a single score for the sentence.

        - `min`: sentence score = minimum of token scores in the sentence
        - `softmin`: sentence score = ``<s, softmax(1-s, t)>``, where `s` denotes the token label scores of the sentence, and ``<a, b> == np.dot(a, b)``.
          Here parameter `t` controls the softmax temperature, such that the score converges toward `min` as ``t -> 0``.
          Unlike `min`, `softmin` is affected by the scores of all tokens in the sentence.

    token_score_method: {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
        Label quality scoring method for each token.

        See :py:func:`cleanlab.rank.get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` documentation for more info.

    sentence_score_kwargs:
        Optional keyword arguments for `sentence_score_method` function (for advanced users only).

        See `~cleanlab.token_classification.rank._softmin_sentence_score` for more info about keyword arguments supported for that scoring method.

    Returns
    -------
    sentence_scores:
        Array of shape ``(N, )`` of scores between 0 and 1, one per sentence in the dataset.

        Lower scores indicate sentences more likely to contain a label issue.

    token_scores:
        List of ``pd.Series``, such that `token_info[i]` contains the
        label quality scores for individual tokens in the `i`-th sentence.

        If `tokens` strings were provided, they are used as index for each ``Series``.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.token_classification.rank import get_label_quality_scores
    >>> labels = [[0, 0, 1], [0, 1]]
    >>> pred_probs = [
    ...     np.array([[0.9, 0.1], [0.7, 0.3], [0.05, 0.95]]),
    ...     np.array([[0.8, 0.2], [0.8, 0.2]]),
    ... ]
    >>> sentence_scores, token_scores = get_label_quality_scores(labels, pred_probs)
    >>> sentence_scores
    array([0.7, 0.2])
    >>> token_scores
    [0    0.90
    1    0.70
    2    0.95
    dtype: float64, 0    0.8
    1    0.2
    dtype: float64]
    """
    methods = ["min", "softmin"]
    assert sentence_score_method in methods, "Select from the following methods:\n%s" % "\n".join(
        methods
    )

    labels_flatten = np.array([l for label in labels for l in label])
    pred_probs_flatten = np.array([p for pred_prob in pred_probs for p in pred_prob])

    sentence_length = [len(label) for label in labels]

    def nested_list(x, sentence_length):
        i = iter(x)
        return [[next(i) for _ in range(length)] for length in sentence_length]

    token_scores = main_get_label_quality_scores(
        labels=labels_flatten, pred_probs=pred_probs_flatten, method=token_score_method
    )
    scores_nl = nested_list(token_scores, sentence_length)

    if sentence_score_method == "min":
        sentence_scores = np.array(list(map(np.min, scores_nl)))
    else:
        assert sentence_score_method == "softmin"
        temperature = sentence_score_kwargs.get("temperature", 0.05)
        sentence_scores = _softmin_sentence_score(scores_nl, temperature=temperature)

    if tokens:
        token_info = [pd.Series(scores, index=token) for scores, token in zip(scores_nl, tokens)]
    else:
        token_info = [pd.Series(scores) for scores in scores_nl]
    return sentence_scores, token_info


def issues_from_scores(
    sentence_scores: np.ndarray, *, token_scores: Optional[list] = None, threshold: float = 0.1
) -> Union[list, np.ndarray]:
    """
    Converts scores output by `~cleanlab.token_classification.rank.get_label_quality_scores`
    to a list of issues of similar format as output by :py:func:`token_classification.filter.find_label_issues <cleanlab.token_classification.filter.find_label_issues>`.

    Issues are sorted by label quality score, from most to least severe.

    Only considers as issues those tokens with label quality score lower than `threshold`,
    so this parameter determines the number of issues that are returned.
    This method is intended for converting the most severely mislabeled examples to a format compatible with
    ``summary`` methods like :py:func:`token_classification.summary.display_issues <cleanlab.token_classification.summary.display_issues>`.
    This method does not estimate the number of label errors since the `threshold` is arbitrary,
    for that instead use :py:func:`token_classification.filter.find_label_issues <cleanlab.token_classification.filter.find_label_issues>`,
    which estimates the label errors via Confident Learning rather than score thresholding.

    Parameters
    ----------
    sentence_scores:
        Array of shape `(N, )` of overall sentence scores, where `N` is the number of sentences in the dataset.

        Same format as the `sentence_scores` returned by `~cleanlab.token_classification.rank.get_label_quality_scores`.

    token_scores:
        Optional list such that `token_scores[i]` contains the individual token scores for the `i`-th sentence.

        Same format as the `token_scores` returned by `~cleanlab.token_classification.rank.get_label_quality_scores`.

    threshold:
        Tokens (or sentences, if `token_scores` is not provided) with quality scores above the `threshold` are not
        included in the result.

    Returns
    ---------
    issues:
        List of label issues identified by comparing quality scores to threshold, such that each element is a tuple ``(i, j)``, which
        indicates that the `j`-th token of the `i`-th sentence has a label issue.

        These tuples are ordered in `issues` list based on the token label quality score.

        Use :py:func:`token_classification.summary.display_issues <cleanlab.token_classification.summary.display_issues>`
        to view these issues within the original sentences.

        If `token_scores` is not provided, returns array of integer indices (rather than tuples) of the sentences whose label quality score
        falls below the `threshold` (also sorted by overall label quality score of each sentence).

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.token_classification.rank import issues_from_scores
    >>> sentence_scores = np.array([0.1, 0.3, 0.6, 0.2, 0.05, 0.9, 0.8, 0.0125, 0.5, 0.6])
    >>> issues_from_scores(sentence_scores)
    array([7, 4])

    Changing the score threshold

    >>> issues_from_scores(sentence_scores, threshold=0.5)
    array([7, 4, 0, 3, 1])

    Providing token scores along with sentence scores finds issues at the token level

    >>> token_scores = [
    ...     [0.9, 0.6],
    ...     [0.0, 0.8, 0.8],
    ...     [0.8, 0.8],
    ...     [0.1, 0.02, 0.3, 0.4],
    ...     [0.1, 0.2, 0.03, 0.4],
    ...     [0.1, 0.2, 0.3, 0.04],
    ...     [0.1, 0.2, 0.4],
    ...     [0.3, 0.4],
    ...     [0.08, 0.2, 0.5, 0.4],
    ...     [0.1, 0.2, 0.3, 0.4],
    ... ]
    >>> issues_from_scores(sentence_scores, token_scores=token_scores)
    [(1, 0), (3, 1), (4, 2), (5, 3), (8, 0)]
    """
    if token_scores:
        issues_with_scores = []
        for sentence_index, scores in enumerate(token_scores):
            for token_index, score in enumerate(scores):
                if score < threshold:
                    issues_with_scores.append((sentence_index, token_index, score))

        issues_with_scores = sorted(issues_with_scores, key=lambda x: x[2])
        issues = [(i, j) for i, j, _ in issues_with_scores]
        return issues

    else:
        ranking = np.argsort(sentence_scores)
        cutoff = 0
        while sentence_scores[ranking[cutoff]] < threshold and cutoff < len(ranking):
            cutoff += 1
        return ranking[:cutoff]


def _softmin_sentence_score(
    token_scores: List[np.ndarray], *, temperature: float = 0.05
) -> np.ndarray:
    """
    Sentence overall label quality scoring using the "softmin" method.

    Parameters
    ----------
    token_scores:
        Per-token label quality scores in nested list format,
        where `token_scores[i]` is a list of scores for each toke in the i'th sentence.

    temperature:
        Temperature of the softmax function.

        Lower values encourage this method to converge toward the label quality score of the token with the lowest quality label in the sentence.

        Higher values encourage this method to converge toward the average label quality score of all tokens in the sentence.

    Returns
    ---------
    sentence_scores:
        Array of shape ``(N, )``, where N is the number of sentences in the dataset, with one overall label quality score for each sentence.

    Examples
    ---------
    >>> from cleanlab.token_classification.rank import _softmin_sentence_score
    >>> token_scores = [[0.9, 0.6], [0.0, 0.8, 0.8], [0.8]]
    >>> _softmin_sentence_score(token_scores)
    array([6.00741787e-01, 1.80056239e-07, 8.00000000e-01])
    """
    if temperature == 0:
        return np.array([np.min(scores) for scores in token_scores])

    if temperature == np.inf:
        return np.array([np.mean(scores) for scores in token_scores])

    def fun(scores: np.ndarray) -> float:
        return np.dot(
            scores, softmax(x=1 - np.array(scores), temperature=temperature, axis=0, shift=True)
        )

    sentence_scores = list(map(fun, token_scores))
    return np.array(sentence_scores)
