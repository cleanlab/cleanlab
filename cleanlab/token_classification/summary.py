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
Methods to display sentences and their label issues in a token classification dataset (text data), as well as summarize the types of issues identified.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cleanlab.internal.token_classification_utils import color_sentence, get_sentence


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
    Display token classification label issues, showing sentence with problematic token(s) highlighted.

    Can also shows given and predicted label for each token identified to have label issue.

    Parameters
    ----------
    issues:
        List of tuples ``(i, j)`` representing a label issue for the `j`-th token of the `i`-th sentence.

        Same format as output by :py:func:`token_classification.filter.find_label_issues <cleanlab.token_classification.filter.find_label_issues>`
        or :py:func:`token_classification.rank.issues_from_scores <cleanlab.token_classification.rank.issues_from_scores>`.

    tokens:
        Nested list such that `tokens[i]` is a list of tokens (strings/words) that comprise the `i`-th sentence.

    labels:
        Optional nested list of given labels for all tokens, such that `labels[i]` is a list of labels, one for each token in the `i`-th sentence.
        For a dataset with K classes, each label must be in 0, 1, ..., K-1.

        If `labels` is provided, this function also displays given label of the token identified with issue.

    pred_probs:
        Optional list of np arrays, such that `pred_probs[i]` has shape ``(T, K)`` if the `i`-th sentence contains T tokens.

        Each row of `pred_probs[i]` corresponds to a token `t` in the `i`-th sentence,
        and contains model-predicted probabilities that `t` belongs to each of the K possible classes.

        Columns of each `pred_probs[i]` should be ordered such that the probabilities correspond to class 0, 1, ..., K-1.

        If `pred_probs` is provided, this function also displays predicted label of the token identified with issue.

    exclude:
        Optional list of given/predicted label swaps (tuples) to be ignored. For example, if `exclude=[(0, 1), (1, 0)]`,
        tokens whose label was likely swapped between class 0 and 1 are not displayed. Class labels must be in 0, 1, ..., K-1.

    class_names:
        Optional length K list of names of each class, such that `class_names[i]` is the string name of the class corresponding to `labels` with value `i`.

        If `class_names` is provided, display these string names for predicted and given labels, otherwise display the integer index of classes.

    top: int, default=20
        Maximum number of issues to be printed.

    Examples
    --------
    >>> from cleanlab.token_classification.summary import display_issues
    >>> issues = [(2, 0), (0, 1)]
    >>> tokens = [
    ...     ["A", "?weird", "sentence"],
    ...     ["A", "valid", "sentence"],
    ...     ["An", "sentence", "with", "a", "typo"],
    ... ]
    >>> display_issues(issues, tokens)
    Sentence 2, token 0:
    ----
    An sentence with a typo
    ...
    ...
    Sentence 0, token 1:
    ----
    A ?weird sentence
    """
    if not class_names:
        print(
            "Classes will be printed in terms of their integer index since `class_names` was not provided. "
        )
        print("Specify this argument to see the string names of each class. \n")

    top = min(top, len(issues))
    shown = 0
    is_tuple = isinstance(issues[0], tuple)

    for issue in issues:
        if is_tuple:
            i, j = issue
            sentence = get_sentence(tokens[i])
            word = tokens[i][j]

            if pred_probs:
                prediction = pred_probs[i][j].argmax()
            if labels:
                given = labels[i][j]
            if pred_probs and labels:
                if (given, prediction) in exclude:
                    continue

            if pred_probs and class_names:
                prediction = class_names[prediction]
            if labels and class_names:
                given = class_names[given]

            shown += 1
            print(f"Sentence index: {i}, Token index: {j}")
            print(f"Token: {word}")
            if labels and not pred_probs:
                print(f"Given label: {given}")
            elif not labels and pred_probs:
                print(f"Predicted label according to provided pred_probs: {prediction}")
            elif labels and pred_probs:
                print(
                    f"Given label: {given}, predicted label according to provided pred_probs: {prediction}"
                )
            print("----")
            print(color_sentence(sentence, word))
        else:
            shown += 1
            sentence = get_sentence(tokens[issue])
            print(f"Sentence issue: {sentence}")
        if shown == top:
            break
        print("\n")


def common_label_issues(
    issues: List[Tuple[int, int]],
    tokens: List[List[str]],
    *,
    labels: Optional[list] = None,
    pred_probs: Optional[list] = None,
    class_names: Optional[List[str]] = None,
    top: int = 10,
    exclude: List[Tuple[int, int]] = [],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Display the tokens (words) that most commonly have label issues.

    These may correspond to words that are ambiguous or systematically misunderstood by the data annotators.

    Parameters
    ----------
    issues:
        List of tuples ``(i, j)`` representing a label issue for the `j`-th token of the `i`-th sentence.

        Same format as output by :py:func:`token_classification.filter.find_label_issues <cleanlab.token_classification.filter.find_label_issues>`
        or :py:func:`token_classification.rank.issues_from_scores <cleanlab.token_classification.rank.issues_from_scores>`.

    tokens:
        Nested list such that `tokens[i]` is a list of tokens (strings/words) that comprise the `i`-th sentence.

    labels:
        Optional nested list of given labels for all tokens in the same format as `labels` for `~cleanlab.token_classification.summary.display_issues`.

        If `labels` is provided, this function also displays given label of the token identified to commonly suffer from label issues.

    pred_probs:
        Optional list of model-predicted probabilities (np arrays) in the same format as `pred_probs` for
        `~cleanlab.token_classification.summary.display_issues`.

        If both `labels` and `pred_probs` are provided, also reports each type of given/predicted label swap for tokens identified to commonly suffer from label issues.

    class_names:
        Optional length K list of names of each class, such that `class_names[i]` is the string name of the class corresponding to `labels` with value `i`.

        If `class_names` is provided, display these string names for predicted and given labels, otherwise display the integer index of classes.

    top:
        Maximum number of tokens to print information for.

    exclude:
        Optional list of given/predicted label swaps (tuples) to be ignored in the same format as `exclude` for
        `~cleanlab.token_classification.summary.display_issues`.

    verbose:
        Whether to also print out the token information in the returned DataFrame `df`.

    Returns
    -------
    df:
        If both `labels` and `pred_probs` are provided, DataFrame `df` contains columns ``['token', 'given_label',
        'predicted_label', 'num_label_issues']``, and each row contains information for a specific token and
        given/predicted label swap, ordered by the number of label issues inferred for this type of label swap.

        Otherwise, `df` only has columns ['token', 'num_label_issues'], and each row contains the information for a specific
        token, ordered by the number of total label issues involving this token.

    Examples
    --------
    >>> from cleanlab.token_classification.summary import common_label_issues
    >>> issues = [(2, 0), (0, 1)]
    >>> tokens = [
    ...     ["A", "?weird", "sentence"],
    ...     ["A", "valid", "sentence"],
    ...     ["An", "sentence", "with", "a", "typo"],
    ... ]
    >>> df = common_label_issues(issues, tokens)
    >>> df
        token  num_label_issues
    0      An                 1
    1  ?weird                 1
    """
    count: Dict[str, Any] = {}
    if not labels or not pred_probs:
        for issue in issues:
            i, j = issue
            word = tokens[i][j]
            if word not in count:
                count[word] = 0
            count[word] += 1

        words = [word for word in count.keys()]
        freq = [count[word] for word in words]
        rank = np.argsort(freq)[::-1][:top]

        for r in rank:
            print(
                f"Token '{words[r]}' is potentially mislabeled {freq[r]} times throughout the dataset\n"
            )

        info = [[word, f] for word, f in zip(words, freq)]
        info = sorted(info, key=lambda x: x[1], reverse=True)
        return pd.DataFrame(info, columns=["token", "num_label_issues"])

    if not class_names:
        print(
            "Classes will be printed in terms of their integer index since `class_names` was not provided. "
        )
        print("Specify this argument to see the string names of each class. \n")

    n = pred_probs[0].shape[1]
    for issue in issues:
        i, j = issue
        word = tokens[i][j]
        label = labels[i][j]
        pred = pred_probs[i][j].argmax()
        if word not in count:
            count[word] = np.zeros([n, n], dtype=int)
        if (label, pred) not in exclude:
            count[word][label][pred] += 1
    words = [word for word in count.keys()]
    freq = [np.sum(count[word]) for word in words]
    rank = np.argsort(freq)[::-1][:top]

    for r in rank:
        matrix = count[words[r]]
        most_frequent = np.argsort(count[words[r]].flatten())[::-1]
        print(
            f"Token '{words[r]}' is potentially mislabeled {freq[r]} times throughout the dataset"
        )
        if verbose:
            print(
                "---------------------------------------------------------------------------------------"
            )
            for f in most_frequent:
                i, j = f // n, f % n
                if matrix[i][j] == 0:
                    break
                if class_names:
                    print(
                        f"labeled as class `{class_names[i]}` but predicted to actually be class `{class_names[j]}` {matrix[i][j]} times"
                    )
                else:
                    print(
                        f"labeled as class {i} but predicted to actually be class {j} {matrix[i][j]} times"
                    )
        print()
    info = []
    for word in words:
        for i in range(n):
            for j in range(n):
                num = count[word][i][j]
                if num > 0:
                    if not class_names:
                        info.append([word, i, j, num])
                    else:
                        info.append([word, class_names[i], class_names[j], num])
    info = sorted(info, key=lambda x: x[3], reverse=True)
    return pd.DataFrame(
        info, columns=["token", "given_label", "predicted_label", "num_label_issues"]
    )


def filter_by_token(
    token: str, issues: List[Tuple[int, int]], tokens: List[List[str]]
) -> List[Tuple[int, int]]:
    """
    Return subset of label issues involving a particular token.

    Parameters
    ----------
    token:
        A specific token you are interested in.

    issues:
        List of tuples ``(i, j)`` representing a label issue for the `j`-th token of the `i`-th sentence.
        Same format as output by :py:func:`token_classification.filter.find_label_issues <cleanlab.token_classification.filter.find_label_issues>`
        or :py:func:`token_classification.rank.issues_from_scores <cleanlab.token_classification.rank.issues_from_scores>`.

    tokens:
        Nested list such that `tokens[i]` is a list of tokens (strings/words) that comprise the `i`-th sentence.

    Returns
    ----------
    issues_subset:
        List of tuples ``(i, j)`` representing a label issue for the `j`-th token of the `i`-th sentence, in the same format as `issues`.
        But restricting to only those issues that involve the specified `token`.

    Examples
    --------
    >>> from cleanlab.token_classification.summary import filter_by_token
    >>> token = "?weird"
    >>> issues = [(2, 0), (0, 1)]
    >>> tokens = [
    ...     ["A", "?weird", "sentence"],
    ...     ["A", "valid", "sentence"],
    ...     ["An", "sentence", "with", "a", "typo"],
    ... ]
    >>> filter_by_token(token, issues, tokens)
    [(0, 1)]
    """
    returned_issues = []
    for issue in issues:
        i, j = issue
        if token.lower() == tokens[i][j].lower():
            returned_issues.append(issue)
    return returned_issues
