"""
Methods to find label issues in span classification datasets (text data), each token in a sentence receives one or more class labels.

The underlying label error detection algorithms are in `cleanlab.token_classification`.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union

from cleanlab.token_classification.filter import find_label_issues as find_label_issues_token
from cleanlab.token_classification.summary import display_issues as display_issues_token
from cleanlab.token_classification.rank import (
    get_label_quality_scores as get_label_quality_scores_token,
)
from cleanlab.internal.util import get_num_classes
from cleanlab.internal.token_classification_utils import color_sentence, get_sentence


def find_label_issues(
    labels: list,
    pred_probs: list,
    **kwargs,
) -> Union[Dict[int, List[Tuple[int, int]]], List[Tuple[int, int]]]:
    """Identifies tokens with label issues in a span classification dataset.

    Tokens identified with issues will be ranked by their individual label quality score.

    To rank the sentences based on their overall label quality, use :py:func:`experimental.span_classification.get_label_quality_scores <cleanlab.experimental.span_classification.get_label_quality_scores>`

    Parameters
    ----------
    labels:
        For single class span classification dataset, `labels` is a nested list of given labels for all tokens, such that `labels[i]` is a list of labels, one for each token in the `i`-th sentence.

        For multi-class span classification dataset, `labels` must be a nested list of lists, such that `labels[i]` is a list of lists, one for each token in the `i`-th sentence.
        `labels[i][j]` is a list of integers, each representing a span class label for the `j`-th token in the `i`-th sentence.

        For a dataset with K classes, each label must be in 0, 1, ..., K-1.

    pred_probs:
        List of np arrays, such that `pred_probs[i]` has shape ``(T, K)`` if the `i`-th sentence contains T tokens.

        Each row of `pred_probs[i]` corresponds to a token `t` in the `i`-th sentence,
        and contains model-predicted probabilities that `t` belongs to each of the K possible span classes.

        Columns of each `pred_probs[i]` should be ordered such that the probabilities correspond to class 0, 1, ..., K-1.

    See documentation for :py:func:`token_classification.filter.find_label_issues <cleanlab.token_classification.filter.find_label_issues>` for optional parameters description.

    Returns
    -------
    issues: list or dict
        For single span class, the return type is a list of label issues identified by cleanlab, such that each element is a tuple ``(i, j)``, which
        indicates that the `j`-th token of the `i`-th sentence has a label issue.

        For multiple span classes, the return type is a dictionary with span class as keys, and the value is a list of label issues for that class.

        These tuples are ordered in `issues` list based on the likelihood that the corresponding token is mislabeled.

        Use :py:func:`experimental.span_classification.get_label_quality_scores <cleanlab.experimental.span_classification.get_label_quality_scores>`
        to view these issues within the original sentences.

    Examples
    --------
    For a binary span classification task:
    >>> import numpy as np
    >>> from cleanlab.experimental.span_classification import find_label_issues
    >>> labels = [[0, 0, 1, 1], [1, 1, 0]]
    >>> pred_probs = [
    ...     np.array([0.9, 0.9, 0.9, 0.1]),
    ...     np.array([0.1, 0.1, 0.9]),
    ... ]
    >>> find_label_issues(labels, pred_probs)
    [(0, 3)]
    For a multi-class span classification task:
    >>> labels = [
    ...     [[0], [1, 2], [1, 3], [0]],
    ...     [[1], [2, 3], [3]],
    ... ]
    >>> pred_probs = [
    ...     np.array([[0.9, 0.2, 0.3], [0.9, 0.9, 0.2], [0.9, 0.1, 0.7], [0.1, 0.1, 0.1]]),
    ...     np.array([[0.1, 0.9, 0.1], [0.1, 0.9, 0.9], [0.1, 0.9, 0.9]]),
    ... ]
    >>> find_label_issues(labels, pred_probs)
    {1: [(0, 0), (1, 0)], 2: [(1, 0), (1, 2)], 3: []}
    """
    if not labels or not pred_probs:
        raise ValueError("labels/pred_probs cannot be empty.")

    labels_flat = [l for sentence in labels for l in sentence]
    num_span_class = get_num_classes(labels_flat)

    if num_span_class <= 2:
        pred_probs_token = [np.stack([1 - probs, probs], axis=1) for probs in pred_probs]
        return find_label_issues_token(labels, pred_probs_token, **kwargs)

    # type checks for multi-class span classification
    if not isinstance(labels_flat[0], list):
        raise ValueError("labels must be a nested list of lists, one for each sentence.")

    cls_label_issues = {}
    # iterate over each span class, excluding the 'O' class
    for cl in range(1, num_span_class):
        cls_labels = [[1 if cl in label else 0 for label in sentence] for sentence in labels]
        cls_pred_probs = [np.array([pred[cl - 1] for pred in sentence]) for sentence in pred_probs]
        pred_probs_token = [np.stack([1 - probs, probs], axis=1) for probs in cls_pred_probs]

        cls_label_issues[cl] = find_label_issues_token(cls_labels, pred_probs_token, **kwargs)

    return cls_label_issues


def display_issues(
    issues: Union[dict, list],
    tokens: List[List[str]],
    *,
    labels: Optional[list] = None,
    pred_probs: Optional[list] = None,
    exclude: List[Tuple[int, int]] = [],
    class_names: Optional[List[str]] = None,
    top: int = 20,  # number of issues to display per class
    threshold: float = 0.5,
) -> None:
    """
    Display span classification label issues, showing sentence with problematic tokens highlighted. Can also display auxiliary information
    such as labels and predicted probabilities when available.

    This method is useful for visualizing the label issues identified in each span class.

    Parameters
    ----------
    issues:
        For single span class, the input is a list of tuples ``(i, j)`` representing a label issue for the `j`-th token of the `i`-th sentence.
        For multiple span classes, the input is a dictionary with span class as keys, and the value is a list of label issues for that class.

    tokens:
        Nested list such that `tokens[i]` is a list of tokens (strings/words) that comprise the `i`-th sentence.

    labels:
        For single class span classification dataset, `labels` is a nested list of given labels for all tokens, such that `labels[i]` is a list of labels, one for each token in the `i`-th sentence.

        For multi-class span classification dataset, `labels` must be a nested list of lists, such that `labels[i]` is a list of lists, one for each token in the `i`-th sentence.
        `labels[i][j]` is a list of integers, each representing a span class label for the `j`-th token in the `i`-th sentence.

        For a dataset with K classes, each label must be in 0, 1, ..., K-1.

    pred_probs:
        List of np arrays, such that `pred_probs[i]` has shape ``(T, K)`` if the `i`-th sentence contains T tokens.

        Each row of `pred_probs[i]` corresponds to a token `t` in the `i`-th sentence,
        and contains model-predicted probabilities that `t` belongs to each of the K possible span classes.

        Columns of each `pred_probs[i]` should be ordered such that the probabilities correspond to class 0, 1, ..., K-1.

    exclude:
        List of tuples ``(cl, pred_res)`` such that the issue of the tokens will be excluded from display if the token is predicted as `pred_res` for span class `cl`.
        `pred_res` is 1 if the token is predicted as inside span, and 0 if outside span.

    class_names:
        Optional length K list of names of each class, such that `class_names[i]` is the string name of the class corresponding to `labels` with value `i`.

        If `class_names` is provided, display these string names for predicted and given labels, otherwise display the integer index of classes.

    top: int, default=20
        Maximum number of issues to be printed.

    threshold: float, default=0.5
        Threshold value to exclude tokens from display based on their predicted probabilities. This is only used when `exclude` is provided.
    """
    if not issues or not tokens:
        raise ValueError("issues/tokens cannot be empty.")

    if isinstance(issues, list):
        # single span class
        display_issues_token(
            issues,
            tokens,
            labels=labels,
            pred_probs=pred_probs,
            exclude=exclude,
            class_names=class_names,
            top=top,
        )
        return

    for cl, cl_issues in issues.items():
        # sentence level issues
        if cl_issues and not isinstance(cl_issues[0], tuple):
            display_issues_token(
                cl_issues,
                tokens,
                labels=labels,
                pred_probs=pred_probs,
                exclude=exclude,
                class_names=class_names,
                top=top,
            )
            continue

        shown = min(top, len(cl_issues))
        for issue in cl_issues:
            i, j = issue
            sentence = get_sentence(tokens[i])
            word = tokens[i][j]

            if exclude and pred_probs:
                # check if the token is excluded via threshold 0.5
                pred_res = 1 if pred_probs[i][j][cl - 1] > threshold else 0
                if (cl, pred_res) in exclude:
                    continue

            shown -= 1
            # build issue message for display
            issue_message = ""
            if class_names:
                issue_message += f"Span Class: {class_names[cl]}\n"
            else:
                issue_message += f"Span Class: {cl}\n"
            issue_message += f"Sentence index: {i}, Token index: {j}\n"
            issue_message += f"Token: {word}\n"
            if labels or pred_probs:
                issue_message += "According to provided labels/pred_probs, "
            if labels:
                label_str = "inside" if cl in labels[i][j] else "outside"
                issue_message += f"token marked as {label_str} span "
            if pred_probs:
                if labels:
                    issue_message += "but "
                else:
                    issue_message += "token "
                probs = pred_probs[i][j][cl - 1]  # kth class is at index k-1
                issue_message += f"predicted inside span with probability: {probs}"
            issue_message += "\n----"
            print(issue_message)
            print(color_sentence(sentence, word))

            if shown == 0:
                break
            print("\n")


def get_label_quality_scores(
    labels: list,
    pred_probs: list,
    **kwargs,
) -> Union[Tuple[np.ndarray, list], Tuple[dict, dict]]:
    """
    Compute label quality scores for labels in each sentence, and for individual tokens in each sentence.

    Each score is between 0 and 1.

    Lower scores indicate token labels that are less likely to be correct, or sentences that are more likely to contain a mislabeled token.

    Parameters
    ----------
    labels:
        For single class span classification dataset, `labels` is a nested list of given labels for all tokens, such that `labels[i]` is a list of labels, one for each token in the `i`-th sentence.

        For multi-class span classification dataset, `labels` must be a nested list of lists, such that `labels[i]` is a list of lists, one for each token in the `i`-th sentence.
        `labels[i][j]` is a list of integers, each representing a span class label for the `j`-th token in the `i`-th sentence.

        For a dataset with K classes, each label must be in 0, 1, ..., K-1.

    pred_probs:
        List of np arrays, such that `pred_probs[i]` has shape ``(T, K)`` if the `i`-th sentence contains T tokens.

        Each row of `pred_probs[i]` corresponds to a token `t` in the `i`-th sentence,
        and contains model-predicted probabilities that `t` belongs to each of the K possible span classes.

        Columns of each `pred_probs[i]` should be ordered such that the probabilities correspond to class 0, 1, ..., K-1.

    See documentation of :py:meth:`token_classification.rank.get_label_quality_scores<cleanlab.token_classification.rank.get_label_quality_scores>` for optional parameters description.

    Returns
    -------
    sentence_scores:
        A dictionary with span class as keys, and the value is an array of shape ``(N,)`` where `N` is the number of sentences.

        Each element of the array is a score between 0 and 1 indicating the overall label quality of the sentence.

    token_scores:
        A dictionary with span class as keys, and the value is a list of ``pd.Series``, such that the i-th element of the list
        contains the label quality scores for individual tokens in the `i`-th sentence.

        If `tokens` strings were provided, they are used as index for each ``Series``.

    Examples
    --------
    For a multi span classification task:
    >>> import numpy as np
    >>> from cleanlab.experimental.span_classification import get_label_quality_scores
    >>> labels = [
    ...     [[0], [1, 2], [1, 3], [0]],
    ...     [[1], [2, 3], [3]],
    ... ]
    >>> pred_probs = [
    ...     np.array([[0.9, 0.2, 0.3], [0.9, 0.9, 0.2], [0.9, 0.1, 0.7], [0.1, 0.1, 0.1]]),
    ...     np.array([[0.1, 0.9, 0.1], [0.1, 0.9, 0.9], [0.1, 0.9, 0.9]]),
    ... ]
    >>> sentence_scores, token_scores = get_label_quality_scores(labels, pred_probs)
    """
    if not labels or not pred_probs:
        raise ValueError("labels/pred_probs cannot be empty.")

    labels_flat = [l for sentence in labels for l in sentence]
    num_span_class = get_num_classes(labels_flat)

    if num_span_class <= 2:
        pred_probs_token = [np.stack([1 - probs, probs], axis=1) for probs in pred_probs]
        return get_label_quality_scores_token(labels, pred_probs_token, **kwargs)

    # type checks for multi-class span classification
    if not isinstance(labels_flat[0], list):
        raise ValueError("labels must be a nested list of lists, one for each sentence.")
    if not isinstance(pred_probs[0][0], np.ndarray) and not isinstance(pred_probs[0][0], list):
        raise ValueError("pred_probs must be a list of np arrays, one for each sentence.")

    sentence_scores = {}
    label_scores = {}
    # iterate over each span class, excluding the 'O' class
    for cl in range(1, num_span_class):
        cls_labels = [[1 if cl in label else 0 for label in sentence] for sentence in labels]
        cls_pred_probs = [np.array([pred[cl - 1] for pred in sentence]) for sentence in pred_probs]
        pred_probs_token = [np.stack([1 - probs, probs], axis=1) for probs in cls_pred_probs]

        sentence_scores[cl], label_scores[cl] = get_label_quality_scores_token(
            cls_labels, pred_probs_token, **kwargs
        )

    return sentence_scores, label_scores
