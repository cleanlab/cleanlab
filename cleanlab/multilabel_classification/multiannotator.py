"""
Methods for analysis of multi-label classification data labeled by multiple annotators.

This module extends the functionality of `cleanlab.multiannotator` to handle multi-label
classification datasets where each example can belong to multiple classes simultaneously.

The key functions provided are:

* `~cleanlab.multilabel_classification.multiannotator.get_active_learning_scores`:
  Compute active learning scores for multi-label datasets with multiple annotators,
  using a one-vs-rest approach to extend the ActiveLab algorithm.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cleanlab.multiannotator import get_active_learning_scores as get_al_scores_multiclass


def get_active_learning_scores(
    labels_multiannotator: Optional[Union[pd.DataFrame, np.ndarray, List[List[int]]]] = None,
    pred_probs: Optional[np.ndarray] = None,
    pred_probs_unlabeled: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns an ActiveLab quality score for each example in a multi-label dataset, to estimate which examples are most informative to (re)label next in active learning.

    This function extends the ActiveLab algorithm to multi-label classification settings where each example
    can belong to multiple classes simultaneously. It uses a one-vs-rest approach, computing active learning
    scores for each class independently and then aggregating them.

    We consider settings where one example can be labeled by one or more annotators and some examples have no labels at all so far.
    Each annotator provides a set of labels (zero or more classes) for each example they annotate.

    The score is between 0 and 1, and can be used to prioritize what data to collect additional labels for.
    Lower scores indicate examples whose true label we are least confident about based on the current data;
    collecting additional labels for these low-scoring examples will be more informative than collecting labels for other examples.

    You can use this function to get active learning scores for: examples that already have one or more labels (specify ``labels_multiannotator`` and ``pred_probs``
    as arguments), or for unlabeled examples (specify ``pred_probs_unlabeled``), or for both types of examples (specify all of the above arguments).

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame, np.ndarray, or List[List[int]], optional
        Multi-label annotations from multiple annotators. Can be one of:

        * A 2D pandas DataFrame or numpy array of shape ``(N, M)`` where N is the number of examples and M is the number of annotators.
          Each entry is a list/tuple of integers representing the classes assigned to that example by that annotator.
          Use an empty list ``[]`` or NaN for examples not labeled by a particular annotator.

        * A list of lists of lists: ``[[[class_indices], ...], ...]`` with outer length N (examples) and inner length M (annotators).
          Each inner-most list contains the class indices labeled by that annotator for that example.

    pred_probs : np.ndarray, optional
        An array of shape ``(N, K)`` of predicted class probabilities from a trained classifier model,
        where K is the number of classes. For multi-label classification, these are independent probabilities
        for each class (not required to sum to 1).

    pred_probs_unlabeled : np.ndarray, optional
        An array of shape ``(N, K)`` of predicted class probabilities from a trained classifier model
        for examples that have no annotator labels so far.

    Returns
    -------
    active_learning_scores : np.ndarray
        Array of shape ``(N,)`` indicating the ActiveLab quality scores for each labeled example.
        This array is empty if no labeled data was provided via ``labels_multiannotator``.
        Examples with the lowest scores are those we should label next.

    active_learning_scores_unlabeled : np.ndarray
        Array of shape ``(N,)`` indicating the active learning quality scores for each unlabeled example.
        Returns an empty array if no unlabeled data is provided.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.multilabel_classification.multiannotator import get_active_learning_scores
    >>> # 3 examples, 2 annotators, 3 classes
    >>> labels_multiannotator = [
    ...     [[0, 1], [0]],       # Example 0: annotator 0 labels classes 0,1; annotator 1 labels class 0
    ...     [[1], [1, 2]],       # Example 1: annotator 0 labels class 1; annotator 1 labels classes 1,2
    ...     [[0, 2], []],        # Example 2: annotator 0 labels classes 0,2; annotator 1 did not label
    ... ]
    >>> pred_probs = np.array([
    ...     [0.9, 0.8, 0.3],     # High confidence for classes 0,1
    ...     [0.2, 0.9, 0.7],     # High confidence for classes 1,2
    ...     [0.8, 0.3, 0.85],    # High confidence for classes 0,2
    ... ])
    >>> scores, _ = get_active_learning_scores(labels_multiannotator, pred_probs)
    >>> scores.shape
    (3,)
    """
    # Validate and process inputs
    if pred_probs is None and pred_probs_unlabeled is None:
        raise ValueError(
            "pred_probs and pred_probs_unlabeled cannot both be None, specify at least one of the two."
        )

    if pred_probs is not None:
        if labels_multiannotator is None:
            raise ValueError(
                "labels_multiannotator cannot be None when passing in pred_probs. "
                "Either provide labels_multiannotator to obtain active learning scores for the labeled examples, "
                "or just pass in pred_probs_unlabeled to get active learning scores for unlabeled examples."
            )

    # Get number of classes from pred_probs
    if pred_probs is not None:
        num_classes = pred_probs.shape[1]
        num_examples = pred_probs.shape[0]
    else:
        num_classes = pred_probs_unlabeled.shape[1]
        num_examples = pred_probs_unlabeled.shape[0]

    # Compute active learning scores using one-vs-rest approach
    active_learning_scores = np.zeros(num_examples) if pred_probs is not None else np.array([])

    # For labeled data, process each class
    if pred_probs is not None and labels_multiannotator is not None:
        # Convert labels_multiannotator to a standard format: 3D numpy array of shape (num_examples, num_annotators, num_classes)
        # where each entry is 0 or 1 indicating whether that annotator assigned that class to that example
        labels_binary = _convert_to_binary_labels(labels_multiannotator, num_classes)
        num_annotators = labels_binary.shape[1]

        # For each class, compute active learning scores using the binary classification version
        for class_idx in range(num_classes):
            # Create binary labels for this class (shape: num_examples x num_annotators)
            labels_class = labels_binary[:, :, class_idx]

            # Convert to format expected by multiannotator.get_active_learning_scores
            # For multilabel, each annotator gives 0/1 for each class
            labels_class_converted = _get_class_labels_for_multiannotator(
                labels_multiannotator, class_idx, num_examples, num_annotators
            )

            # Get pred_probs for this class
            pred_probs_class = pred_probs[:, class_idx].reshape(-1, 1)
            # For binary classification, we need 2 columns: P(class=0) and P(class=1)
            pred_probs_class = np.stack([1 - pred_probs_class[:, 0], pred_probs_class[:, 0]], axis=1)

            # Compute active learning scores for this class
            class_scores, _ = get_al_scores_multiclass(
                labels_multiannotator=labels_class_converted,
                pred_probs=pred_probs_class,
            )
            active_learning_scores += class_scores

        # Average across classes
        active_learning_scores /= num_classes

    # Compute scores for unlabeled data
    if pred_probs_unlabeled is not None:
        num_unlabeled = pred_probs_unlabeled.shape[0]
        active_learning_scores_unlabeled = np.zeros(num_unlabeled)

        for class_idx in range(num_classes):
            pred_probs_unlabeled_class = pred_probs_unlabeled[:, class_idx].reshape(-1, 1)
            pred_probs_unlabeled_class = np.stack(
                [1 - pred_probs_unlabeled_class[:, 0], pred_probs_unlabeled_class[:, 0]], axis=1
            )

            _, class_scores_unlabeled = get_al_scores_multiclass(
                pred_probs_unlabeled=pred_probs_unlabeled_class,
            )
            active_learning_scores_unlabeled += class_scores_unlabeled

        active_learning_scores_unlabeled /= num_classes
    else:
        active_learning_scores_unlabeled = np.array([])

    return active_learning_scores, active_learning_scores_unlabeled


def _convert_to_binary_labels(
    labels_multiannotator: Union[pd.DataFrame, np.ndarray, List],
    num_classes: int,
) -> np.ndarray:
    """Convert multi-label annotations to binary format.

    Parameters
    ----------
    labels_multiannotator : Multi-label annotations in various formats
    num_classes : Number of classes

    Returns
    -------
    binary_labels : np.ndarray of shape (num_examples, num_annotators, num_classes)
        Binary indicator (0 or 1) for whether each annotator assigned each class to each example.
    """
    if isinstance(labels_multiannotator, pd.DataFrame):
        num_examples = len(labels_multiannotator)
        num_annotators = len(labels_multiannotator.columns)
        binary_labels = np.zeros((num_examples, num_annotators, num_classes), dtype=int)

        for i in range(num_examples):
            for j in range(num_annotators):
                label_entry = labels_multiannotator.iloc[i, j]
                # Handle both list entries and NaN values
                if isinstance(label_entry, (list, tuple, np.ndarray)) and len(label_entry) > 0:
                    for cls in label_entry:
                        if 0 <= cls < num_classes:
                            binary_labels[i, j, cls] = 1
                elif isinstance(label_entry, (int, np.integer)):
                    if 0 <= label_entry < num_classes:
                        binary_labels[i, j, label_entry] = 1
                # Empty list or None/NaN means no annotation, leave as zeros

    elif isinstance(labels_multiannotator, np.ndarray):
        if labels_multiannotator.dtype == object:
            # Array of lists
            num_examples = len(labels_multiannotator)
            num_annotators = labels_multiannotator.shape[1] if labels_multiannotator.ndim > 1 else 1

            if labels_multiannotator.ndim == 1:
                labels_multiannotator = labels_multiannotator.reshape(-1, 1)

            binary_labels = np.zeros((num_examples, num_annotators, num_classes), dtype=int)

            for i in range(num_examples):
                for j in range(num_annotators):
                    label_entry = labels_multiannotator[i, j]
                    if label_entry is not None and not (isinstance(label_entry, float) and np.isnan(label_entry)):
                        if isinstance(label_entry, (list, tuple, np.ndarray)):
                            for cls in label_entry:
                                if 0 <= cls < num_classes:
                                    binary_labels[i, j, cls] = 1
                        elif isinstance(label_entry, (int, np.integer)):
                            if 0 <= label_entry < num_classes:
                                binary_labels[i, j, label_entry] = 1
        else:
            # Regular numeric array - assume binary indicator format
            # Shape: (num_examples, num_annotators) with class indices
            num_examples = len(labels_multiannotator)
            num_annotators = labels_multiannotator.shape[1] if labels_multiannotator.ndim > 1 else 1

            if labels_multiannotator.ndim == 1:
                labels_multiannotator = labels_multiannotator.reshape(-1, 1)

            binary_labels = np.zeros((num_examples, num_annotators, num_classes), dtype=int)

            for i in range(num_examples):
                for j in range(num_annotators):
                    label_entry = labels_multiannotator[i, j]
                    if not np.isnan(label_entry):
                        cls = int(label_entry)
                        if 0 <= cls < num_classes:
                            binary_labels[i, j, cls] = 1

    elif isinstance(labels_multiannotator, list):
        num_examples = len(labels_multiannotator)
        # Handle nested list structure
        if num_examples > 0 and isinstance(labels_multiannotator[0], list):
            num_annotators = len(labels_multiannotator[0])
        else:
            num_annotators = 1

        binary_labels = np.zeros((num_examples, num_annotators, num_classes), dtype=int)

        for i in range(num_examples):
            example_labels = labels_multiannotator[i]
            if not isinstance(example_labels, list):
                example_labels = [example_labels]

            for j in range(len(example_labels)):
                if j >= num_annotators:
                    break
                label_entry = example_labels[j]
                if label_entry is not None:
                    if isinstance(label_entry, (list, tuple)):
                        for cls in label_entry:
                            if isinstance(cls, (int, np.integer)) and 0 <= cls < num_classes:
                                binary_labels[i, j, cls] = 1
                    elif isinstance(label_entry, (int, np.integer)):
                        if 0 <= label_entry < num_classes:
                            binary_labels[i, j, label_entry] = 1

    else:
        raise ValueError(f"Unsupported labels_multiannotator format: {type(labels_multiannotator)}")

    return binary_labels


def _get_class_labels_for_multiannotator(
    labels_multiannotator: Union[pd.DataFrame, np.ndarray, List],
    class_idx: int,
    num_examples: int,
    num_annotators: int,
) -> pd.DataFrame:
    """Extract single-class labels for use with multiannotator functions.

    Returns a DataFrame where:
    - 1.0 means the annotator assigned this class to the example
    - 0.0 means the annotator did NOT assign this class (but did annotate)
    - NaN means the annotator did not provide any annotation for this example
    """
    labels_class = np.full((num_examples, num_annotators), np.nan)

    if isinstance(labels_multiannotator, pd.DataFrame):
        for i in range(num_examples):
            for j in range(min(num_annotators, len(labels_multiannotator.columns))):
                entry = labels_multiannotator.iloc[i, j]
                # Handle list entries and None/NaN values
                if isinstance(entry, (list, tuple, np.ndarray)):
                    labels_class[i, j] = 1.0 if class_idx in entry else 0.0
                elif isinstance(entry, (int, np.integer)):
                    labels_class[i, j] = 1.0 if entry == class_idx else 0.0
                # None or NaN remains as NaN (missing annotation)

    elif isinstance(labels_multiannotator, np.ndarray):
        if labels_multiannotator.ndim == 1:
            labels_multiannotator = labels_multiannotator.reshape(-1, 1)

        for i in range(min(num_examples, labels_multiannotator.shape[0])):
            for j in range(min(num_annotators, labels_multiannotator.shape[1])):
                entry = labels_multiannotator[i, j]
                if entry is not None and not (isinstance(entry, float) and np.isnan(entry)):
                    if isinstance(entry, (list, tuple, np.ndarray)):
                        labels_class[i, j] = 1.0 if class_idx in entry else 0.0
                    elif isinstance(entry, (int, np.integer)):
                        labels_class[i, j] = 1.0 if entry == class_idx else 0.0

    elif isinstance(labels_multiannotator, list):
        for i in range(min(num_examples, len(labels_multiannotator))):
            example_labels = labels_multiannotator[i]
            if not isinstance(example_labels, list):
                example_labels = [example_labels]

            for j in range(min(num_annotators, len(example_labels))):
                entry = example_labels[j]
                if entry is not None:
                    if isinstance(entry, (list, tuple)):
                        labels_class[i, j] = 1.0 if class_idx in entry else 0.0
                    elif isinstance(entry, (int, np.integer)):
                        labels_class[i, j] = 1.0 if entry == class_idx else 0.0

    return pd.DataFrame(labels_class)
