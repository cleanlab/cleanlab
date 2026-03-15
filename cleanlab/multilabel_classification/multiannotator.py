"""
Methods for analysis of multi-label classification data labeled by multiple annotators.

This module extends the active learning functionality from `cleanlab.multiannotator`
to multi-label classification datasets, where each example can belong to multiple classes.

The key approach is to use a one-vs-rest strategy:
for each of the K classes, we treat it as a binary classification problem
and compute active learning scores. These per-class scores are then aggregated
into an overall active learning score for each example.

For more details on the underlying algorithms, see:
- `the ActiveLab paper <https://arxiv.org/abs/2301.11856>`_
- `the CROWDLAB paper <https://arxiv.org/abs/2210.06812>`_
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cleanlab.internal.constants import CLIPPING_LOWER_BOUND
from cleanlab.internal.multiannotator_utils import (
    assert_valid_inputs_multiannotator,
    assert_valid_pred_probs,
)
from cleanlab.internal.util import get_num_classes
from cleanlab.multiannotator import (
    get_active_learning_scores as get_multiclass_active_learning_scores,
    get_majority_vote_label,
)


def get_active_learning_scores(
    labels_multiannotator: Optional[Union[np.ndarray, List[List[List[int]]]]] = None,
    pred_probs: Optional[np.ndarray] = None,
    pred_probs_unlabeled: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns an ActiveLab quality score for each example in a multi-label dataset,
    to estimate which examples are most informative to (re)label next in active learning.

    This function is the multi-label version of
    `~cleanlab.multiannotator.get_active_learning_scores`.
    It handles datasets where each example can belong to multiple classes and
    has been labeled by multiple annotators.

    We consider settings where one example can be labeled by one or more annotators
    and some examples have no labels at all so far.

    The score is between 0 and 1, and can be used to prioritize what data to collect
    additional labels for. Lower scores indicate examples whose true label we are
    least confident about based on the current data; collecting additional labels
    for these low-scoring examples will be more informative than collecting labels
    for other examples.

    You can use this function to get active learning scores for:
    - examples that already have one or more labels (specify ``labels_multiannotator`` and ``pred_probs``)
    - unlabeled examples (specify ``pred_probs_unlabeled``)
    - both types of examples (specify all arguments)

    Parameters
    ----------
    labels_multiannotator : np.ndarray or List[List[List[int]]], optional
        Multi-label annotations from multiple annotators. Can be provided in two formats:

        1. **3D NumPy array** of shape ``(N, M, K)`` where:
           - N = number of examples
           - M = number of annotators
           - K = number of classes
           - Values are 0, 1, or ``NaN`` (if annotator didn't label that class)

        2. **List of list of lists** with shape ``(N, M)`` where each inner list
           contains the class indices labeled by that annotator for that example.
           e.g., ``[[[0, 1], [0], []], [[2], [1, 2], [2]]]`` for 2 examples,
           3 annotators, and classes 0, 1, 2.

        This argument is optional if you only want scores for unlabeled examples.

    pred_probs : np.ndarray, optional
        An array of shape ``(N, K)`` of predicted class probabilities from a trained
        classifier model for multi-label classification. Note that in multi-label
        classification, probabilities do not need to sum to 1 for each example.
        Required if ``labels_multiannotator`` is provided.

    pred_probs_unlabeled : np.ndarray, optional
        An array of shape ``(N_unlabeled, K)`` of predicted class probabilities for
        examples that have no annotator labels yet.

    Returns
    -------
    active_learning_scores : np.ndarray
        Array of shape ``(N,)`` with ActiveLab quality scores for labeled examples.
        Empty array if no labeled data provided. Lower scores indicate examples
        that should be prioritized for additional labeling.

    active_learning_scores_unlabeled : np.ndarray
        Array of shape ``(N_unlabeled,)`` with active learning scores for unlabeled
        examples. Empty array if no unlabeled data provided. Scores are comparable
        to those for labeled examples.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.multilabel_classification.multiannotator import get_active_learning_scores
    >>> 
    >>> # 3 examples, 2 annotators, 3 classes
    >>> # Format: (examples, annotators, classes)
    >>> labels_multiannotator = np.array([
    ...     [[1, 0, 1], [1, 0, np.nan]],  # Example 0: annotator 0 labels classes 0,2; annotator 1 labels class 0
    ...     [[0, 1, 1], [np.nan, 1, 1]],  # Example 1: both annotators label classes 1,2 (annotator 1 didn't label class 0)
    ...     [[1, 1, 0], [1, 0, 0]],       # Example 2: annotator 0 labels classes 0,1; annotator 1 labels class 0
    ... ])
    >>> pred_probs = np.array([
    ...     [0.8, 0.2, 0.7],  # Example 0
    ...     [0.1, 0.9, 0.8],  # Example 1
    ...     [0.7, 0.6, 0.1],  # Example 2
    ... ])
    >>> scores, _ = get_active_learning_scores(labels_multiannotator, pred_probs)
    """
    # Validate predicted probabilities
    assert_valid_pred_probs(pred_probs=pred_probs, pred_probs_unlabeled=pred_probs_unlabeled)

    # Convert labels to 3D array format if needed
    if labels_multiannotator is not None:
        labels_multiannotator = _convert_to_3d_array(labels_multiannotator)

    # Get number of classes
    if pred_probs is not None:
        num_classes = pred_probs.shape[1]
    elif pred_probs_unlabeled is not None:
        num_classes = pred_probs_unlabeled.shape[1]
    else:
        raise ValueError(
            "At least one of pred_probs or pred_probs_unlabeled must be provided."
        )

    # Compute scores for labeled data
    if labels_multiannotator is not None and pred_probs is not None:
        active_learning_scores = _compute_multilabel_active_learning_scores(
            labels_multiannotator=labels_multiannotator,
            pred_probs=pred_probs,
            num_classes=num_classes,
        )
    else:
        active_learning_scores = np.array([])

    # Compute scores for unlabeled data
    if pred_probs_unlabeled is not None:
        active_learning_scores_unlabeled = _compute_unlabeled_scores(
            pred_probs_unlabeled=pred_probs_unlabeled,
            num_classes=num_classes,
        )
    else:
        active_learning_scores_unlabeled = np.array([])

    return active_learning_scores, active_learning_scores_unlabeled


def _convert_to_3d_array(
    labels_multiannotator: Union[np.ndarray, List[List[List[int]]]]
) -> np.ndarray:
    """Convert multi-label multiannotator labels to a 3D numpy array format.

    Parameters
    ----------
    labels_multiannotator : np.ndarray or List[List[List[int]]]
        Labels in either 3D array format or list-of-lists format.

    Returns
    -------
    labels_3d : np.ndarray
        3D array of shape (N, M, K) with values 0, 1, or NaN.
    """
    if isinstance(labels_multiannotator, np.ndarray):
        # Already in array format
        if labels_multiannotator.ndim != 3:
            raise ValueError(
                f"labels_multiannotator must be a 3D array with shape (N, M, K), "
                f"got array with {labels_multiannotator.ndim} dimensions."
            )
        return labels_multiannotator.astype(float)

    # Convert from list-of-lists format
    # labels_multiannotator is List[List[List[int]]] with shape (N, M)
    # where each inner list contains class indices that ARE present
    # Empty list [] means the annotator labeled the example but no classes apply
    # None means the annotator did not label this example at all
    n_examples = len(labels_multiannotator)
    if n_examples == 0:
        return np.array([])

    n_annotators = len(labels_multiannotator[0])

    # Find number of classes from the data
    max_class = 0
    for example_labels in labels_multiannotator:
        for annotator_labels in example_labels:
            if annotator_labels:  # Non-empty list
                max_class = max(max_class, max(annotator_labels) + 1)
    n_classes = max_class

    # Create 3D array filled with 0 (class not present)
    labels_3d = np.zeros((n_examples, n_annotators, n_classes))

    for i, example_labels in enumerate(labels_multiannotator):
        for j, annotator_labels in enumerate(example_labels):
            if annotator_labels is None:
                # Annotator didn't label this example at all
                labels_3d[i, j, :] = np.nan
            elif len(annotator_labels) == 0:
                # Annotator labeled the example but no classes apply
                # All classes remain 0
                pass
            else:
                # Set specified classes to 1
                for class_idx in annotator_labels:
                    labels_3d[i, j, class_idx] = 1.0

    return labels_3d


def _compute_multilabel_active_learning_scores(
    labels_multiannotator: np.ndarray,
    pred_probs: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute active learning scores for multi-label data using one-vs-rest approach.

    For each class, we treat it as a binary classification problem and compute
    active learning scores. These per-class scores are then aggregated.

    Parameters
    ----------
    labels_multiannotator : np.ndarray
        3D array of shape (N, M, K) with binary labels (0, 1, or NaN).
    pred_probs : np.ndarray
        Array of shape (N, K) with predicted probabilities.
    num_classes : int
        Number of classes (K).

    Returns
    -------
    active_learning_scores : np.ndarray
        Array of shape (N,) with aggregated active learning scores.
    """
    n_examples = labels_multiannotator.shape[0]
    active_learning_scores = np.zeros(n_examples)

    # Compute scores for each class using one-vs-rest
    for k in range(num_classes):
        # Extract binary labels for class k: shape (N, M)
        class_labels = labels_multiannotator[:, :, k]

        # Check if any annotator labeled this class
        has_labels = ~np.isnan(class_labels).all(axis=1)
        if not has_labels.any():
            # No labels for this class, skip
            continue

        # Get pred_probs for this class: shape (N, 2) for binary classification
        # [1 - p_k, p_k]
        class_pred_probs = np.column_stack([
            1 - pred_probs[:, k],
            pred_probs[:, k]
        ])

        # Get active learning scores for this binary classification task
        # Only for examples that have at least one label for this class
        class_labels_labeled = class_labels[has_labels]
        class_pred_probs_labeled = class_pred_probs[has_labels]

        # Use the multiclass active learning scores function
        # with adjust_ thresholds disabled for binary case
        class_scores = np.full(n_examples, np.nan)
        try:
            scores_labeled, _ = get_multiclass_active_learning_scores(
                labels_multiannotator=class_labels_labeled,
                pred_probs=class_pred_probs_labeled,
            )
            class_scores[has_labels] = scores_labeled
        except ValueError:
            # If there's only one annotator or other issues, use a simple fallback
            # Compute a simple quality score based on agreement
            class_scores[has_labels] = _compute_simple_class_scores(
                class_labels_labeled, class_pred_probs_labeled
            )

        # Add to aggregate scores (handling NaN values)
        active_learning_scores = np.nansum(
            np.stack([active_learning_scores, np.nan_to_num(class_scores, nan=0.0)]),
            axis=0
        )

    # Divide by number of classes to get average
    # Count how many classes each example has labels for
    n_labeled_classes = np.sum(~np.isnan(labels_multiannotator).all(axis=1), axis=1)
    n_labeled_classes = np.maximum(n_labeled_classes, 1)  # Avoid division by zero

    active_learning_scores = active_learning_scores / num_classes

    return active_learning_scores


def _compute_simple_class_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> np.ndarray:
    """Compute simple active learning scores for a single binary class.

    This is a fallback when the full multiannotator logic doesn't apply
    (e.g., single annotator case).

    Parameters
    ----------
    labels : np.ndarray
        Binary labels of shape (N, M) with values 0, 1, or NaN.
    pred_probs : np.ndarray
        Predicted probabilities of shape (N, 2).

    Returns
    -------
    scores : np.ndarray
        Array of shape (N,) with active learning scores.
    """
    n_examples = labels.shape[0]
    scores = np.zeros(n_examples)

    for i in range(n_examples):
        annotator_labels = labels[i]
        valid_labels = annotator_labels[~np.isnan(annotator_labels)]

        if len(valid_labels) == 0:
            scores[i] = 0.5  # No labels, neutral score
        elif len(valid_labels) == 1:
            # Single annotator: use model confidence
            label = int(valid_labels[0])
            scores[i] = pred_probs[i, label]
        else:
            # Multiple annotators: compute agreement
            # Simple heuristic: higher agreement = higher score
            consensus = np.mean(valid_labels)
            model_conf = pred_probs[i, 1]  # Probability of positive class
            # Weighted average of annotator agreement and model confidence
            scores[i] = 0.5 * (1 - abs(consensus - model_conf)) + 0.5 * max(consensus, 1 - consensus)

    return scores


def _compute_unlabeled_scores(
    pred_probs_unlabeled: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute active learning scores for unlabeled examples.

    For multi-label, we compute a score based on the uncertainty across all classes.

    Parameters
    ----------
    pred_probs_unlabeled : np.ndarray
        Predicted probabilities of shape (N, K).
    num_classes : int
        Number of classes.

    Returns
    -------
    scores : np.ndarray
        Array of shape (N,) with active learning scores.
    """
    # For unlabeled data, use uncertainty-based scoring
    # Lower scores = more uncertain = more valuable to label

    # Compute entropy-like uncertainty for each class
    # For binary per-class decisions: uncertainty is highest when p ≈ 0.5
    uncertainties = 4 * pred_probs_unlabeled * (1 - pred_probs_unlabeled)  # Max at p=0.5

    # Average uncertainty across classes
    # Lower uncertainty = higher score (more confident)
    # So we invert: score = 1 - uncertainty
    avg_uncertainty = np.mean(uncertainties, axis=1)
    scores = 1 - avg_uncertainty

    return scores
