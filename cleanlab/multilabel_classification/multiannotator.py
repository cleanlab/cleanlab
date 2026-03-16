"""
Methods for active learning with multiple annotators in multi-label classification datasets.

Multi-label classification is a setting where each example can belong to one or more classes (or none at all),
and classes are not mutually exclusive. This module extends the active learning methods from
cleanlab.multiannotator to work with multi-label data.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cleanlab.internal.multilabel_utils import int2onehot
from cleanlab.internal.util import get_num_classes
from cleanlab.multiannotator import get_active_learning_scores as _get_active_learning_scores_multiclass


def get_active_learning_scores(
    labels_multiannotator: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    pred_probs: Optional[np.ndarray] = None,
    pred_probs_unlabeled: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns an ActiveLab quality score for each example in a multi-label dataset with multiple annotators,
    to estimate which examples are most informative to (re)label next in active learning.

    This is the multi-label version of the ActiveLab method. For multi-class classification,
    use :py:func:`multiannotator.get_active_learning_scores <cleanlab.multiannotator.get_active_learning_scores>` instead.

    We consider settings where one example can be labeled by one or more annotators and some examples have no labels at all so far.
    In multi-label classification, each annotator can indicate which of the K classes apply to a given example.

    The score is between 0 and 1, and can be used to prioritize what data to collect additional labels for.
    Lower scores indicate examples whose true label we are least confident about based on the current data;
    collecting additional labels for these low-scoring examples will be more informative than collecting labels for other examples.

    This function computes active learning scores by treating each class as a binary one-vs-rest classification problem.
    For each class, we compute active learning scores using the standard multi-annotator method, then average
    across all classes to get a final score for each example.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame or np.ndarray, optional
        2D pandas DataFrame or array of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.

        For multi-label data with K classes, each entry should be a list of integers indicating which
        classes apply to this example (e.g., ``[0, 2]`` means the example belongs to classes 0 and 2).
        Use an empty list ``[]`` or ``NaN`` for examples not labeled by a particular annotator.

        If pd.DataFrame, column names should correspond to each annotator's ID.
        This argument is optional if you only want to get active learning scores for unlabeled examples.

    pred_probs : np.ndarray, optional
        An array of shape ``(N, K)`` of predicted class probabilities from a trained classifier model
        for multi-label classification. Each entry should be the probability that the example belongs
        to the corresponding class (independent of other classes).
        This argument is optional if you only want to get active learning scores for unlabeled examples.

    pred_probs_unlabeled : np.ndarray, optional
        An array of shape ``(N, K)`` of predicted class probabilities from a trained classifier model
        for unlabeled examples that have no annotator labels yet.
        This argument is optional if you only want to get active learning scores for labeled examples.

    Returns
    -------
    active_learning_scores : np.ndarray
        Array of shape ``(N,)`` indicating the ActiveLab quality scores for each labeled example.
        This array is empty if no labeled data was provided.
        Examples with the lowest scores are those we should label next.

    active_learning_scores_unlabeled : np.ndarray
        Array of shape ``(N,)`` indicating the active learning quality scores for each unlabeled example.
        This array is empty if no unlabeled data was provided.
        Scores for unlabeled data are directly comparable with scores for labeled data.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from cleanlab.multilabel_classification.multiannotator import get_active_learning_scores
    >>>
    >>> # Example with 3 classes, 2 annotators
    >>> labels_multiannotator = pd.DataFrame({
    ...     'annotator_1': [[0, 1], [1], [0, 2], [], [1, 2]],
    ...     'annotator_2': [[0], [1], [2], [0, 1], [1, 2]]
    ... })
    >>> pred_probs = np.array([
    ...     [0.9, 0.8, 0.1],
    ...     [0.1, 0.9, 0.2],
    ...     [0.8, 0.1, 0.9],
    ...     [0.7, 0.6, 0.3],
    ...     [0.2, 0.85, 0.75]
    ... ])
    >>> scores, _ = get_active_learning_scores(labels_multiannotator, pred_probs)
    """
    # Validate that at least one of pred_probs or pred_probs_unlabeled is provided
    if pred_probs is None and pred_probs_unlabeled is None:
        raise ValueError(
            "pred_probs and pred_probs_unlabeled cannot both be None, specify at least one of the two."
        )

    # Validate pred_probs dimensions
    if pred_probs is not None:
        if pred_probs.ndim != 2:
            raise ValueError("pred_probs must be a 2D array with shape (N, K)")
        num_classes = pred_probs.shape[1]
    else:
        if pred_probs_unlabeled is not None:
            if pred_probs_unlabeled.ndim != 2:
                raise ValueError("pred_probs_unlabeled must be a 2D array with shape (N, K)")
            num_classes = pred_probs_unlabeled.shape[1]
        else:
            num_classes = None

    # Validate pred_probs consistency
    if pred_probs is not None and pred_probs_unlabeled is not None:
        if pred_probs.shape[1] != pred_probs_unlabeled.shape[1]:
            raise ValueError(
                "pred_probs and pred_probs_unlabeled must have the same number of classes"
            )

    # Process labeled data
    active_learning_scores = np.array([])
    if labels_multiannotator is not None and pred_probs is not None:
        # Convert DataFrame to numpy array if needed
        if isinstance(labels_multiannotator, pd.DataFrame):
            labels_multiannotator = labels_multiannotator.to_numpy()
        elif not isinstance(labels_multiannotator, np.ndarray):
            raise ValueError(
                "labels_multiannotator must be either a NumPy array or Pandas DataFrame."
            )

        # Check that labels_multiannotator is 2D
        if labels_multiannotator.ndim != 2:
            raise ValueError(
                "labels_multiannotator must be a 2D array or dataframe, "
                "each row represents an example and each column represents an annotator."
            )

        num_examples = labels_multiannotator.shape[0]
        num_annotators = labels_multiannotator.shape[1]

        # Compute per-class active learning scores
        class_scores = np.zeros((num_examples, num_classes))

        for k in range(num_classes):
            # Create binary labels for class k (one-vs-rest)
            binary_labels = np.full((num_examples, num_annotators), np.nan)

            for i in range(num_examples):
                for j in range(num_annotators):
                    label = labels_multiannotator[i, j]
                    # Check for NaN/None - handle both scalar and array cases
                    try:
                        is_na = pd.isna(label)
                        if isinstance(is_na, np.ndarray):
                            is_na = is_na.any()
                    except (ValueError, TypeError):
                        is_na = False

                    if is_na or label is None:
                        binary_labels[i, j] = np.nan
                    elif isinstance(label, (list, np.ndarray)):
                        # Multi-label format: check if class k is in the list
                        binary_labels[i, j] = 1.0 if k in label else 0.0
                    elif isinstance(label, (int, np.integer)):
                        # Single class format (should not happen in multi-label but handle gracefully)
                        binary_labels[i, j] = 1.0 if label == k else 0.0
                    else:
                        binary_labels[i, j] = np.nan

            # Get predicted probabilities for class k (shape: (N,))
            pred_probs_class = pred_probs[:, k]

            # Stack to create 2-column format: [prob_not_class, prob_class]
            pred_probs_binary = np.vstack([1 - pred_probs_class, pred_probs_class]).T

            # Compute active learning scores for this binary problem
            try:
                class_k_scores, _ = _get_active_learning_scores_multiclass(
                    labels_multiannotator=binary_labels,
                    pred_probs=pred_probs_binary,
                )

                # Handle empty scores (e.g., if no valid labels)
                if len(class_k_scores) > 0:
                    class_scores[:, k] = class_k_scores
                else:
                    # If scores are empty, assign a default value
                    class_scores[:, k] = 0.5
            except Exception:
                # If computation fails for this class, use a neutral score
                class_scores[:, k] = 0.5

        # Average scores across all classes
        active_learning_scores = np.mean(class_scores, axis=1)

    # Process unlabeled data
    active_learning_scores_unlabeled = np.array([])
    if pred_probs_unlabeled is not None:
        num_unlabeled = pred_probs_unlabeled.shape[0]
        unlabeled_class_scores = np.zeros((num_unlabeled, num_classes))

        for k in range(num_classes):
            # Get predicted probabilities for class k
            pred_probs_unlabeled_class = pred_probs_unlabeled[:, k]

            # Stack to create 2-column format
            pred_probs_unlabeled_binary = np.vstack(
                [1 - pred_probs_unlabeled_class, pred_probs_unlabeled_class]
            ).T

            # Compute active learning scores for this binary problem
            try:
                _, class_k_scores_unlabeled = _get_active_learning_scores_multiclass(
                    pred_probs_unlabeled=pred_probs_unlabeled_binary,
                )

                if len(class_k_scores_unlabeled) > 0:
                    unlabeled_class_scores[:, k] = class_k_scores_unlabeled
                else:
                    unlabeled_class_scores[:, k] = 0.5
            except Exception:
                unlabeled_class_scores[:, k] = 0.5

        # Average scores across all classes
        active_learning_scores_unlabeled = np.mean(unlabeled_class_scores, axis=1)

    return active_learning_scores, active_learning_scores_unlabeled
