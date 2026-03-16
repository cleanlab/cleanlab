"""Tests for cleanlab.multilabel_classification.multiannotator module."""

import numpy as np
import pandas as pd
import pytest

from cleanlab.multilabel_classification.multiannotator import get_active_learning_scores


def test_get_active_learning_scores_basic():
    """Test basic functionality of get_active_learning_scores for multi-label data."""
    # Simple case: 3 classes, 2 annotators, 5 examples
    labels_multiannotator = pd.DataFrame({
        'annotator_1': [[0, 1], [1], [0, 2], [], [1, 2]],
        'annotator_2': [[0], [1], [2], [0, 1], [1, 2]]
    })

    pred_probs = np.array([
        [0.9, 0.8, 0.1],  # Example 0
        [0.1, 0.9, 0.2],  # Example 1
        [0.8, 0.1, 0.9],  # Example 2
        [0.7, 0.6, 0.3],  # Example 3
        [0.2, 0.85, 0.75]  # Example 4
    ])

    scores, _ = get_active_learning_scores(labels_multiannotator, pred_probs)

    assert isinstance(scores, np.ndarray)
    assert len(scores) == 5
    # Scores should be between 0 and 1
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_get_active_learning_scores_with_unlabeled():
    """Test get_active_learning_scores with both labeled and unlabeled data."""
    labels_multiannotator = pd.DataFrame({
        'annotator_1': [[0], [1], [2]],
        'annotator_2': [[0, 1], [1], [2]]
    })

    pred_probs = np.array([
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.1],
        [0.0, 0.1, 0.9]
    ])

    pred_probs_unlabeled = np.array([
        [0.8, 0.2, 0.1],
        [0.2, 0.85, 0.15]
    ])

    scores, unlabeled_scores = get_active_learning_scores(
        labels_multiannotator, pred_probs, pred_probs_unlabeled
    )

    assert isinstance(scores, np.ndarray)
    assert isinstance(unlabeled_scores, np.ndarray)
    assert len(scores) == 3
    assert len(unlabeled_scores) == 2
    assert np.all(scores >= 0) and np.all(scores <= 1)
    assert np.all(unlabeled_scores >= 0) and np.all(unlabeled_scores <= 1)


def test_get_active_learning_scores_only_unlabeled():
    """Test get_active_learning_scores with only unlabeled data."""
    pred_probs_unlabeled = np.array([
        [0.8, 0.2, 0.1],
        [0.2, 0.85, 0.15],
        [0.1, 0.1, 0.9]
    ])

    scores, unlabeled_scores = get_active_learning_scores(
        pred_probs_unlabeled=pred_probs_unlabeled
    )

    assert len(scores) == 0
    assert len(unlabeled_scores) == 3
    assert np.all(unlabeled_scores >= 0) and np.all(unlabeled_scores <= 1)


def test_get_active_learning_scores_only_labeled():
    """Test get_active_learning_scores with only labeled data."""
    labels_multiannotator = pd.DataFrame({
        'annotator_1': [[0], [1]],
        'annotator_2': [[0], [1]]
    })

    pred_probs = np.array([
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.1]
    ])

    scores, unlabeled_scores = get_active_learning_scores(
        labels_multiannotator, pred_probs
    )

    assert len(scores) == 2
    assert len(unlabeled_scores) == 0
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_get_active_learning_scores_numpy_array():
    """Test get_active_learning_scores with numpy array instead of DataFrame."""
    labels_multiannotator = np.array([
        [[0, 1], [1]],
        [[0], [0, 2]],
        [[1, 2], []]
    ], dtype=object)

    pred_probs = np.array([
        [0.8, 0.7, 0.2],
        [0.9, 0.1, 0.8],
        [0.2, 0.85, 0.75]
    ])

    scores, _ = get_active_learning_scores(labels_multiannotator, pred_probs)

    assert isinstance(scores, np.ndarray)
    assert len(scores) == 3
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_get_active_learning_scores_with_nans():
    """Test get_active_learning_scores with NaN values for unlabeled examples."""
    labels_multiannotator = pd.DataFrame({
        'annotator_1': [[0, 1], None, [2]],
        'annotator_2': [[0], [1], None]
    })

    pred_probs = np.array([
        [0.9, 0.8, 0.1],
        [0.2, 0.9, 0.3],
        [0.1, 0.2, 0.9]
    ])

    scores, _ = get_active_learning_scores(labels_multiannotator, pred_probs)

    assert isinstance(scores, np.ndarray)
    assert len(scores) == 3
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_get_active_learning_scores_single_annotator():
    """Test get_active_learning_scores with single annotator per example."""
    labels_multiannotator = pd.DataFrame({
        'annotator_1': [[0], np.nan, [1], np.nan],
        'annotator_2': [np.nan, [0], np.nan, [2]],
        'annotator_3': [np.nan, np.nan, np.nan, np.nan]
    })

    pred_probs = np.array([
        [0.9, 0.1, 0.0],
        [0.8, 0.1, 0.2],
        [0.1, 0.9, 0.1],
        [0.0, 0.2, 0.9]
    ])

    scores, _ = get_active_learning_scores(labels_multiannotator, pred_probs)

    assert isinstance(scores, np.ndarray)
    assert len(scores) == 4
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_get_active_learning_scores_error_both_none():
    """Test that error is raised when both pred_probs and pred_probs_unlabeled are None."""
    with pytest.raises(ValueError, match="pred_probs and pred_probs_unlabeled cannot both be None"):
        get_active_learning_scores()


def test_get_active_learning_scores_error_mismatched_classes():
    """Test that error is raised when pred_probs and pred_probs_unlabeled have different number of classes."""
    pred_probs = np.array([[0.9, 0.1], [0.1, 0.9]])  # 2 classes
    pred_probs_unlabeled = np.array([[0.8, 0.1, 0.1]])  # 3 classes

    with pytest.raises(ValueError, match="pred_probs and pred_probs_unlabeled must have the same number of classes"):
        get_active_learning_scores(pred_probs=pred_probs, pred_probs_unlabeled=pred_probs_unlabeled)


def test_get_active_learning_scores_error_labels_without_pred_probs():
    """Test error when labels_multiannotator is provided without pred_probs."""
    labels_multiannotator = pd.DataFrame({'a': [[0], [1]]})

    with pytest.raises(ValueError, match="pred_probs and pred_probs_unlabeled cannot both be None"):
        get_active_learning_scores(labels_multiannotator=labels_multiannotator)


def test_get_active_learning_scores_error_invalid_labels_type():
    """Test error when labels_multiannotator has invalid type."""
    labels_multiannotator = "invalid"
    pred_probs = np.array([[0.9, 0.1], [0.1, 0.9]])

    with pytest.raises(ValueError, match="labels_multiannotator must be either a NumPy array or Pandas DataFrame"):
        get_active_learning_scores(labels_multiannotator, pred_probs)


def test_get_active_learning_scores_error_wrong_dimensions():
    """Test error when labels_multiannotator has wrong dimensions."""
    # Create a true 1D array (not 2D)
    labels_multiannotator = np.array([0, 1, 2])  # 1D array
    pred_probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.0, 1.0]])

    with pytest.raises(ValueError, match="labels_multiannotator must be a 2D array"):
        get_active_learning_scores(labels_multiannotator, pred_probs)


def test_get_active_learning_scores_consistency_with_original():
    """Test that multi-label version produces sensible results."""
    # Basic sanity check that scores are in valid range
    labels_multiannotator = pd.DataFrame({
        'annotator_1': [[0], [1], [0]],
        'annotator_2': [[0], [1], [0]]
    })

    pred_probs = np.array([
        [0.9, 0.1],  # Example 0: high confidence for class 0
        [0.1, 0.9],  # Example 1: high confidence for class 1
        [0.85, 0.15]  # Example 2: similar to example 0
    ])

    scores, _ = get_active_learning_scores(labels_multiannotator, pred_probs)

    assert len(scores) == 3
    # All scores should be between 0 and 1
    assert np.all(scores >= 0) and np.all(scores <= 1)
    # Scores should not all be identical
    assert len(np.unique(scores)) > 1


def test_get_active_learning_scores_empty_labels():
    """Test handling of examples with no labels from any annotator."""
    labels_multiannotator = pd.DataFrame({
        'annotator_1': [[], [], [0]],
        'annotator_2': [[], [1], []]
    })

    pred_probs = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.9, 0.1]
    ])

    scores, _ = get_active_learning_scores(labels_multiannotator, pred_probs)

    assert len(scores) == 3
    assert np.all(scores >= 0) and np.all(scores <= 1)
