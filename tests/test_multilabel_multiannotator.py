"""
Tests for multilabel_classification.multiannotator module.
"""

import numpy as np
import pandas as pd
import pytest

from cleanlab.multilabel_classification.multiannotator import (
    get_active_learning_scores,
    _convert_to_binary_labels,
    _get_class_labels_for_multiannotator,
)


def test_get_active_learning_scores_basic():
    """Test basic functionality of get_active_learning_scores."""
    # Simple 3-class problem with 2 annotators
    labels_multiannotator = [
        [[0, 1], [0]],       # Example 0: annotator 0 labels classes 0,1; annotator 1 labels class 0
        [[1], [1, 2]],       # Example 1: annotator 0 labels class 1; annotator 1 labels classes 1,2
        [[0, 2], []],        # Example 2: annotator 0 labels classes 0,2; annotator 1 did not label
    ]
    
    pred_probs = np.array([
        [0.9, 0.8, 0.3],     # Example 0: High confidence for classes 0,1
        [0.2, 0.9, 0.7],     # Example 1: High confidence for classes 1,2
        [0.8, 0.3, 0.85],    # Example 2: High confidence for classes 0,2
    ])
    
    scores, scores_unlabeled = get_active_learning_scores(labels_multiannotator, pred_probs)
    
    # Check output shapes
    assert scores.shape == (3,)
    assert scores_unlabeled.shape == (0,)
    
    # Check scores are in valid range [0, 1]
    assert np.all(scores >= 0) and np.all(scores <= 1)
    
    print(f"Active learning scores: {scores}")


def test_get_active_learning_scores_with_unlabeled():
    """Test with unlabeled data."""
    labels_multiannotator = [
        [[0], [0]],
        [[1], [1]],
    ]
    
    pred_probs = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
    ])
    
    pred_probs_unlabeled = np.array([
        [0.5, 0.5],
        [0.3, 0.7],
        [0.8, 0.2],
    ])
    
    scores, scores_unlabeled = get_active_learning_scores(
        labels_multiannotator, pred_probs, pred_probs_unlabeled
    )
    
    assert scores.shape == (2,)
    assert scores_unlabeled.shape == (3,)
    
    assert np.all(scores >= 0) and np.all(scores <= 1)
    assert np.all(scores_unlabeled >= 0) and np.all(scores_unlabeled <= 1)


def test_get_active_learning_scores_single_annotator():
    """Test with a single annotator."""
    labels_multiannotator = [
        [[0, 1]],
        [[1, 2]],
        [[0, 2]],
    ]
    
    pred_probs = np.array([
        [0.9, 0.8, 0.1],
        [0.1, 0.9, 0.8],
        [0.8, 0.1, 0.9],
    ])
    
    scores, _ = get_active_learning_scores(labels_multiannotator, pred_probs)
    
    assert scores.shape == (3,)
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_get_active_learning_scores_dataframe_input():
    """Test with pandas DataFrame input."""
    # Create DataFrame with list entries
    labels_multiannotator = pd.DataFrame({
        'annotator_0': [[0, 1], [1], [0, 2]],
        'annotator_1': [[0], [1, 2], [2]],
    })
    
    pred_probs = np.array([
        [0.9, 0.8, 0.3],
        [0.2, 0.9, 0.7],
        [0.8, 0.3, 0.85],
    ])
    
    scores, _ = get_active_learning_scores(labels_multiannotator, pred_probs)
    
    assert scores.shape == (3,)
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_convert_to_binary_labels_list():
    """Test _convert_to_binary_labels with list input."""
    labels = [
        [[0, 1], [0]],
        [[1], [1, 2]],
        [[0, 2], []],
    ]
    
    binary_labels = _convert_to_binary_labels(labels, num_classes=3)
    
    assert binary_labels.shape == (3, 2, 3)
    
    # Check example 0
    assert binary_labels[0, 0, 0] == 1
    assert binary_labels[0, 0, 1] == 1
    assert binary_labels[0, 0, 2] == 0
    assert binary_labels[0, 1, 0] == 1
    assert binary_labels[0, 1, 1] == 0
    assert binary_labels[0, 1, 2] == 0
    
    # Check example 2, annotator 1 (empty list)
    assert np.all(binary_labels[2, 1, :] == 0)


def test_convert_to_binary_labels_dataframe():
    """Test _convert_to_binary_labels with DataFrame input."""
    labels = pd.DataFrame({
        'a0': [[0, 1], [1], [0, 2]],
        'a1': [[0], [1, 2], [2]],
    })
    
    binary_labels = _convert_to_binary_labels(labels, num_classes=3)
    
    assert binary_labels.shape == (3, 2, 3)
    
    # Check specific entries
    assert binary_labels[0, 0, 0] == 1
    assert binary_labels[0, 0, 1] == 1
    assert binary_labels[1, 1, 2] == 1


def test_get_class_labels_for_multiannotator():
    """Test _get_class_labels_for_multiannotator."""
    labels = [
        [[0, 1], [0]],
        [[1], [1, 2]],
    ]
    
    class_0_labels = _get_class_labels_for_multiannotator(labels, class_idx=0, num_examples=2, num_annotators=2)
    
    assert class_0_labels.shape == (2, 2)
    
    # Example 0, annotator 0 has class 0
    assert class_0_labels.iloc[0, 0] == 1.0
    # Example 0, annotator 1 has class 0
    assert class_0_labels.iloc[0, 1] == 1.0
    # Example 1, annotator 0 does not have class 0
    assert class_0_labels.iloc[1, 0] == 0.0
    # Example 1, annotator 1 does not have class 0
    assert class_0_labels.iloc[1, 1] == 0.0


def test_get_active_learning_scores_only_unlabeled():
    """Test with only unlabeled data."""
    pred_probs_unlabeled = np.array([
        [0.5, 0.5],
        [0.3, 0.7],
        [0.8, 0.2],
    ])
    
    scores, scores_unlabeled = get_active_learning_scores(
        pred_probs_unlabeled=pred_probs_unlabeled
    )
    
    assert scores.shape == (0,)
    assert scores_unlabeled.shape == (3,)
    
    assert np.all(scores_unlabeled >= 0) and np.all(scores_unlabeled <= 1)


def test_get_active_learning_scores_empty_labels():
    """Test handling of empty labels."""
    labels_multiannotator = [
        [[], []],
        [[0], []],
    ]
    
    pred_probs = np.array([
        [0.5, 0.5],
        [0.8, 0.2],
    ])
    
    scores, _ = get_active_learning_scores(labels_multiannotator, pred_probs)
    
    assert scores.shape == (2,)
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_get_active_learning_scores_error_both_none():
    """Test error when both pred_probs are None."""
    with pytest.raises(ValueError, match="cannot both be None"):
        get_active_learning_scores(labels_multiannotator=[[[0], [1]]])


def test_get_active_learning_scores_error_missing_labels():
    """Test error when pred_probs provided but labels_multiannotator is None."""
    with pytest.raises(ValueError, match="labels_multiannotator cannot be None"):
        get_active_learning_scores(pred_probs=np.array([[0.5, 0.5]]))


if __name__ == '__main__':
    test_get_active_learning_scores_basic()
    test_get_active_learning_scores_with_unlabeled()
    test_get_active_learning_scores_single_annotator()
    test_get_active_learning_scores_dataframe_input()
    test_convert_to_binary_labels_list()
    test_convert_to_binary_labels_dataframe()
    test_get_class_labels_for_multiannotator()
    test_get_active_learning_scores_only_unlabeled()
    test_get_active_learning_scores_empty_labels()
    print("All tests passed!")
