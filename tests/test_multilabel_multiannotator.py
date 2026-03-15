"""
Tests for multi-label classification with multiple annotators.
"""

import numpy as np
import pytest

from cleanlab.multilabel_classification.multiannotator import (
    get_active_learning_scores,
    _convert_to_3d_array,
)


class TestGetActiveLearningScores:
    """Tests for get_active_learning_scores function."""

    def test_basic_3d_array_input(self):
        """Test with basic 3D array input."""
        # 3 examples, 2 annotators, 3 classes
        labels = np.array([
            [[1, 0, 1], [1, 0, np.nan]],
            [[0, 1, 1], [np.nan, 1, 1]],
            [[1, 1, 0], [1, 0, 0]],
        ], dtype=float)

        pred_probs = np.array([
            [0.8, 0.2, 0.7],
            [0.1, 0.9, 0.8],
            [0.7, 0.6, 0.1],
        ])

        scores, _ = get_active_learning_scores(labels, pred_probs)

        assert isinstance(scores, np.ndarray)
        assert scores.shape == (3,)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_list_of_lists_input(self):
        """Test with list-of-lists format input."""
        # 2 examples, 2 annotators
        labels = [
            [[0, 2], [0]],       # Example 0
            [[1, 2], [1, 2]],    # Example 1
        ]

        pred_probs = np.array([
            [0.8, 0.2, 0.7],
            [0.1, 0.9, 0.8],
        ])

        scores, _ = get_active_learning_scores(labels, pred_probs)

        assert isinstance(scores, np.ndarray)
        assert scores.shape == (2,)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_unlabeled_only(self):
        """Test with only unlabeled data."""
        pred_probs_unlabeled = np.array([
            [0.8, 0.2, 0.7],
            [0.1, 0.9, 0.8],
        ])

        scores, unlabeled_scores = get_active_learning_scores(
            pred_probs_unlabeled=pred_probs_unlabeled
        )

        assert scores.size == 0  # No labeled data
        assert isinstance(unlabeled_scores, np.ndarray)
        assert unlabeled_scores.shape == (2,)
        assert np.all(unlabeled_scores >= 0) and np.all(unlabeled_scores <= 1)

    def test_labeled_and_unlabeled(self):
        """Test with both labeled and unlabeled data."""
        labels = np.array([
            [[1, 0, 1], [1, 0, np.nan]],
            [[0, 1, 1], [np.nan, 1, 1]],
        ], dtype=float)

        pred_probs = np.array([
            [0.8, 0.2, 0.7],
            [0.1, 0.9, 0.8],
        ])

        pred_probs_unlabeled = np.array([
            [0.7, 0.6, 0.1],
            [0.3, 0.4, 0.9],
        ])

        scores, unlabeled_scores = get_active_learning_scores(
            labels, pred_probs, pred_probs_unlabeled
        )

        assert scores.shape == (2,)
        assert unlabeled_scores.shape == (2,)
        assert np.all(scores >= 0) and np.all(scores <= 1)
        assert np.all(unlabeled_scores >= 0) and np.all(unlabeled_scores <= 1)

    def test_single_annotator(self):
        """Test with single annotator."""
        # 3 examples, 1 annotator, 3 classes
        labels = np.array([
            [[1, 0, 1]],
            [[0, 1, 1]],
            [[1, 1, 0]],
        ], dtype=float)

        pred_probs = np.array([
            [0.8, 0.2, 0.7],
            [0.1, 0.9, 0.8],
            [0.7, 0.6, 0.1],
        ])

        scores, _ = get_active_learning_scores(labels, pred_probs)

        assert scores.shape == (3,)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_empty_class_labels(self):
        """Test when some classes have no labels."""
        labels = np.array([
            [[1, np.nan, np.nan], [1, np.nan, np.nan]],
            [[np.nan, 1, np.nan], [np.nan, 1, np.nan]],
        ], dtype=float)

        pred_probs = np.array([
            [0.8, 0.2, 0.5],
            [0.1, 0.9, 0.3],
        ])

        scores, _ = get_active_learning_scores(labels, pred_probs)

        assert scores.shape == (2,)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_no_overlap_annotators(self):
        """Test when annotators label different examples."""
        labels = np.array([
            [[1, 0, 1], [np.nan, np.nan, np.nan]],
            [[np.nan, np.nan, np.nan], [0, 1, 1]],
        ], dtype=float)

        pred_probs = np.array([
            [0.8, 0.2, 0.7],
            [0.1, 0.9, 0.8],
        ])

        scores, _ = get_active_learning_scores(labels, pred_probs)

        assert scores.shape == (2,)
        assert np.all(scores >= 0) and np.all(scores <= 1)


class TestConvertTo3DArray:
    """Tests for _convert_to_3d_array function."""

    def test_already_3d_array(self):
        """Test that 3D arrays are returned as-is."""
        arr = np.array([
            [[1, 0], [1, 0]],
            [[0, 1], [0, 1]],
        ])
        result = _convert_to_3d_array(arr)
        np.testing.assert_array_equal(result, arr.astype(float))

    def test_list_of_lists_conversion(self):
        """Test conversion from list-of-lists format."""
        labels = [
            [[0, 2], [0]],
            [[1, 2], [1, 2]],
        ]
        result = _convert_to_3d_array(labels)

        expected_shape = (2, 2, 3)  # 2 examples, 2 annotators, 3 classes
        assert result.shape == expected_shape
        assert result[0, 0, 0] == 1  # Example 0, annotator 0, class 0
        assert result[0, 0, 2] == 1  # Example 0, annotator 0, class 2
        assert result[0, 0, 1] == 0  # Example 0, annotator 0, class 1 (not labeled = absent)
        assert result[0, 1, 0] == 1  # Example 0, annotator 1, class 0
        assert result[0, 1, 1] == 0  # Example 0, annotator 1, class 1 (not labeled = absent)

    def test_list_of_lists_with_none(self):
        """Test conversion from list-of-lists format with None (unlabeled)."""
        labels = [
            [[0, 2], None],  # Example 0: annotator 1 didn't label at all
            [[1, 2], [1, 2]],
        ]
        result = _convert_to_3d_array(labels)

        expected_shape = (2, 2, 3)  # 2 examples, 2 annotators, 3 classes
        assert result.shape == expected_shape
        # First example, second annotator should be all NaN (didn't label)
        assert np.all(np.isnan(result[0, 1, :]))
        # Second example should have all values (no None)
        assert not np.any(np.isnan(result[1, :, :]))

    def test_empty_list(self):
        """Test with empty list."""
        labels = []
        result = _convert_to_3d_array(labels)
        assert result.size == 0

    def test_wrong_dimensions(self):
        """Test error on wrong number of dimensions."""
        arr = np.array([[1, 0], [1, 0]])  # 2D
        with pytest.raises(ValueError, match="3D array"):
            _convert_to_3d_array(arr)


class TestErrorCases:
    """Tests for error handling."""

    def test_no_pred_probs(self):
        """Test error when no pred_probs provided."""
        with pytest.raises(ValueError, match="pred_probs"):
            get_active_learning_scores()

    def test_labels_without_pred_probs(self):
        """Test error when labels provided without pred_probs."""
        labels = np.array([[[1, 0], [1, 0]]])
        with pytest.raises(ValueError, match="pred_probs"):
            get_active_learning_scores(labels_multiannotator=labels)


class TestScoreProperties:
    """Tests for properties of the computed scores."""

    def test_score_range(self):
        """Test that scores are always in [0, 1]."""
        np.random.seed(42)

        # Generate random data
        n_examples = 10
        n_annotators = 3
        n_classes = 5

        labels = np.random.randint(0, 2, size=(n_examples, n_annotators, n_classes)).astype(float)
        # Add some NaN values
        mask = np.random.random((n_examples, n_annotators, n_classes)) < 0.2
        labels[mask] = np.nan

        pred_probs = np.random.uniform(0, 1, size=(n_examples, n_classes))

        scores, _ = get_active_learning_scores(labels, pred_probs)

        assert np.all(scores >= 0) and np.all(scores <= 1)
        assert not np.any(np.isnan(scores))

    def test_scores_are_reasonable(self):
        """Test that scores are reasonable values in [0, 1]."""
        # Example with very uncertain predictions (close to 0.5)
        uncertain_probs = np.full((1, 3), 0.5)

        # Example with confident predictions
        confident_probs = np.array([[0.9, 0.9, 0.9]])

        labels = np.array([[[1, 1, 1], [1, 1, 1]]], dtype=float)

        uncertain_scores, _ = get_active_learning_scores(labels, uncertain_probs)
        confident_scores, _ = get_active_learning_scores(labels, confident_probs)

        # Both should be in valid range
        assert 0 <= uncertain_scores[0] <= 1
        assert 0 <= confident_scores[0] <= 1

        # With high-confidence predictions matching labels, scores should be relatively high
        assert confident_scores[0] > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
