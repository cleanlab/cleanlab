import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st

from cleanlab.datalab.internal.spurious_correlation import (
    SpuriousCorrelations,
    relative_room_for_improvement,
)


def generate_correlated_feature(labels, correlation_coefficient):
    # Generate a random feature
    random_feature = np.random.randn(len(labels))
    # Create a correlated feature
    correlated_feature = (
        correlation_coefficient * labels + (1 - correlation_coefficient) * random_feature
    )
    return correlated_feature


@pytest.fixture
def sample_data():
    """Sample data for testing spurious correlations."""
    N = 100
    np.random.seed(42)
    _proportion = 0.5 + np.random.rand() * 0.2
    _labels = np.random.choice([0, 1], size=N, p=[_proportion, 1 - _proportion])

    _data = pd.DataFrame(
        {
            "blurry_correlated": generate_correlated_feature(_labels, 0.7),
            "dark_uncorrelated": generate_correlated_feature(_labels, 0.01),
        }
    )
    return {"data": _data, "labels": _labels}


@pytest.fixture
def spurious_instance(sample_data):
    return SpuriousCorrelations(**sample_data)


def test_initialization(spurious_instance):
    assert hasattr(spurious_instance, "data")
    assert hasattr(spurious_instance, "labels")
    assert hasattr(spurious_instance, "properties_of_interest")


def test_calculate_correlations(spurious_instance):
    """Test that scoring on arbitrary dataframes with property scores for columns works."""

    # Run main method
    data_scores = spurious_instance.calculate_correlations()

    # Check that the output dataframe reflects the input dataframe
    assert isinstance(data_scores, pd.DataFrame)
    assert len(data_scores) == 2
    assert set(data_scores.columns) == {"property", "label_prediction_error"}
    assert data_scores["property"].tolist() == ["blurry_correlated", "dark_uncorrelated"]

    # Check that the scores are replicable
    scores = data_scores["label_prediction_error"].tolist()
    np.testing.assert_almost_equal(scores, [0.100, 0.420], decimal=3)


@given(
    baseline_accuracy=st.floats(min_value=0, max_value=1),
    mean_accuracy=st.floats(min_value=0, max_value=1),
)
def test_relative_room_for_improvement(baseline_accuracy, mean_accuracy):
    score = relative_room_for_improvement(baseline_accuracy, mean_accuracy)

    # Property 1: Score is always between 0 and 1 inclusive
    assert 0 <= score <= 1

    # Property 2: If mean_accuracy is less than baseline_accuracy, score should be 1
    if mean_accuracy < baseline_accuracy:
        assert (
            score == 1
        ), f"score: {score} is not 1, baseline_accuracy: {baseline_accuracy}, mean_accuracy: {mean_accuracy}"

    # Property 3: If mean_accuracy is equal to 1, score should be 0
    if mean_accuracy == 1:
        assert (
            score == 0
        ), f"score: {score} is not 0, baseline_acc  uracy: {baseline_accuracy}, mean_accuracy: {mean_accuracy}"
