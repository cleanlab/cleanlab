import pytest
import pandas as pd
from hypothesis import given, strategies as st

from cleanlab.datalab.internal.spurious_correlation import (
    SpuriousCorrelations,
    relative_room_for_improvement,
)


@pytest.fixture
def sample_data():
    """Sample data for testing spurious correlations."""
    _data = pd.DataFrame({"blurry_score": [0.5, 0.2, 0.8], "dark_score": [0.1, 0.5, 0.4]})
    _labels = [1, 0, 1]
    return {"data": _data, "labels": _labels}


@pytest.fixture
def spurious_instance(sample_data):
    return SpuriousCorrelations(**sample_data)


def test_initialization(spurious_instance):
    assert hasattr(spurious_instance, "data")
    assert hasattr(spurious_instance, "labels")
    assert hasattr(spurious_instance, "properties_of_interest")


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
