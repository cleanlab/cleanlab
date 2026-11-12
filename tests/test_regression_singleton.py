import numpy as np
import pandas as pd
import pytest

from cleanlab.regression.rank import get_label_quality_scores


@pytest.mark.parametrize(
    "container",
    [
        list,
        np.asarray,
        lambda x: np.array([x]),
        lambda x: np.array(x)[:, None],
        pd.Series,
        pd.DataFrame,
    ],
    ids=["list", "array", "row", "column", "series", "dataframe"],
)
@pytest.mark.parametrize("residual", [0.0, 1.25])
def test_single_example_residual_scores_keep_the_sample_axis(container, residual):
    labels = container([2.5])
    predictions = container([2.5 + residual])

    scores = get_label_quality_scores(labels, predictions, method="residual")

    assert scores.shape == (1,)
    np.testing.assert_allclose(scores, [np.exp(-residual)])


def test_scoring_examples_one_at_a_time_matches_the_complete_batch():
    labels = np.array([1.0, 3.0, -2.0])
    predictions = np.array([1.5, 2.0, -2.0])
    expected = get_label_quality_scores(labels, predictions, method="residual")

    actual = np.concatenate(
        [
            get_label_quality_scores([label], [prediction], method="residual")
            for label, prediction in zip(labels, predictions)
        ]
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("scalar", [2.5, np.array(2.5)])
def test_scalar_inputs_are_still_rejected(scalar):
    with pytest.raises(ValueError, match="1-Dimensional"):
        get_label_quality_scores(scalar, scalar, method="residual")


@pytest.mark.parametrize("invalid", [np.ones((2, 2)), np.ones((2, 1, 2))])
def test_nonsingleton_extra_axes_are_still_rejected(invalid):
    with pytest.raises(ValueError, match="1-Dimensional"):
        get_label_quality_scores(invalid, invalid, method="residual")


def test_empty_inputs_are_still_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        get_label_quality_scores([], [], method="residual")
