import numpy as np
import pytest
from scipy.special import softmax as reference_softmax

from cleanlab.internal.multilabel_scorer import softmin
from cleanlab.internal.numerics import softmax
from cleanlab.internal.object_detection_utils import softmin1d
from cleanlab.token_classification.rank import get_label_quality_scores


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_low_temperature_softmax_stays_normalized_for_stored_scores(dtype, axis):
    values = np.array([[0.2, 0.3, 0.4], [0.5, 0.4, 0.3]], dtype=dtype)
    original = values.copy()
    expected = reference_softmax(values.astype(np.float64) / 1e-6, axis=axis)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = softmax(values, temperature=1e-6, axis=axis, shift=True)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual.sum(axis=axis), 1.0)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("dtype,magnitude", [(np.float32, 3e38), (np.float64, 1e308)])
@pytest.mark.parametrize("axis", [None, 1])
def test_shift_removes_large_common_offset_before_temperature_scaling(dtype, magnitude, axis):
    values = np.full((2, 3), magnitude, dtype=dtype)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = softmax(values, temperature=1e-6, axis=axis, shift=True)

    expected = np.full((2, 3), 1 / (6 if axis is None else 3))
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
def test_shifting_integer_inputs_does_not_wrap_coordinate_differences(dtype):
    bounds = np.iinfo(dtype)
    values = np.array([bounds.min, bounds.max], dtype=dtype)

    actual = softmax(values, shift=True)

    np.testing.assert_allclose(actual, reference_softmax(values.astype(np.float64)), atol=1e-40)


def test_low_temperature_softmin_aggregators_preserve_half_precision_scores():
    scores = np.array([0.2, 0.3, 0.4], dtype=np.float16)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        object_score = softmin1d(scores, temperature=1e-6)
        multilabel_score = softmin(scores[None, :], temperature=1e-6)

    np.testing.assert_allclose(object_score, float(scores.min()))
    np.testing.assert_allclose(multilabel_score, [float(scores.min())])


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_token_quality_scores_remain_finite_at_low_softmin_temperatures(dtype):
    probabilities = [np.array([[0.3, 0.7], [0.8, 0.2]], dtype=dtype)]

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        sentence_scores, token_scores = get_label_quality_scores(
            [[0, 0]],
            probabilities,
            sentence_score_method="softmin",
            sentence_score_kwargs={"temperature": 1e-6},
        )

    np.testing.assert_allclose(sentence_scores, [float(probabilities[0][0, 0])])
    np.testing.assert_allclose(token_scores[0], probabilities[0][:, 0])
