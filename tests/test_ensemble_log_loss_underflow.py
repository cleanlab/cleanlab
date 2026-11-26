import numpy as np
import pytest

from cleanlab.rank import get_label_quality_ensemble_scores


def _constant_confidence_predictions(num_classes, confidence):
    probabilities = np.full((num_classes, num_classes), (1 - confidence) / (num_classes - 1))
    np.fill_diagonal(probabilities, confidence)
    return probabilities


@pytest.mark.parametrize("num_classes", [50, 100, 200])
def test_default_log_loss_search_handles_uniform_multiclass_predictions(num_classes):
    labels = np.arange(num_classes)
    probabilities = np.full((num_classes, num_classes), 1 / num_classes)
    probabilities.flags.writeable = False

    scores = get_label_quality_ensemble_scores(
        labels,
        [probabilities, probabilities],
        weight_ensemble_members_by="log_loss_search",
        verbose=False,
    )

    np.testing.assert_allclose(scores, 1 / num_classes)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("temperature", [1.0, 200.0, 1000.0])
@pytest.mark.parametrize("reverse_models", [False, True])
def test_log_loss_search_retains_relative_model_weights(temperature, reverse_models):
    labels = np.arange(100)
    weaker = _constant_confidence_predictions(100, 0.01)
    stronger = _constant_confidence_predictions(100, 0.02)
    probabilities = [weaker, stronger]
    if reverse_models:
        probabilities.reverse()
    originals = [p.copy() for p in probabilities]

    scores = get_label_quality_ensemble_scores(
        labels,
        probabilities,
        weight_ensemble_members_by="log_loss_search",
        log_loss_search_T_values=[temperature],
        verbose=False,
    )

    # Their likelihood ratio is exactly 1:2 for every labeled example.
    weaker_weight = 0.5**temperature / (1 + 0.5**temperature)
    expected = 0.01 * weaker_weight + 0.02 * (1 - weaker_weight)
    np.testing.assert_allclose(scores, expected)
    assert np.isfinite(scores).all()
    for probability, original in zip(probabilities, originals):
        np.testing.assert_array_equal(probability, original)
