import numpy as np
import pandas as pd
import pytest

import cleanlab.multiannotator as multiannotator


def make_inputs(dtype=np.float64, missing=False, models=3, classes=5):
    rng = np.random.default_rng(62)
    consensus = np.arange(28) % (classes - 1 if classes > 2 else classes)
    labels = np.tile(consensus[:, None], (1, 5)).astype(float)
    labels[:, 3:] = rng.integers(classes, size=(28, 2))
    if missing:
        labels[:, 1:][rng.random((28, 4)) < 0.4] = np.nan
        labels[:2] = consensus[:2, None]
        labels[2, 1:] = np.nan
    probabilities = rng.random((models, 28, classes))
    probabilities[:, np.arange(28), consensus] += 1.0
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    probabilities = probabilities.astype(dtype)
    counts = (~np.isnan(labels)).sum(axis=1)
    agreement = (labels == consensus[:, None]).sum(axis=1) / counts
    return labels, consensus, probabilities, counts, agreement


def scalar_posterior_reference(
    labels, probabilities, counts, agreement, weights, annotator_weights
):
    consensus_likelihood = np.mean(agreement[counts != 1])
    other_likelihood = (1 - consensus_likelihood) / (probabilities.shape[2] - 1)
    result = np.empty(probabilities.shape[1:], dtype=float)
    for row, annotations in enumerate(labels):
        present = ~np.isnan(annotations)
        row_weights = np.concatenate((weights, annotator_weights[present]))
        for label in range(probabilities.shape[2]):
            likelihoods = [probabilities[model, row, label] for model in range(len(probabilities))]
            likelihoods.extend(
                consensus_likelihood if annotation == label else other_likelihood
                for annotation in annotations[present]
            )
            result[row, label] = np.average(likelihoods, weights=row_weights)
    return result


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("models", [1, 3, 12])
def test_ensemble_posterior_matches_scalar_weighted_averages(dtype, missing, models):
    labels, consensus, probabilities, counts, agreement = make_inputs(dtype, missing, models)
    labels_before = labels.copy()
    probabilities_before = probabilities.copy()
    labels.setflags(write=False)
    probabilities.setflags(write=False)

    posterior, weights, annotator_weights = (
        multiannotator._get_post_pred_probs_and_weights_ensemble(
            labels, consensus, probabilities, counts, agreement, verbose=False
        )
    )
    expected = scalar_posterior_reference(
        labels, probabilities, counts, agreement, weights, annotator_weights
    )
    np.testing.assert_array_equal(posterior, expected)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=5e-4)
    np.testing.assert_array_equal(labels, labels_before)
    np.testing.assert_array_equal(probabilities, probabilities_before)


def test_ensemble_posterior_batches_classes_in_one_average_per_example(monkeypatch):
    inputs = make_inputs()
    original_average = np.average
    calls = []

    def average(*args, **kwargs):
        calls.append(1)
        return original_average(*args, **kwargs)

    monkeypatch.setattr(np, "average", average)
    multiannotator._get_post_pred_probs_and_weights_ensemble(*inputs, verbose=False)
    labels = inputs[0]
    # Each example needs one posterior reduction; annotator agreement uses one per annotator.
    assert len(calls) <= labels.shape[0] + labels.shape[1]


def test_equal_posterior_classes_keep_their_exact_tie():
    labels = np.array([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=float)
    consensus = np.array([0, 0, 1, 1])
    probabilities = np.full((3, 4, 2), 0.5)
    posterior, _, _ = multiannotator._get_post_pred_probs_and_weights_ensemble(
        labels, consensus, probabilities, np.full(4, 2), np.full(4, 0.5), verbose=False
    )
    np.testing.assert_array_equal(posterior, np.full((4, 2), 0.5))


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("models", [3, 12])
def test_public_ensemble_reports_match_scalar_reference(monkeypatch, missing, models):
    labels, _, probabilities, _, _ = make_inputs(missing=missing, models=models)
    labels = pd.DataFrame(
        labels,
        index=[f"example-{i}" for i in range(len(labels))],
        columns=[f"annotator-{i}" for i in range(labels.shape[1])],
    )
    actual = multiannotator.get_label_quality_multiannotator_ensemble(
        labels, probabilities, return_weights=True, verbose=False
    )
    original_helper = multiannotator._get_post_pred_probs_and_weights_ensemble

    def scalar_helper(
        labels_multiannotator,
        consensus_label,
        prior_pred_probs,
        num_annotations,
        annotator_agreement,
        **kwargs,
    ):
        _, weights, annotator_weights = original_helper(
            labels_multiannotator,
            consensus_label,
            prior_pred_probs,
            num_annotations,
            annotator_agreement,
            **kwargs,
        )
        posterior = scalar_posterior_reference(
            labels_multiannotator,
            prior_pred_probs,
            num_annotations,
            annotator_agreement,
            weights,
            annotator_weights,
        )
        return posterior, weights, annotator_weights

    monkeypatch.setattr(multiannotator, "_get_post_pred_probs_and_weights_ensemble", scalar_helper)
    expected = multiannotator.get_label_quality_multiannotator_ensemble(
        labels, probabilities, return_weights=True, verbose=False
    )
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], pd.DataFrame):
            pd.testing.assert_frame_equal(actual[key], expected[key], check_exact=True)
        else:
            np.testing.assert_array_equal(actual[key], expected[key])


def test_noncontiguous_predictions_keep_scalar_reduction_order():
    labels, consensus, probabilities, counts, agreement = make_inputs(missing=True, models=12)
    probabilities = probabilities.transpose(0, 2, 1).copy().transpose(0, 2, 1)
    assert not probabilities.flags.c_contiguous
    posterior, weights, annotator_weights = (
        multiannotator._get_post_pred_probs_and_weights_ensemble(
            labels, consensus, probabilities, counts, agreement, verbose=False
        )
    )
    expected = scalar_posterior_reference(
        labels, probabilities, counts, agreement, weights, annotator_weights
    )
    np.testing.assert_array_equal(posterior, expected)
