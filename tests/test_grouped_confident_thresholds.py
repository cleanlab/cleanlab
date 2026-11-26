import numpy as np
import pandas as pd
import pytest

from cleanlab.count import compute_confident_joint, get_confident_thresholds
from cleanlab.internal.constants import CONFIDENT_THRESHOLDS_LOWER_BOUND


def _reference_thresholds(labels, probabilities):
    return np.clip(
        [
            probabilities[labels == label, label].mean() if np.any(labels == label) else 2
            for label in range(probabilities.shape[1])
        ],
        CONFIDENT_THRESHOLDS_LOWER_BOUND,
        None,
    )


@pytest.mark.parametrize("probability_dtype", [np.float16, np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize("label_dtype", [np.int32, np.uint64, np.float64])
@pytest.mark.parametrize("missing_classes", [False, True])
def test_grouping_preserves_class_means_exactly(probability_dtype, label_dtype, missing_classes):
    rng = np.random.default_rng(712)
    probabilities = rng.random((300, 7)).astype(probability_dtype)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    labels = (np.arange(len(probabilities)) % 7).astype(label_dtype)
    if missing_classes:
        labels[labels == 2] = 3
        labels[labels == 6] = 0
    labels = labels[::-1]
    probabilities = probabilities[::-1]
    labels.setflags(write=False)
    probabilities.setflags(write=False)
    original_labels, original_probabilities = labels.copy(), probabilities.copy()

    actual = get_confident_thresholds(labels, probabilities)
    expected = _reference_thresholds(labels, probabilities)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(labels, original_labels)
    np.testing.assert_array_equal(probabilities, original_probabilities)


@pytest.mark.parametrize("label_container", [list, pd.Series, pd.DataFrame])
def test_grouping_keeps_supported_label_containers(label_container):
    labels = [0, 2, 0, 2]
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.6, 0.2, 0.2], [0.2, 0.3, 0.5]])
    actual = get_confident_thresholds(label_container(labels), probabilities)
    np.testing.assert_array_equal(actual, [0.7, 2.0, 0.6])


def test_grouping_keeps_lower_bound_and_unseen_classes():
    probabilities = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(
        get_confident_thresholds([0, 1], probabilities),
        [CONFIDENT_THRESHOLDS_LOWER_BOUND, CONFIDENT_THRESHOLDS_LOWER_BOUND, 2.0],
    )
    np.testing.assert_array_equal(get_confident_thresholds([], np.empty((0, 3))), [2, 2, 2])


def test_multilabel_thresholds_preserve_one_vs_rest_means():
    labels = [[0], [0, 2], [], [2]]
    probabilities = np.array([[0.9, 0.2, 0.3], [0.8, 0.4, 0.7], [0.1, 0.3, 0.2], [0.2, 0.1, 0.6]])
    expected = []
    for label in range(probabilities.shape[1]):
        binary_labels = np.array([label in example for example in labels], dtype=int)
        binary_probabilities = np.column_stack(
            (1 - probabilities[:, label], probabilities[:, label])
        )
        expected.append(_reference_thresholds(binary_labels, binary_probabilities))

    np.testing.assert_array_equal(
        get_confident_thresholds(labels, probabilities, multi_label=True), expected
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_confident_joint_matches_explicit_reference_thresholds(dtype):
    rng = np.random.default_rng(89)
    labels = rng.choice([0, 2, 3, 5], size=200)
    probabilities = rng.random((len(labels), 7)).astype(dtype)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    thresholds = _reference_thresholds(labels, probabilities)

    expected = compute_confident_joint(labels, probabilities, thresholds=thresholds)
    actual = compute_confident_joint(labels, probabilities)

    np.testing.assert_array_equal(actual, expected)
