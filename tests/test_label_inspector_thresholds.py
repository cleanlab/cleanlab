import numpy as np
import pytest

from cleanlab.count import get_confident_thresholds, num_label_issues
from cleanlab.experimental.label_issues_batched import LabelInspector, find_label_issues_batched


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_thresholds_average_raw_probabilities_before_clipping(dtype, batch_size):
    labels = np.tile(np.array([0, 1], dtype=np.uint64), 4)
    self_confidence = np.array([0, 0, 1e-5, 1e-5, 0, 0, 2e-5, 2e-5], dtype=dtype)
    probabilities = np.column_stack((1 - self_confidence, self_confidence))
    probabilities[labels == 0] = probabilities[labels == 0, ::-1]
    inspector = LabelInspector(num_class=2, n_jobs=1, verbose=False)

    for start in range(0, len(labels), batch_size):
        inspector.update_confident_thresholds(
            labels[start : start + batch_size], probabilities[start : start + batch_size]
        )

    np.testing.assert_allclose(inspector.get_confident_thresholds(), [7.5e-6, 7.5e-6], rtol=1e-7)
    np.testing.assert_allclose(
        inspector.get_confident_thresholds(),
        get_confident_thresholds(labels, probabilities),
        rtol=1e-7,
    )
    np.testing.assert_array_equal(inspector.examples_per_class, [4, 4])
    assert inspector.examples_processed_thresh == 8


def test_missing_class_thresholds_are_replaced_when_that_class_arrives():
    labels = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.6, 0.2, 0.2],
            [0.1, 0.7, 0.2],
            [0.2, 0.5, 0.3],
            [0.1, 0.1, 0.8],
            [0.3, 0.3, 0.4],
        ]
    )
    inspector = LabelInspector(num_class=3, n_jobs=1, verbose=False)

    for end in (2, 4, 6):
        inspector.update_confident_thresholds(labels[end - 2 : end], probabilities[end - 2 : end])
        np.testing.assert_allclose(
            inspector.get_confident_thresholds(),
            get_confident_thresholds(labels[:end], probabilities[:end]),
        )
    np.testing.assert_allclose(inspector.get_confident_thresholds(), [0.7, 0.6, 0.6])


@pytest.mark.parametrize("calibrated", [False, True])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_unseen_classes_do_not_create_spurious_confident_label_issues(calibrated, n_jobs):
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array(
        [[0.1, 0.1, 0.8], [0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.1, 0.2, 0.7]]
    )
    options = {"estimation_method": "off_diagonal_calibrated"} if calibrated else {}
    inspector = LabelInspector(num_class=3, n_jobs=n_jobs, verbose=False, num_issue_kwargs=options)
    inspector.update_confident_thresholds(labels, probabilities)

    inspector.score_label_quality(labels, probabilities)

    assert inspector.get_num_issues() == num_label_issues(labels, probabilities, **options) == 0
    assert inspector.get_label_issues().size == 0
    np.testing.assert_allclose(inspector.get_confident_thresholds(), [0.5, 0.55, 2.0])


@pytest.mark.parametrize("batch_size", [1, 2, 100])
def test_public_batched_filter_keeps_unseen_class_predictions_unconfident(batch_size):
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array(
        [[0.1, 0.1, 0.8], [0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.1, 0.2, 0.7]]
    )

    issues = find_label_issues_batched(
        labels, probabilities, batch_size=batch_size, n_jobs=1, verbose=False
    )

    assert issues.size == 0
