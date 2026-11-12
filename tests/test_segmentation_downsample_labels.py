import numpy as np
import pytest

from cleanlab.experimental.label_issues_batched import LabelInspector
from cleanlab.segmentation.filter import find_label_issues


def make_block_dataset(factor):
    majority = np.tile(np.array([[0, 1, 2], [3, 0, 2]]), (4, 1, 1))
    labels = majority.repeat(factor, axis=1).repeat(factor, axis=2)
    for offset in range((factor * factor - 1) // 2):
        row, column = divmod(offset, factor)
        labels[:, row::factor, column::factor] = (majority + 3) % 4

    true_classes = majority.copy()
    true_classes[-1] = (true_classes[-1] + 1) % 4
    probabilities = np.full((4, 4, 2, 3), 0.05)
    np.put_along_axis(probabilities, true_classes[:, None], 0.85, axis=1)
    full_probabilities = probabilities.repeat(factor, axis=2).repeat(factor, axis=3)
    return labels, full_probabilities, majority, probabilities


@pytest.mark.parametrize("factor", [2, 3])
@pytest.mark.parametrize("batch_size", [1, 1000])
def test_downsampling_uses_the_majority_class_of_each_block(factor, batch_size):
    labels, probabilities, majority, coarse_probabilities = make_block_dataset(factor)
    original_labels = labels.copy()
    original_probabilities = probabilities.copy()
    coarse_issues = find_label_issues(
        majority, coarse_probabilities, n_jobs=1, verbose=False, batch_size=batch_size
    )
    expected = coarse_issues.repeat(factor, axis=1).repeat(factor, axis=2)
    expected &= probabilities.argmax(axis=1) != labels
    assert not expected[:-1].any()
    assert expected[-1].all()

    actual = find_label_issues(
        labels,
        probabilities,
        downsample=factor,
        n_jobs=1,
        verbose=False,
        batch_size=batch_size,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(labels, original_labels)
    np.testing.assert_array_equal(probabilities, original_probabilities)


@pytest.mark.parametrize("factor", [2, 3])
@pytest.mark.parametrize("permutation", [np.array([3, 1, 0, 2]), np.array([1, 2, 3, 0])])
def test_downsampled_issues_do_not_depend_on_class_numbering(factor, permutation):
    labels, probabilities, _, _ = make_block_dataset(factor)
    expected = find_label_issues(labels, probabilities, downsample=factor, n_jobs=1, verbose=False)

    actual = find_label_issues(
        permutation[labels],
        probabilities[:, np.argsort(permutation)],
        downsample=factor,
        n_jobs=1,
        verbose=False,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.int64, np.float64])
def test_downsampling_ties_choose_an_existing_smallest_class(monkeypatch, dtype):
    labels = np.array([[[0, 3], [0, 3]], [[1, 3], [3, 1]], [[2, 3], [3, 2]]], dtype=dtype)
    probabilities = np.full((3, 4, 2, 2), 0.25)
    observed = []
    original_update = LabelInspector.update_confident_thresholds

    def record_labels(self, labels, pred_probs):
        observed.extend(labels.tolist())
        return original_update(self, labels, pred_probs)

    monkeypatch.setattr(LabelInspector, "update_confident_thresholds", record_labels)

    find_label_issues(labels, probabilities, downsample=2, n_jobs=1, verbose=False)

    assert observed == [0, 1, 2]


def test_no_downsampling_preserves_the_supplied_pixel_labels(monkeypatch):
    labels, probabilities, _, _ = make_block_dataset(2)
    observed = []
    original_update = LabelInspector.update_confident_thresholds

    def record_labels(self, labels, pred_probs):
        observed.extend(labels.tolist())
        return original_update(self, labels, pred_probs)

    monkeypatch.setattr(LabelInspector, "update_confident_thresholds", record_labels)

    find_label_issues(labels, probabilities, downsample=1, n_jobs=1, verbose=False)

    np.testing.assert_array_equal(observed, labels.ravel())
