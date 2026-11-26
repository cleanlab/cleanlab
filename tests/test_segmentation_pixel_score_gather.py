import numpy as np
import pytest

from cleanlab.segmentation.filter import find_label_issues
from cleanlab.segmentation.rank import get_label_quality_scores


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("downsample", [1, 2])
def test_pixel_issue_scores_preserve_labeled_probabilities(dtype, strided, downsample):
    rng = np.random.default_rng(17)
    probabilities = rng.random((2, 3, 4, 6)).astype(dtype)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    labels = probabilities.argmax(axis=1)
    labels[:, 0, :] = (labels[:, 0, :] + 1) % 3
    if strided:
        probabilities = probabilities[:, :, ::-1, ::-1]
        labels = labels[:, ::-1, ::-1]
    probabilities.flags.writeable = False
    labels.flags.writeable = False
    original_probabilities, original_labels = probabilities.copy(), labels.copy()

    image_scores, pixel_scores = get_label_quality_scores(
        labels,
        probabilities,
        method="num_pixel_issues",
        downsample=downsample,
        n_jobs=1,
        verbose=False,
    )

    expected_pixels = np.empty(labels.shape, dtype=dtype)
    for image, row, column in np.ndindex(labels.shape):
        expected_pixels[image, row, column] = probabilities[
            image, labels[image, row, column], row, column
        ]
    np.testing.assert_array_equal(pixel_scores, expected_pixels)
    assert pixel_scores.dtype == dtype
    issue_mask = find_label_issues(
        labels, probabilities, downsample=downsample, n_jobs=1, verbose=False
    )
    np.testing.assert_array_equal(image_scores, 1 - issue_mask.mean(axis=(1, 2)))
    np.testing.assert_array_equal(probabilities, original_probabilities)
    np.testing.assert_array_equal(labels, original_labels)


def test_pixel_issue_scores_keep_integer_valued_float_labels():
    labels = np.array([[[0.0, 1.0], [1.0, 0.0]]])
    probabilities = np.array([[[[0.9, 0.3], [0.6, 0.8]], [[0.1, 0.7], [0.4, 0.2]]]])
    _, scores = get_label_quality_scores(
        labels, probabilities, method="num_pixel_issues", n_jobs=1, verbose=False
    )
    np.testing.assert_array_equal(scores, [[[0.9, 0.7], [0.4, 0.8]]])
