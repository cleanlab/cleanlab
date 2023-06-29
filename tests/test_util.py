# coding: utf-8

from cleanlab.internal import util
import numpy as np
import pytest

from cleanlab.internal.label_quality_utils import get_normalized_entropy
from cleanlab.internal.multilabel_utils import int2onehot, onehot2int
from cleanlab.internal.util import num_unique_classes, format_labels, get_missing_classes
from cleanlab.internal.validation import assert_valid_class_labels


noise_matrix = np.array([[1.0, 0.0, 0.2], [0.0, 0.7, 0.2], [0.0, 0.3, 0.6]])

noise_matrix_2 = np.array(
    [
        [1.0, 0.3],
        [0.0, 0.7],
    ]
)

joint_matrix = np.array([[0.1, 0.0, 0.1], [0.1, 0.1, 0.1], [0.2, 0.1, 0.2]])

joint_matrix_2 = np.array(
    [
        [0.2, 0.3],
        [0.4, 0.1],
    ]
)

single_element = np.array([1])


def test_print_inm():
    for m in [noise_matrix, noise_matrix_2, single_element]:
        util.print_inverse_noise_matrix(m, round_places=3)


def test_print_joint():
    for m in [joint_matrix, joint_matrix_2, single_element]:
        util.print_joint_matrix(m, round_places=3)


def test_print_square():
    for m in [noise_matrix, noise_matrix_2, single_element]:
        util.print_square_matrix(noise_matrix, round_places=3)


def test_print_noise_matrix():
    for m in [noise_matrix, noise_matrix_2, single_element]:
        util.print_noise_matrix(noise_matrix, round_places=3)


def test_pu_f1():
    s = [1, 1, 1, 0, 0, 0]
    p = [1, 1, 1, 0, 0, 0]
    assert abs(util.estimate_pu_f1(s, p) - 1) < 1e-4


def test_value_counts_str():
    r = util.value_counts(["a", "b", "a"])
    assert all(np.array([2, 1]) - r < 1e-4)


def test_pu_remove_noise():
    nm = np.array(
        [
            [0.9, 0.0, 0.0],
            [0.0, 0.7, 0.4],
            [0.1, 0.3, 0.6],
        ]
    )
    r = util.remove_noise_from_class(nm, 0)
    assert np.all(r - nm < 1e-4)


def test_round_preserving_sum():
    vec = np.array([1.1] * 10)
    ints = util.round_preserving_sum(vec)
    # Make sure one of ints is now 2 to preserve sum of 11
    assert np.any(ints == 2)
    assert sum(ints) == 11


def test_one_hot():
    num_classes = 4
    labels = [[0], [0, 1], [0, 1], [2], [0, 2, 3]]
    assert onehot2int(int2onehot(labels, K=num_classes)) == labels


def test_num_unique():
    labels = [[0], [0, 1], [0, 1], [2], [0, 2, 3]]
    assert num_unique_classes(labels) == 4


def test_missing_classes():
    labels = [0, 1]  # class 2 is missing
    pred_probs = np.array([[0.8, 0.1, 0.1], [0.4, 0.5, 0.1]])
    assert get_missing_classes(labels, pred_probs=pred_probs) == [2]


def test_round_preserving_row_totals():
    mat = np.array(
        [
            [1.7, 1.8, 1.5],
            [1.1, 1.4, 1.5],
            [1.3, 1.3, 1.4],
        ]
    )
    mat_int = util.round_preserving_row_totals(mat)
    # Check that row sums are preserved
    assert np.all(mat_int.sum(axis=1) == mat.sum(axis=1))


def test_confusion_matrix():
    true = [0, 1, 1, 2, 2, 2]
    pred = [0, 0, 1, 1, 1, 2]
    cmat = util.confusion_matrix(true, pred)
    assert np.shape(cmat) == (3, 3)
    assert cmat[0][0] == 1
    assert cmat[1][1] == 1
    assert cmat[2][2] == 1
    assert cmat[1][0] == 1
    assert cmat[2][1] == 2
    assert cmat[0][1] == 0
    assert cmat[0][2] == 0
    assert cmat[2][0] == 0
    assert cmat[0][1] == 0


def test_confusion_matrix_nonconsecutive():
    true = [-1, -1, -1, 1]
    pred = [1, 1, -1, 1]
    cmat = util.confusion_matrix(true, pred)
    assert np.shape(cmat) == (2, 2)
    assert cmat[0][0] == 1
    assert cmat[0][1] == 2
    assert cmat[1][0] == 0
    assert cmat[1][1] == 1


def test_format_labels():
    # test 1D labels
    str_labels = np.array(["b", "b", "a", "c", "a"])
    labels, label_map = format_labels(str_labels)
    assert all(labels == np.array([1, 1, 0, 2, 0]))
    assert label_map[0] == "a"
    assert label_map[1] == "b"
    assert label_map[2] == "c"
    assert_valid_class_labels(labels)


def test_normalized_entropy():
    """Check that normalized entropy is well well-behaved and in [0, 1]."""
    # test tiny numbers
    for dtype in [np.float16, np.float32, np.float64]:
        info = np.finfo(dtype)
        # some NumPy versions have bugs, therefore we provide a fallback
        # (fallback is the value of the smalles datatype float16)
        smallest_normal = getattr(info, "smallest_normal", 6.104e-05)
        smallest_subnormal = getattr(info, "smallest_subnormal", 6e-08)
        for val in [info.eps, smallest_normal, smallest_subnormal, 0]:
            entropy = get_normalized_entropy(np.array([[1.0, val]], dtype=dtype))
            assert 0.0 <= entropy <= 1.0
    # test multiple _assert_valid_inputs
    entropy = get_normalized_entropy(np.array([[0.0, 1.0], [0.5, 0.5]]))
    assert all((0.0 <= entropy) & (entropy <= 1.0))

    # raise errors for wrong probabilities.
    with pytest.raises(ValueError):
        get_normalized_entropy(np.array([[-1.0, 0.5]]))  # negative
        get_normalized_entropy(np.array([[2.0, 0.5]]))  # larger 1


def test_force_two_dimensions():
    # Test with 2D array
    X = np.zeros((5, 5))
    X_reshaped = util.force_two_dimensions(X)
    assert X_reshaped.shape == (5, 5), "The shape of 2D array should remain unchanged."

    # Test with 4D array
    X = np.zeros((5, 5, 5, 5))
    X_reshaped = util.force_two_dimensions(X)
    assert X_reshaped.shape == (
        5,
        125,
    ), "The shape of 4D array should be flattened to two dimensions."

    # Test with None input
    assert util.force_two_dimensions(None) is None, "None input should return None."
