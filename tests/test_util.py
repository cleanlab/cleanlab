# coding: utf-8

from cleanlab.internal import util
import numpy as np


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
