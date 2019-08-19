#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, absolute_import, division, unicode_literals, with_statement


from cleanlab import latent_algebra, latent_estimation
import numpy as np
import pytest


s = [0] * 10 + [1] * 5 + [2] * 15
nm = np.array([
    [1.0, 0.0, 0.2],
    [0.0, 0.7, 0.2],
    [0.0, 0.3, 0.6]
])


def test_latent_py_ps_inv():
    ps, py, inv = latent_algebra.compute_ps_py_inv_noise_matrix(s, nm)
    assert(all(abs(np.dot(inv, ps) - py) < 1e-3))
    assert(all(abs(np.dot(nm, py) - ps) < 1e-3))
    return ps, py, inv


def test_latent_inv():
    ps, py, inv = test_latent_py_ps_inv()
    inv2 = latent_algebra.compute_inv_noise_matrix(py, nm)
    assert(np.all(abs(inv - inv2) < 1e-3))


def test_latent_nm():
    ps, py, inv = test_latent_py_ps_inv()
    nm2 = latent_algebra.compute_noise_matrix_from_inverse(ps, inv, py)
    assert(np.all(abs(nm - nm2) < 1e-3))


def test_latent_py():
    ps, py, inv = test_latent_py_ps_inv()
    py2 = latent_algebra.compute_py(ps, nm, inv)
    assert(np.all(abs(py - py2) < 1e-3))


def test_latent_py_warning():
    ps, py, inv = test_latent_py_ps_inv()
    with pytest.raises(TypeError) as e:
        with pytest.warns(UserWarning) as w:
            py2 = latent_algebra.compute_py(
                ps = np.array([[[0.1, 0.3, 0.6]]]),
                noise_matrix = nm,
                inverse_noise_matrix = inv,
            )
            py2 = latent_algebra.compute_py(
                ps = np.array([[0.1], [0.2], [0.7]]),
                noise_matrix = nm,
                inverse_noise_matrix = inv,
            )
            assert(True)


def test_compute_py_err():
    ps, py, inv = test_latent_py_ps_inv()
    try:
        py = latent_algebra.compute_py(
            ps = ps,
            noise_matrix = nm,
            inverse_noise_matrix = inv,
            py_method = 'marginal_ps',
        )
    except ValueError as e:
        assert('y_count' in str(e))
        with pytest.raises(ValueError) as e:
            py = latent_algebra.compute_py(
                ps = ps,
                noise_matrix = nm,
                inverse_noise_matrix = inv,
                py_method = 'marginal_ps',
            )


def test_compute_py_marginal_ps():
    ps, py, inv = test_latent_py_ps_inv()
    cj = nm * ps * len(s)
    y_count = cj.sum(axis = 0)
    py2 = latent_algebra.compute_py(
        ps = ps,
        noise_matrix = nm,
        inverse_noise_matrix = inv,
        py_method = 'marginal_ps',
        y_count = y_count
    )
    assert(all(abs(py - py2) < 1e-2))


def test_pyx():
    psx = np.array([
        [0.1, 0.3, 0.6],
        [0.1, 0.0, 0.9],
        [0.1, 0.0, 0.9],
        [1.0, 0.0, 0.0],
        [0.1, 0.8, 0.1],
    ])
    ps, py, inv = test_latent_py_ps_inv()
    pyx = latent_algebra.compute_pyx(psx, nm, inv)
    assert(np.all(np.sum(pyx, axis = 1) - 1 < 1e-4))


def test_pyx_error():  
    psx = np.array([0.1, 0.3, 0.6])
    ps, py, inv = test_latent_py_ps_inv()
    try:
        pyx = latent_algebra.compute_pyx(psx, nm, inv)
    except ValueError as e:
        assert('should be (N, K)' in str(e))
    with pytest.raises(ValueError) as e:
        pyx = latent_algebra.compute_pyx(psx, nm, inv)