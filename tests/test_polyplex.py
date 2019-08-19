
# coding: utf-8

from __future__ import print_function, absolute_import, division, unicode_literals, with_statement


from cleanlab.polyplex import joint_bounds, slope_intercept, joint_min_max
import numpy as np
import pytest


def test_polyplex(py = [0.1, 0.2, 0.2, 0.5]):
    K = len(py)
    x, y_min, y_max = joint_bounds(py)
    xs = range(1, K+1)
    min_slopes = [slope_intercept(*list(zip(x, y_min))[i-1:i+1])[0] for i in xs]
    max_slopes = [slope_intercept(*list(zip(x, y_max))[i-1:i+1])[0] for i in xs]
    polyplex = [joint_min_max(trace, py) for trace in range(K+1)]
    mins, maxs = [np.array(z) for z in zip(*polyplex)]
    # Slope_intercept and joint_min_max produce consistent results
    assert(all((np.ediff1d(mins) - min_slopes) < 1e-2))
    assert(all((np.ediff1d(maxs) - max_slopes) < 1e-2))


def test_joint_min_max_float(py = [0.1, 0.1, 0.3, 0.5]):
    vals = np.array((joint_min_max(1, py), joint_min_max(2, py)))
    averages = vals.mean(axis = 0)
    assert(all((averages - joint_min_max(1.5, py)) < 1e-4))