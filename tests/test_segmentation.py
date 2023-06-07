# Copyright (C) 2017-2023  Cleanlab Inc.
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with cleanlab.  If not, see <https://www.gnu.org/licenses/>.

"""
Scripts to test cleanlab.segmentation package
"""
import numpy as np
import os
import numpy as np
import random

np.random.seed(0)
import pytest
import matplotlib.pyplot as plt

# Filter
from cleanlab.segmentation.filter import (
    find_label_issues,
    # _check_input,
)

# Rank
from cleanlab.segmentation.rank import (
    get_label_quality_scores,
    issues_from_scores,
    # _get_label_quality_per_image,
)

# Summary
from cleanlab.segmentation.summary import (
    display_issues,
    common_label_issues,
    filter_by_class,
    # _generate_colormap,
)


def generate_three_image_dataset(bad_index):
    good_gt = np.zeros((10, 10))
    good_gt[:5, :] = 1.0
    bad_gt = np.ones((10, 10))
    bad_gt[:5, :] = 0.0
    good_pr = np.random.random((2, 10, 10))
    good_pr[0, :5, :] = good_pr[0, :5, :] / 10
    good_pr[1, 5:, :] = good_pr[1, 5:, :] / 10

    val = np.binary_repr([4, 2, 1][bad_index], width=3)
    error = [int(case) for case in val]

    labels = []
    pred = []
    for case in val:
        if case == "0":
            labels.append(good_gt)
            pred.append(good_pr)
        else:
            labels.append(bad_gt)
            pred.append(good_pr)

    labels = np.array(labels)
    pred_probs = np.array(pred)
    return labels, pred_probs, error


labels, pred_probs, error = generate_three_image_dataset(random.randint(0, 2))


def test_find_label_issues():
    issues = find_label_issues(labels, pred_probs, downsample=1, n_jobs=None, batch_size=1000)
    assert np.argmax(error) == np.argmax(issues.sum((1, 2)))

    issues = find_label_issues(labels, pred_probs, downsample=2, batch_size=1000)
    assert np.argmax(error) == np.argmax(issues.sum((1, 2)))

    issues = find_label_issues(labels, pred_probs, downsample=5, n_jobs=None, batch_size=1000)
    assert np.argmax(error) == np.argmax(issues.sum((1, 2)))


# def test_get_label_quality_per_image():
#     rand_pixel_scores = np.random.rand(
#         100,
#     )
#     _get_label_quality_per_image(rand_pixel_scores, method="softmin", temperature=0.1)

#     with pytest.raises(Exception) as e:
#         _get_label_quality_per_image(rand_pixel_scores)


# # test_find_label_issues()
# test_get_label_quality_per_image()
