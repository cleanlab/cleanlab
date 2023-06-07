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

from cleanlab.segmentation.rank import _get_label_quality_per_image
import pytest


def test_get_label_quality_per_image():
    rand_pixel_scores = np.random.rand(
        100,
    )
    _get_label_quality_per_image(rand_pixel_scores, method="softmin", temperature=0.1)

    with pytest.raises(Exception) as e:
        _get_label_quality_per_image(rand_pixel_scores)
