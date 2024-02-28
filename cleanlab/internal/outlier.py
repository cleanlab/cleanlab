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
Helper functions used internally for outlier detection tasks.
"""

import numpy as np


def transform_distances_to_scores(
    avg_distances: np.ndarray, t: int, scaling_factor: float
) -> np.ndarray:
    """Returns an outlier score for each example based on its average distance to its k nearest neighbors.

    The transformation of a distance, :math:`d` , to a score, :math:`o` , is based on the following formula:

    .. math::
        o = \\exp\\left(-dt\\right)

    where :math:`t` scales the distance to a score in the range [0,1].

    Parameters
    ----------
    avg_distances : np.ndarray
        An array of distances of shape ``(N)``, where N is the number of examples.
        Each entry represents an example's average distance to its k nearest neighbors.

    t : int
        A sensitivity parameter that modulates the strength of the transformation from distances to scores.
        Higher values of `t` result in more pronounced differentiation between the scores of examples
        lying in the range [0,1].

    scaling_factor : float
        A scaling factor used to normalize the distances before they are converted into scores. A valid
        scaling factor is any positive number. The choice of scaling factor should be based on the
        distribution of distances between neighboring examples. A good rule of thumb is to set the
        scaling factor to the median distance between neighboring examples. A lower scaling factor
        results in more pronounced differentiation between the scores of examples lying in the range [0,1].

    Returns
    -------
    ood_features_scores : np.ndarray
        An array of outlier scores of shape ``(N,)`` for N examples.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.outlier import transform_distances_to_scores
    >>> distances = np.array([[0.0, 0.1, 0.25],
    ...                       [0.15, 0.2, 0.3]])
    >>> avg_distances = np.mean(distances, axis=1)
    >>> transform_distances_to_scores(avg_distances, t=1, scaling_factor=1)
    array([0.88988177, 0.80519832])
    """
    # Map ood_features_scores to range 0-1 with 0 = most concerning
    ood_features_scores: np.ndarray = np.exp(-1 * avg_distances / scaling_factor * t)
    return ood_features_scores
