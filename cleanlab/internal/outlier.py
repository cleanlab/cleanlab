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


def transform_distances_to_scores(distances: np.ndarray, k: int, t: int) -> np.ndarray:
    """Returns an outlier score for each example based on its average distance to its k nearest neighbors.

    The transformation of a distance, :math:`d` , to a score, :math:`o` , is based on the following formula:

    .. math::
        o = \\exp\\left(-dt\\right)

    where :math:`t` scales the distance to a score in the range [0,1].

    Parameters
    ----------
    distances : np.ndarray
        An array of distances of shape ``(N, num_neighbors)``, where N is the number of examples.
        Each row contains the distances to each example's `num_neighbors` nearest neighbors.
        It is assumed that each row is sorted in ascending order.

    k : int
        Number of neighbors used to compute the average distance to each example.
        This assumes that the second dimension of distances is k or greater, but it
        uses slicing to avoid indexing errors.

    t : int
        Controls transformation of distances between examples into similarity scores that lie in [0,1].

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
    >>> transform_distances_to_scores(distances, k=2, t=1)
    array([0.95122942, 0.83945702])
    """
    # Calculate average distance to k-nearest neighbors
    avg_knn_distances = distances[:, :k].mean(axis=1)

    # Map ood_features_scores to range 0-1 with 0 = most concerning
    ood_features_scores: np.ndarray = np.exp(-1 * avg_knn_distances * t)
    return ood_features_scores
