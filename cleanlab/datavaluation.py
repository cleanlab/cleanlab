# Copyright (C) 2017-2024  Cleanlab Inc.
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

import numpy as np
from scipy.sparse import csr_matrix


def data_shapley_knn(
    knn_graph: csr_matrix,
    labels: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Compute the Shapley values of data points based on a knn graph.
    Based on KNN-Shapley value described in https://arxiv.org/abs/1911.07128
    The larger the score, the more valuable the data point is, the more contribution it will make to the model's training.

    Parameters
    ----------
    knn_graph : csr_matrix
        A sparse matrix representing the knn graph.
    labels: np.ndarray
        The labels of the data points.
    k: int
        The number of nearest neighbors to consider.
    """
    N = labels.shape[0]
    scores = np.zeros((N, N))
    dist = knn_graph.indices.reshape(N, -1)

    for y, s, dist_i in zip(labels, scores, dist):
        idx = dist_i[::-1]
        ans = labels[idx]
        s[idx[k - 1]] = float(ans[k - 1] == y)
        ans_matches = (ans == y).flatten()
        for j in range(k - 2, -1, -1):
            s[idx[j]] = s[idx[j + 1]] + float(int(ans_matches[j]) - int(ans_matches[j + 1]))
    return 0.5 * (np.mean(scores / k, axis=0) + 1)
