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
"""
Methods for quantifying the value of each data point in a Machine Learning dataset.
Data Valuation helps us assess individual training data points' contributions to a ML model's predictive performance.
"""


from typing import Callable, Optional, Union, cast

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


def _knn_shapley_score(knn_graph: csr_matrix, labels: np.ndarray, k: int) -> np.ndarray:
    """Compute the Shapley values of data points based on a knn graph."""
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


def _process_knn_graph_from_features(
    features: np.ndarray, metric: Optional[Union[str, Callable]], k: int = 10
) -> csr_matrix:
    """Calculate the knn graph from the features if it is not provided in the kwargs."""
    if k > len(features):  # Ensure number of neighbors less than number of examples
        raise ValueError(
            f"Number of nearest neighbors k={k} cannot exceed the number of examples N={len(features)} passed into the estimator (knn)."
        )
    if metric == None:
        metric = (
            "cosine"
            if features.shape[1] > 3
            else "euclidean" if features.shape[0] > 100 else euclidean
        )
    knn = NearestNeighbors(n_neighbors=k, metric=metric).fit(features)
    knn_graph = knn.kneighbors_graph(mode="distance")
    try:
        check_is_fitted(knn)
    except NotFittedError:
        knn.fit(features)
    return knn_graph


def data_shapley_knn(
    labels: np.ndarray,
    *,
    features: Optional[np.ndarray] = None,
    knn_graph: Optional[csr_matrix] = None,
    metric: Optional[Union[str, Callable]] = None,
    k: int = 10,
) -> np.ndarray:
    """
    Compute the Data Shapley values of data points using a K-Nearest Neighbors (KNN) graph.

    This function calculates the contribution (Data Shapley value) of each data point in a dataset
    for model training, either directly from data features or using a precomputed KNN graph.

    The examples in the dataset with lowest data valuation scores contribute least
    to a trained ML model’s performance (those whose value falls below a threshold are flagged with this type of issue).
    The data valuation score is an approximate Data Shapley value, calculated based on the labels of the top k nearest neighbors of an example. Details on this KNN-Shapley value can be found in these papers:
    https://arxiv.org/abs/1908.08619 and https://arxiv.org/abs/1911.07128.

    Parameters
    ----------
    labels :
        An array of labels for the data points(only for multi-class classification datasets).
    features :
        Feature embeddings (vector representations) of every example in the dataset.

            Necessary if `knn_graph` is not supplied.

            If provided, this must be a 2D array with shape (num_examples, num_features).
    knn_graph :
        A precomputed sparse KNN graph. If not provided, it will be computed from the `features` using the specified `metric`.
    metric : Optional[str or Callable], default=None
        The distance metric for KNN graph construction.
        Supports metrics available in ``sklearn.neighbors.NearestNeighbors``
        Default metric is ``"cosine"`` for ``dim(features) > 3``, otherwise ``"euclidean"`` for lower-dimensional data.
        The euclidean is computed with an efficient implementation from scikit-learn when the number of examples is greater than 100.
        When the number of examples is 100 or fewer, a more numerically stable version of the euclidean distance from scipy is used.
    k :
        The number of neighbors to consider for the KNN graph and Data Shapley value computation.
        Must be less than the total number of data points.
        The value may not exceed the number of neighbors of each data point stored in the KNN graph.

    Returns
    -------
    scores :
        An array of transformed Data Shapley values for each data point, calibrated to indicate their relative importance.
        These scores have been adjusted to fall within 0 to 1.
        Values closer to 1 indicate data points that are highly influential and positively contribute to a trained ML model's performance.
        Conversely, scores below 0.5 indicate data points estimated to  negatively impact model performance.

    Raises
    ------
    ValueError
        If neither `knn_graph` nor `features` are provided, or if `k` is larger than the number of examples in `features`.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.data_valuation import data_shapley_knn
    >>> labels = np.array([0, 1, 0, 1, 0])
    >>> features = np.array([[0, 1, 2, 3, 4]]).T
    >>> data_shapley_knn(labels=labels, features=features, k=4)
    array([0.55 , 0.525, 0.55 , 0.525, 0.55 ])
    """
    if knn_graph is None and features is None:
        raise ValueError("Either knn_graph or features must be provided.")

    if knn_graph is None:
        knn_graph = _process_knn_graph_from_features(cast(np.ndarray, features), metric, k)
    return _knn_shapley_score(knn_graph, labels, k)
