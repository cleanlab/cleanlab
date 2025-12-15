"""
Methods for quantifying the value of each data point in a Machine Learning dataset.
Data Valuation helps us assess individual training data points' contributions to a ML model's predictive performance.
"""

from typing import Callable, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix

from cleanlab.internal.neighbor.knn_graph import create_knn_graph_and_index


def _knn_shapley_score(neighbor_indices: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Compute the Data Shapley values of data points using neighbor indices in a K-Nearest Neighbors (KNN) graph.

    This function leverages equations (18) and (19) from the paper available at https://arxiv.org/abs/1908.08619
    for computational efficiency.

    Parameters
    ----------
    neighbor_indices :
        A 2D array where each row contains the indices of the k-nearest neighbors for each data point.
    y :
        A 1D array of target values corresponding to the data points.
    k :
        The number of nearest neighbors to consider for each data point.

    Notes
    -----
    - The training set is used as its own test set for the KNN-Shapley value computation, meaning y_test is the same as y_train.
    - `neighbor_indices` are assumed to be pre-sorted by distance, with the nearest neighbors appearing first, and with at least `k` neighbors.
    - Unlike the referenced paper, this implementation does not account for an upper error bound epsilon.
      Consequently, K* is treated as equal to K instead of K* = max(K, 1/epsilon).
        - This simplification implies that the term min(K, j + 1) will always be j + 1, which is offset by the
          corresponding denominator term in the inner loop.
        - Dividing by K in the end achieves the same result as dividing by K* in the paper.
    - The pre-allocated `scores` array incorporates equation (18) for j = k - 1, ensuring efficient computation.
    """
    N = y.shape[0]
    scores = np.zeros((N, N))

    for y_alpha, s_alpha, idx in zip(y, scores, neighbor_indices):
        y_neighbors = y[idx]
        ans_matches = (y_neighbors == y_alpha).flatten()
        for j in range(k - 2, -1, -1):
            s_alpha[idx[j]] = s_alpha[idx[j + 1]] + float(
                int(ans_matches[j]) - int(ans_matches[j + 1])
            )
    return np.mean(scores / k, axis=0)


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
        Conversely, scores below 0.5 indicate data points estimated to negatively impact model performance.

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

    # Use provided knn_graph or compute it from features
    if knn_graph is None:
        knn_graph, _ = create_knn_graph_and_index(features, n_neighbors=k, metric=metric)

    num_examples = labels.shape[0]
    distances = knn_graph.indices.reshape(num_examples, -1)
    scores = _knn_shapley_score(neighbor_indices=distances, y=labels, k=k)
    return 0.5 * (scores + 1)
