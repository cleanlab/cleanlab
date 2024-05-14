from scipy.spatial.distance import euclidean

from cleanlab.typing import FeatureArray, Metric

HIGH_DIMENSION_CUTOFF: int = 3
"""
If the number of columns (M) in the `features` array is greater than this cutoff value,
then by default, K-nearest-neighbors will use the "cosine" metric.
The cosine metric is more suitable for high-dimensional data.
Otherwise the "euclidean" distance will be used.

"""
ROW_COUNT_CUTOFF: int = 100
"""
Only affects settings where Euclidean metrics would be used by default.
If the number of rows (N) in the `features` array is greater than this cutoff value,
then by default, Euclidean distances are computed via the "euclidean" metric
(implemented in sklearn for efficiency reasons).
Otherwise, Euclidean distances are by default computed via
the ``euclidean`` metric from scipy (slower but numerically more precise/accurate).
"""


# Metric decision functions
def _euclidean_large_dataset() -> str:
    return "euclidean"


def _euclidean_small_dataset() -> Metric:
    return euclidean


def _cosine_metric() -> str:
    return "cosine"


def decide_euclidean_metric(features: FeatureArray) -> Metric:
    """
    Decide the appropriate Euclidean metric implementation based on the size of the dataset.

    Parameters
    ----------
    features :
        The input features array.

    Returns
    -------
    metric :
        A string or a callable representing a specific implementation of computing the euclidean distance.

    Note
    ----
    A choice is made between two implementations
    of the euclidean metric based on the number of rows in the feature array.
    If the number of rows (N) in the feature array is greater than another predefined
    cutoff value (ROW_COUNT_CUTOFF), the ``"euclidean"`` metric is used. This
    is because the euclidean metric performs better on larger datasets.
    If neither condition is met, the ``euclidean`` metric function from scipy is returned.

    See also
    --------
    ROW_COUNT_CUTOFF: The cutoff value for the number of rows in the feature array.
    sklearn.metrics.pairwise.euclidean_distances: The euclidean metric function from scikit-learn.
    scipy.spatial.distance.euclidean: The euclidean metric function from scipy.
    """
    num_rows = features.shape[0]
    if num_rows > ROW_COUNT_CUTOFF:
        return _euclidean_large_dataset()
    else:
        return _euclidean_small_dataset()


# Main function to decide the metric
def decide_default_metric(features: FeatureArray) -> Metric:
    """
    Decide the KNN metric to be used based on the shape of the feature array.

    Parameters
    ----------
    features :
        The input feature array, with shape (N, M), where N is the number of samples and M is the number of features.

    Returns
    -------
    metric :
        The distance metric to be used for neighbor search. It can be either a string
        representing the metric name ("cosine" or "euclidean") or a callable
        representing the metric function from scipy (euclidean).

    Note
    ----
    The decision of which metric to use is based on the shape of the feature array.
    If the number of columns (M) in the feature array is greater than a predefined
    cutoff value (HIGH_DIMENSION_CUTOFF), the "cosine" metric is used. This is because the cosine
    metric is more suitable for high-dimensional data.

    Otherwise, a euclidean metric is used.
    That is handled by the :py:meth:`~cleanlab.internal.neighbor.metric.decide_euclidean_metric` function.

    See Also
    --------
    HIGH_DIMENSION_CUTOFF: The cutoff value for the number of columns in the feature array.
    sklearn.metrics.pairwise.cosine_distances: The cosine metric function from scikit-learn
    """
    if features.shape[1] > HIGH_DIMENSION_CUTOFF:
        return _cosine_metric()
    return decide_euclidean_metric(features)
