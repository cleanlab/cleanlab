"""
Helper functions used internally for outlier detection tasks.
"""

from typing import Optional
import numpy as np

from cleanlab.internal.constants import EPSILON


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
    return np.exp(-t * avg_distances / max(scaling_factor, EPSILON))


def correct_precision_errors(
    scores: np.ndarray,
    avg_distances: np.ndarray,
    metric: str,
    C: int = 100,
    p: Optional[int] = None,
):
    """
    Ensure that scores where avg_distances are below the tolerance threshold get a score of one.

    Parameters
    ----------
    scores :
        An array of scores of shape ``(N)``, where N is the number of examples.
        Each entry represents a score between 0 and 1.

    avg_distances :
        An array of distances of shape ``(N)``, where N is the number of examples.
        Each entry represents an example's average distance to its k nearest neighbors.

    metric :
        The metric used by the knn algorithm to calculate the distances.
        It must be 'cosine', 'euclidean' or 'minkowski', otherwise this function does nothing.

    C :
        Multiplier used to increase the tolerance of the acceptable precision differences.
        It is a multiplicative factor of the machine epsilon that is used to calculate the tolerance.
        For the type of values that are used in the distances, a value of 100 should be a sensible
        default value for small values of the distances, below the order of 1.

    p :
        This value is only used when metric is 'minkowski'.
        A ValueError will be raised if metric is 'minkowski' and 'p' was not provided.

    Returns
    -------
    fixed_scores :
        An array of scores of shape ``(N,)`` for N examples with scores between 0 and 1.
    """
    if metric == "cosine":
        tolerance = C * np.finfo(np.float64).epsneg
    elif metric == "euclidean":
        tolerance = np.sqrt(C * np.finfo(np.float64).eps)
    elif metric == "minkowski":
        if p is None:
            raise ValueError("When metric is 'minkowski' you must specify the 'p' parameter")
        tolerance = (C * np.finfo(np.float64).eps) ** (1 / p)
    else:
        return scores

    candidates_mask = avg_distances < tolerance
    scores[candidates_mask] = 1
    return scores
