import numpy as np
from cleanlab.outlier import OutOfDistribution
from sklearn.neighbors import NearestNeighbors
from cleanlab.internal.regression_utils import assert_valid_inputs
from typing import Dict, Callable

""" Generates label quality scores for every sample in regression dataset """


def get_label_quality_scores(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    method: str = "outre",  # TODO update name once finalised
) -> np.ndarray:
    """
    Returns label quality score for each example in the regression dataset.

    Each score is a continous value in the range [0,1]
    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels : np.ndarray
        Raw labels from original dataset.
        1D array of shape ``(N, )`` containing the given labels for each example (aka. Y-value, response, target, dependent variable, ...), where N is number of examples in the dataset.

    predictions : np.ndarray
        1D array of shape ``(N,)`` containing the predicted label for each example in the dataset.  These should be out-of-sample predictions from a trained regression model, which you can obtain for every example in your dataset via :ref:`cross-validation <pred_probs_cross_val>`.

    method : {"residual", "outre"}, default="outre"

    Returns
    -------
    label_quality_scores:
        Array of shape ``(N, )`` of scores between 0 and 1, one per datapoint in the dataset.

        Lower scores indicate datapoints more likely to contain a label issue.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.regression.rank import get_label_quality_scores
    >>> labels = np.array([1,2,3,4])
    >>> predictions = np.array([2,2,5,4.1])
    >>> label_quality_scores = get_label_quality_scores(labels, predictions)
    >>> label_quality_scores
    array([0.36787944, 1.        , 0.13533528, 0.90483742])
    """

    # Check if inputs are valid
    assert_valid_inputs(labels=labels, predictions=predictions, method=method)

    scoring_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        "residual": get_residual_score_for_each_label,
        "outre": get_outre_score_for_each_label,
    }

    scoring_func = scoring_funcs.get(method, None)
    if not scoring_func:
        raise ValueError(
            f"""
            {method} is not a valid scoring method.
            Please choose a valid scoring technique: {scoring_funcs.keys()}.
            """
        )

    # Calculate scores
    label_quality_scores = scoring_func(labels, predictions)
    return label_quality_scores


def get_residual_score_for_each_label(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """Returns a residual label-quality score for each datapoint.

    This is function to compute label-quality scores for regression datasets,
    where lower score indicate labels less likely to be correct.

    Residual based scores can work better for datasets where independent variables
    are based out of normal distribution.

    Parameters
    ----------
    labels: np.ndarray
        Labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.regression.rank.get_label_quality_scores>` function.

    predictions: np.ndarray
        Predicted labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.regression.rank.get_label_quality_scores>` function.

    Returns
    -------
    label_quality_scores: np.ndarray
        Contains one score (between 0 and 1) per example.
        Lower scores indicate more likely mislabled examples.

    """
    residual = predictions - labels
    label_quality_scores = np.exp(-abs(residual))
    return label_quality_scores


# TODO - change name of function in test
def get_outre_score_for_each_label(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    variance: float = 10,
) -> np.ndarray:
    """Returns OUTRE based label-quality scores.

    This function computes label-quality scores for regression datasets,
    where a lower score indicates labels that are less likely to be correct.

    Parameters
    ----------
    labels: np.ndarray
        Labels in the same format as expected by the :py:func:`get_label_quality_scores <cleanlab.regression.rank.get_label_quality_scores>` function.

    predictions: np.ndarray
        Predicted labels in the same format as expected by the :py:func:`get_label_quality_scores <cleanlab.regression.rank.get_label_quality_scores>` function.

    variance: float, default = 10
        Manipulates variance of the distribution of residual.

    Returns
    -------
    label_quality_scores: np.ndarray
        Contains one score (between 0 and 1) per example.
        Lower scores indicate more likely mislabled examples.
    """
    residual = predictions - labels
    labels = (labels - labels.mean()) / labels.std()
    residual = np.sqrt(variance) * ((residual - residual.mean()) / residual.std())

    # 2D features by combining labels and residual
    features = np.array([labels, residual]).T

    neighbors = int(np.ceil(0.1 * labels.shape[0]))
    knn = NearestNeighbors(n_neighbors=neighbors, metric="euclidean").fit(features)
    ood = OutOfDistribution(params={"knn": knn})

    label_quality_scores = ood.score(features=features)
    return label_quality_scores
