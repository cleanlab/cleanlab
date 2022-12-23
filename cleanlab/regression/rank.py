import numpy as np
from cleanlab.outlier import OutOfDistribution
from sklearn.neighbors import NearestNeighbors
from cleanlab.internal.regression_utils import assert_valid_inputs

""" Generates label quality scores for every sample in regression dataset """


def get_label_quality_scores(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    method: str = "TO_BE_NAMED",  # TODO update name once finalised
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
        Array of shape ``(N, )`` consisting given labels, where N is number of datapoints in the regression dataset.

    predictions : np.ndarray
        Predicated labels from regressor fitted on the dataset.
        Array of shape ``(N,)`` consisting predicted labels, where N is number of datapoints in the regression dataset.

    method : {"residual", "TO_BE_NAMED"}, default="TO_BE_NAMED" #TODO - update name once finalised

    Returns
    -------
    label_quality_scores:
        Array of shape ``(N, )`` of scores between 0 and 1, one per datapoint in the dataset.

        Lower scores indicate datapoint more likely to contain a label issue.

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

    scoring_funcs = {
        "residual": get_residual_score_for_each_label,
        "TO_BE_NAMED": get_score_to_named_for_each_label,  # TODO - update name once finalised
    }

    # TODO - update name once finalised
    try:
        scoring_func = scoring_funcs[method]
    except KeyError:
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
    """Returns the residual based label-quality scores for each datapoints.

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


# TODO - change name of the function
# TODO - change name of function in test
def get_score_to_named_for_each_label(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    variance: float = 10,
) -> np.ndarray:
    """Returns label-quality scores.

    This is function to compute label-quality scores for regression datasets,
    where lower score indicate labels less likely to be correct.

    Parameters
    ----------
    labels: np.ndarray
        Labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.regression.rank.get_label_quality_scores>` function.

    predictions: np.ndarray
        Predicted labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.regression.rank.get_label_quality_scores>` function.

    variance: float, default = 10
        Manipulates variance of the distribution of residual.

    Returns
    -------
    label_quality_scores: np.ndarray
        Contains one score (between 0 and 1) per example.
        Lower scores indicate more likely mislabled examples.
    """

    neighbors = int(np.ceil(0.1 * labels.shape[0]))
    knn = NearestNeighbors(n_neighbors=neighbors, metric="euclidean")

    residual = predictions - labels

    labels = (labels - labels.mean()) / labels.std()
    residual = np.sqrt(variance) * ((residual - residual.mean()) / residual.std())

    # 2D features by combining labels and residual
    features = np.array([labels, residual]).T

    knn.fit(features)
    ood = OutOfDistribution(params={"knn": knn})
    label_quality_scores = ood.score(features=features)
    return label_quality_scores
