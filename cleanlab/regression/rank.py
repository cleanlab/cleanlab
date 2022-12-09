import numpy as np
from cleanlab.outlier import OutOfDistribution
from sklearn.neighbors import NearestNeighbors

""" Generates label quality scores for every sample in regression dataset """


def get_label_quality_scores(
    labels: np.ndarray,
    pred_labels: np.ndarray,
    *,
    method: str = "residual",
) -> np.ndarray:
    """
    Returns label quality score for each example in the regression dataset.

    Each score is continous value in range [0,1]
    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels : np.ndarray
        Raw labels from original dataset.
        Array of shape ``(N, )`` consisting given labels, where N is number of datapoints in the regression dataset.

    pred_labels : np.ndarray
        Predicated labels from regressor fitted on the dataset.
        Array of shape ``(N,)`` consisting predicted labels, where N is number of datapoints in the regression dataset.

    method : {"residual", "TO_BE_NAMED"}, default="residual" #TODO - update name once finalised

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
    >>> pred_labels = np.array([2,2,5,4.1])
    >>> label_quality_scores = get_label_quality_scores(labels, pred_labels)
    >>> label_quality_scores
    array([0.36787944, 1.        , 0.13533528, 0.90483742])
    """

    # TODO - add error trigger function in utils.
    if not isinstance(labels, np.ndarray) or not isinstance(pred_labels, np.ndarray):
        raise TypeError("labels and pred_labels must be of type np.ndarray")

    assert (
        labels.shape == pred_labels.shape
    ), f"shape of label {labels.shape} and predicted labels {pred_labels.shape} are not same."

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
            Please choose a valid scoring technique: residual, TO_BE_NAMED.
            """
        )

    # Calculate scores
    label_quality_score = scoring_func(labels, pred_labels)
    return label_quality_score


def get_residual_score_for_each_label(
    labels: np.ndarray,
    pred_labels: np.ndarray,
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

    pred_labels: np.ndarray
        Predicted labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.regression.rank.get_label_quality_scores>` function.

    Returns
    -------
    label_quality_scores: np.ndarray
        Contains one score (between 0 and 1) per example.
        Lower scores indicate more likely mislabled examples.

    """
    residual = pred_labels - labels
    label_quality_scores = np.exp(-abs(residual))
    return label_quality_scores


# TODO - change name of the function
def get_score_to_named_for_each_label(
    label: np.ndarray,
    pred_labels: np.ndarray,
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

    pred_labels: np.ndarray
        Predicted labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.regression.rank.get_label_quality_scores>` function.

    variance: float, default = 10
        Manipulates variance of the distribution of residual.

    Returns
    -------
    label_quality_scores: np.ndarray
        Contains one score (between 0 and 1) per example.
        Lower scores indicate more likely mislabled examples.
    """

    neighbors = int(np.ceil(0.1 * label.shape[0]))
    print(neighbors)
    knn = NearestNeighbors(n_neighbors=neighbors, metric="euclidean")

    residual = pred_labels - label

    label = (label - label.mean()) / label.std()
    residual = np.sqrt(variance) * ((residual - residual.mean()) / residual.std())

    # 2D features by combining labels and residual
    features = np.array([label, residual]).T

    knn.fit(features)
    ood = OutOfDistribution(params={"knn": knn})
    label_quality_scores = ood.score(features=features)
    return label_quality_scores
