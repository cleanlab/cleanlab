import numpy as np


def get_label_quality_scores(labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """
    Returns label quality score for each example in the regression dataset.

    Each score is continous value in range [0,1]
    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels:
        Raw labels from original dataset.
        Array of shape ``(N, )`` consisting given labels, where N is number of datapoints in the regression dataset.

    pred_labels:
        Predicated labels from regressor fitted on the dataset.
        Array of shape ``(N,)`` consisting predicted labels, where N is number of datapoints in the regression dataset.

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

    assert (
        labels.shape == pred_labels.shape
    ), f"shape of label {labels.shape} and predicted labels {pred_labels.shape} are not same."

    residual = pred_labels - labels
    quality_scores = np.exp(-abs(residual))
    return quality_scores
