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
Methods to score the quality of each label in a regression dataset. These can be used to rank the examples whose Y-value is most likely erroneous.

Note: Label quality scores are most accurate when they are computed based on out-of-sample `predictions` from your regression model.
To obtain out-of-sample predictions for every datapoint in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`. This is encouraged to get better results.

If you have a sklearn-compatible regression model, consider using `cleanlab.regression.learn.CleanLearning` instead, which can more accurately identify noisy label values.
"""

from typing import Dict, Callable
import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors

from cleanlab.outlier import OutOfDistribution
from cleanlab.internal.regression_utils import assert_valid_prediction_inputs

from cleanlab.internal.constants import TINY_VALUE


def get_label_quality_scores(
    labels: ArrayLike,
    predictions: ArrayLike,
    *,
    method: str = "outre",
) -> np.ndarray:
    """
    Returns label quality score for each example in the regression dataset.

    Each score is a continous value in the range [0,1]

    * 1 - clean label (given label is likely correct).
    * 0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels : array_like
        Raw labels from original dataset.
        1D array of shape ``(N, )`` containing the given labels for each example (aka. Y-value, response/target/dependent variable), where N is number of examples in the dataset.

    predictions : np.ndarray
        1D array of shape ``(N,)`` containing the predicted label for each example in the dataset.  These should be out-of-sample predictions from a trained regression model, which you can obtain for every example in your dataset via :ref:`cross-validation <pred_probs_cross_val>`.

    method : {"residual", "outre"}, default="outre"
        String specifying which method to use for scoring the quality of each label and identifying which labels appear most noisy.

    Returns
    -------
    label_quality_scores:
        Array of shape ``(N, )`` of scores between 0 and 1, one per example in the dataset.

        Lower scores indicate examples more likely to contain a label issue.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.regression.rank import get_label_quality_scores
    >>> labels = np.array([1,2,3,4])
    >>> predictions = np.array([2,2,5,4.1])
    >>> label_quality_scores = get_label_quality_scores(labels, predictions)
    >>> label_quality_scores
    array([0.00323821, 0.33692597, 0.00191686, 0.33692597])
    """

    # Check if inputs are valid
    labels, predictions = assert_valid_prediction_inputs(
        labels=labels, predictions=predictions, method=method
    )

    scoring_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        "residual": _get_residual_score_for_each_label,
        "outre": _get_outre_score_for_each_label,
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


def _get_residual_score_for_each_label(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """Returns a residual label-quality score for each example.

    This is function to compute label-quality scores for regression datasets,
    where lower score indicate labels less likely to be correct.

    Residual based scores can work better for datasets where independent variables
    are based out of normal distribution.

    Parameters
    ----------
    labels: np.ndarray
        Labels in the same format expected by the `~cleanlab.regression.rank.get_label_quality_scores` function.

    predictions: np.ndarray
        Predicted labels in the same format expected by the `~cleanlab.regression.rank.get_label_quality_scores` function.

    Returns
    -------
    label_quality_scores: np.ndarray
        Contains one score (between 0 and 1) per example.
        Lower scores indicate more likely mislabled examples.

    """
    residual = predictions - labels
    label_quality_scores = np.exp(-abs(residual))
    return label_quality_scores


def _get_outre_score_for_each_label(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    residual_scale: float = 5,
    frac_neighbors: float = 0.5,
    neighbor_metric: str = "euclidean",
) -> np.ndarray:
    """Returns OUTRE based label-quality scores.

    This function computes label-quality scores for regression datasets,
    where a lower score indicates labels that are less likely to be correct.

    Parameters
    ----------
    labels: np.ndarray
        Labels in the same format as expected by the `~cleanlab.regression.rank.get_label_quality_scores` function.

    predictions: np.ndarray
        Predicted labels in the same format as expected by the `~cleanlab.regression.rank.get_label_quality_scores` function.

    residual_scale: float, default = 5
        Multiplicative factor to adjust scale (standard deviation) of the residuals relative to the labels.

    frac_neighbors: float, default = 0.5
        Fraction of examples in dataset that should be considered as `n_neighbors` in the ``NearestNeighbors`` object used internally to assess outliers.

    neighbor_metric: str, default = "euclidean"
        The parameter is passed to sklearn NearestNeighbors. # TODO add reference to sklearn.NearestNeighbor?

    Returns
    -------
    label_quality_scores: np.ndarray
        Contains one score (between 0 and 1) per example.
        Lower scores indicate more likely mislabled examples.
    """
    residual = predictions - labels
    labels = (labels - labels.mean()) / (labels.std() + TINY_VALUE)
    residual = residual_scale * ((residual - residual.mean()) / (residual.std() + TINY_VALUE))

    # 2D features by combining labels and residual
    features = np.array([labels, residual]).T

    neighbors = int(np.ceil(frac_neighbors * labels.shape[0]))
    knn = NearestNeighbors(n_neighbors=neighbors, metric=neighbor_metric).fit(features)
    ood = OutOfDistribution(params={"knn": knn})

    label_quality_scores = ood.score(features=features)
    return label_quality_scores
