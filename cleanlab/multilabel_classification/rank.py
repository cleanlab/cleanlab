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
Methods to rank the severity of label issues in multi-label classification datasets.
Here each example can belong to one or more classes, or none of the classes at all.
Unlike in standard multi-class classification, predicted class probabilities from model need not sum to 1 for each row in multi-label classification.
"""

import numpy as np  # noqa: F401: Imported for type annotations
import numpy.typing as npt
from typing import List, TypeVar, Dict, Any, Optional, Tuple

from cleanlab.internal.validation import assert_valid_inputs
from cleanlab.internal.util import get_num_classes
from cleanlab.internal.multilabel_utils import int2onehot
from cleanlab.internal.multilabel_scorer import MultilabelScorer, ClassLabelScorer, Aggregator


T = TypeVar("T", bound=npt.NBitBase)


def _labels_to_binary(
    labels: List[List[int]],
    pred_probs: npt.NDArray["np.floating[T]"],
) -> np.ndarray:
    """Validate the inputs to the multilabel scorer. Also transform the labels to a binary representation."""
    assert_valid_inputs(
        X=None, y=labels, pred_probs=pred_probs, multi_label=True, allow_one_class=True
    )
    num_classes = get_num_classes(labels=labels, pred_probs=pred_probs, multi_label=True)
    binary_labels = int2onehot(labels, K=num_classes)
    return binary_labels


def _create_multilabel_scorer(
    method: str,
    adjust_pred_probs: bool,
    aggregator_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[MultilabelScorer, Dict]:
    """This function acts as a factory that creates a MultilabelScorer."""
    base_scorer = ClassLabelScorer.from_str(method)
    base_scorer_kwargs = {"adjust_pred_probs": adjust_pred_probs}
    if aggregator_kwargs:
        aggregator = Aggregator(**aggregator_kwargs)
        scorer = MultilabelScorer(base_scorer, aggregator)
    else:
        scorer = MultilabelScorer(base_scorer)
    return scorer, base_scorer_kwargs


def get_label_quality_scores(
    labels: List[List[int]],
    pred_probs: npt.NDArray["np.floating[T]"],
    *,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
    aggregator_kwargs: Dict[str, Any] = {"method": "exponential_moving_average", "alpha": 0.8},
) -> npt.NDArray["np.floating[T]"]:
    """Computes a label quality score each example in a multi-label classification dataset.

    Scores are between 0 and 1 with lower scores indicating examples whose label more likely contains an error.
    For each example, this method internally computes a separate score for each individual class
    and then aggregates these per-class scores into an overall label quality score for the example.

    To estimate exactly which examples are mislabeled in a multi-label classification dataset,
    you can also use :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` with argument ``multi_label=True``.

    Parameters
    ----------
    labels : List[List[int]]
       List of noisy labels for multi-label classification where each example can belong to multiple classes.
       Refer to documentation for this argument in :py:func:`cleanlab.multilabel_classification.filter.find_label_issues <cleanlab.multilabel_classification.filter.find_label_issues>` for further details.

    pred_probs : np.ndarray
      An array of shape ``(N, K)`` of model-predicted class probabilities.
      Refer to documentation for this argument in :py:func:`cleanlab.multilabel_classification.filter.find_label_issues <cleanlab.multilabel_classification.filter.find_label_issues>` for further details.

    method : {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default = "self_confidence"
      Method to calculate separate per class annotation scores that are subsequently aggregated to form an overall label quality score.
      These scores are separately calculated for each class based on the corresponding column of `pred_probs` in a one-vs-rest manner,
      and are standard label quality scores for multi-class classification.

      See also
      --------
      :py:func:`rank.get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function for details about each option.

    adjust_pred_probs : bool, default = False
      Account for class imbalance in the label-quality scoring by adjusting predicted probabilities.
      Refer to documentation for this argument in :py:func:`rank.get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` for details.


    aggregator_kwargs : dict, default = {"method": "exponential_moving_average", "alpha": 0.8}
      A dictionary of hyperparameter values for aggregating per class scores into an overall label quality score for each example.
      Options for ``"method"`` include: ``"exponential_moving_average"`` or ``"softmin"`` or your own callable function.
      See :py:class:`internal.multilabel_scorer.Aggregator <cleanlab.internal.multilabel_scorer.Aggregator>` for details about each option and other possible hyperparameters.

    Returns
    -------
    label_quality_scores : np.ndarray
      A 1D array of shape ``(N,)`` with a label quality score (between 0 and 1) for each example in the dataset.
      Lower scores indicate examples whose label is more likely to contain annotation errors.


    Examples
    --------
    >>> from cleanlab.multilabel_classification import get_label_quality_scores
    >>> import numpy as np
    >>> labels = [[1], [0,2]]
    >>> pred_probs = np.array([[0.1, 0.9, 0.1], [0.4, 0.1, 0.9]])
    >>> scores = get_label_quality_scores(labels, pred_probs)
    >>> scores
    array([0.9, 0.5])
    """
    binary_labels = _labels_to_binary(labels, pred_probs)
    scorer, base_scorer_kwargs = _create_multilabel_scorer(
        method=method,
        adjust_pred_probs=adjust_pred_probs,
        aggregator_kwargs=aggregator_kwargs,
    )
    return scorer(binary_labels, pred_probs, base_scorer_kwargs=base_scorer_kwargs)


def get_label_quality_scores_per_class(
    labels: List[List[int]],
    pred_probs: npt.NDArray["np.floating[T]"],
    *,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
) -> np.ndarray:
    """Scores the quality of each individual class annotation for each example in a multi-label classification dataset.

    Scores are between 0 and 1 with lower scores indicating examples where this class was less likely chosen correctly.
     Remember each class either applies to the example or not, so low scores indicate examples where this class is likely annotated as True when it should be False or vice-versa.

    These scores are aggregated across classes into a single quality score for each example in the `multilabel_classification.rank.get_label_quality_scores` method.

    To estimate exactly which examples are mislabeled in a multi-label classification dataset,
    you can also use :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` with argument ``multi_label=True``.

    Parameters
    ----------
    labels : List[List[int]]
       List of noisy labels for multi-label classification where each example can belong to multiple classes.
       Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.multilabel_classification.filter.find_label_issues>` for further details.

    pred_probs : np.ndarray
      An array of shape ``(N, K)`` of model-predicted class probabilities.
      Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.multilabel_classification.filter.find_label_issues>` for further details.

    method : {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default = "self_confidence"
      Method to calculate separate per class annotation scores.
      Refer to documentation for this argument in :py:func:`get_label_quality_scores <cleanlab.multilabel_classification.rank.get_label_quality_scores>` for further details.

    adjust_pred_probs : bool, default = False
      Account for class imbalance in the label-quality scoring by adjusting predicted probabilities.
      Refer to documentation for this argument in :py:func:`rank.get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` for details.

    aggregator_kwargs : dict, default = {"method": "exponential_moving_average", "alpha": 0.8}
      A dictionary of hyperparameter values for aggregating per class scores into an overall label quality score for each example.
      Options for ``"method"`` include: ``"exponential_moving_average"`` or ``"softmin"`` or your own callable function.
      See :py:class:`internal.multilabel_scorer.Aggregator <cleanlab.internal.multilabel_scorer.Aggregator>` for details about each option and other possible hyperparameters.

    Returns
    -------
    label_quality_scores : np.ndarray
      A 1D array of shape ``(N,)`` with a label quality score (between 0 and 1) for each example in the dataset.
      Lower scores indicate examples whose label is more likely to contain annotation errors.

    To get a score for each class annotation for each example, use the :py:func:`cleanlab.multilabel_classification.classification.rank.get_label_quality_scores_per_class <cleanlab.multilabel_classification.rank.get_label_quality_scores_per_class>` method instead.
    Examples
    --------
    >>> from cleanlab.multilabel_classification import get_label_quality_scores
    >>> import numpy as np
    >>> labels = [[1], [0,2]]
    >>> pred_probs = np.array([[0.1, 0.9, 0.1], [0.4, 0.1, 0.9]])
    >>> scores = get_label_quality_scores(labels, pred_probs)
    >>> scores
    array([0.9, 0.5])
    """
    binary_labels = _labels_to_binary(labels, pred_probs)
    scorer, base_scorer_kwargs = _create_multilabel_scorer(
        method=method,
        adjust_pred_probs=adjust_pred_probs,
    )
    return scorer.get_class_label_quality_scores(
        labels=binary_labels, pred_probs=pred_probs, base_scorer_kwargs=base_scorer_kwargs
    )
