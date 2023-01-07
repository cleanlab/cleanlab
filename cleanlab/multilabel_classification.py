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
from typing import List, TypeVar, Dict, Any

from cleanlab.internal.validation import assert_valid_inputs
from cleanlab.internal.util import get_num_classes
from cleanlab.internal.multilabel_scorer import MultilabelScorer, ClassLabelScorer, Aggregator
from cleanlab.internal.multilabel_utils import int2onehot


T = TypeVar("T", bound=npt.NBitBase)


def get_label_quality_scores(
    labels: List[List[int]],
    pred_probs: npt.NDArray["np.floating[T]"],
    *,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
    aggregator_kwargs: Dict[str, Any] = {"method": "exponential_moving_average", "alpha": 0.8}
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
      Multi-label classification labels for each example, which is allowed to belong to multiple classes.
      The i-th element of `labels` corresponds to list of classes that i-th example belongs to (e.g. ``labels = [[1,2],[1],[0],..]``).

      Important
      ---------
      *Format requirements*: For dataset with K classes, individual class labels must be integers in 0, 1, ..., K-1.

    pred_probs : np.ndarray
      A 2D array of shape ``(N, K)`` of model-predicted class probabilities ``P(label=k|x)``.
      Each row of this matrix corresponds to an example `x` and contains the predicted probabilities
      that `x` belongs to each possible class, for each of the K classes.
      The columns of this array must be ordered such that these probabilities correspond to class 0, 1, ..., K-1.
      In multi-label classification (where classes are not mutually exclusive), the rows of `pred_probs` need not sum to 1.

      Note
      ----
      Estimated label quality scores are most accurate when they are computed based on out-of-sample ``pred_probs`` from your model.
      To obtain out-of-sample predicted probabilities for every example in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`.
      This is encouraged to get better results.

    method : {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default = "self_confidence"
      Method to calculate separate per class annotation scores that are subsequently aggregated to form an overall label quality score.
      These scores are separately calculated for each class based on the corresponding column of `pred_probs` in a one-vs-rest manner,
      and are standard label quality scores for multi-class classification.

      See also
      --------
      :py:func:`rank.get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function for details about each option.

    adjust_pred_probs : bool, default = False
      Account for class imbalance in the label-quality scoring by adjusting predicted probabilities
      via subtraction of class confident thresholds and renormalization.
      Set this to ``True`` if you prefer to account for class-imbalance.
      See `Northcutt et al., 2021 <https://jair.org/index.php/jair/article/view/12125>`_.

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

    assert_valid_inputs(
        X=None, y=labels, pred_probs=pred_probs, multi_label=True, allow_one_class=True
    )
    num_classes = get_num_classes(labels=labels, pred_probs=pred_probs, multi_label=True)
    binary_labels = int2onehot(labels, K=num_classes)
    base_scorer = ClassLabelScorer.from_str(method)
    base_scorer_kwargs = {"adjust_pred_probs": adjust_pred_probs}
    aggregator = Aggregator(**aggregator_kwargs)
    scorer = MultilabelScorer(base_scorer, aggregator)
    return scorer(binary_labels, pred_probs, base_scorer_kwargs=base_scorer_kwargs)
