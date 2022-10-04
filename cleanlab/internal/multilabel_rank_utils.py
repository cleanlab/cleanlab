# Copyright (C) 2017-2022  Cleanlab Inc.
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
Helper classes and functions used internally for computing multilabel label quality scores.
"""

from enum import Enum
from typing import Callable, Optional

import numpy as np
from sklearn.utils.multiclass import is_multilabel

from cleanlab.rank import (
    get_self_confidence_for_each_label,
    get_normalized_margin_for_each_label,
    get_confidence_weighted_entropy_for_each_label,
)


class _Wrapper:
    """Helper class for wrapping callable functions as attributes of an Enum instead of
    setting them as methods of the Enum class.


    This class is only intended to be used internally for the BaseQualityScorer or
    other cases where functions are used for enumeration values.
    """

    def __init__(self, f: Callable) -> None:
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __repr__(self):
        return self.f.__name__


class BaseQualityScorer(Enum):
    """Enum for the different methods to compute label quality scores."""

    SELF_CONFIDENCE = _Wrapper(get_self_confidence_for_each_label)
    NORMALIZED_MARGIN = _Wrapper(get_normalized_margin_for_each_label)
    CONFIDENCE_WEIGHTED_ENTROPY = _Wrapper(get_confidence_weighted_entropy_for_each_label)

    def __call__(self, labels: np.ndarray, pred_probs: np.ndarray, **kwargs) -> np.ndarray:
        """Returns the label-quality scores for each datapoint based on the given labels and predicted probabilities."""
        return self.value(labels, pred_probs, **kwargs)


class MultilabelScorer:
    """A class for aggregating label quality scores for multi-label classification problems."""

    def __init__(
        self,
        base_scorer: BaseQualityScorer = BaseQualityScorer.SELF_CONFIDENCE,
        aggregator: Optional[Callable[..., np.ndarray]] = None,
        *,
        strict: bool = True,
    ):
        """
        Initialize object with a base scoring function that is applied to each label and function that pools scores accross labels.

        Parameters
        ----------
        base_scorer:
            A function that computes a quality score for a single label in a multi-label classification problem.

        aggregator:
            A function that aggregates the scores computed by base_scorer over all labels.
            If None, the scores are averaged.

        strict:
            If True, raises an error if the labels are not binary or are incompatible with the predicted probabilities.

        Examples
        --------
        >>> from cleanlab.internal.multilabel_rank import MultilabelScorer, BaseQualityScorer
        >>> import numpy as np
        >>> scorer = MultilabelScorer(
        ...     base_scorer = BaseQualityScorer.NORMALIZED_MARGIN,
        ...     aggregator = np.min,
        ... )
        >>> labels = np.array([[0, 1, 0], [1, 0, 1]])
        >>> pred_probs = np.array([[0.1, 0.9, 0.1], [0.4, 0.1, 0.9]])
        >>> scores = scorer(labels, pred_probs)
        >>> scores
        array([0.9, 0.4])
        """
        self.base_scorer = base_scorer
        if aggregator is None:
            self.aggregator = np.mean
        else:
            self.aggregator = aggregator
        self.strict = strict

    def __call__(self, labels: np.ndarray, pred_probs: np.ndarray, **kwargs) -> np.ndarray:
        """
        Computes a quality score for each label in a multi-label classification problem
        based on out-of-sample predicted probabilities.
        The score is computed by averaging the base_scorer over all labels.

        Parameters
        ----------
        labels:
            A 2D array of shape (n_samples, n_labels) with binary labels.

        pred_probs:
            A 2D array of shape (n_samples, n_labels) with predicted probabilities.

        kwargs:
            Additional keyword arguments to pass to the base_scorer.

        Returns
        -------
        scores:
            A 1D array of shape (n_samples,) with the quality scores for each datapoint.

        Examples
        --------
        >>> from cleanlab.internal.multilabel_rank import MultilabelScorer
        >>> import numpy as np
        >>> scorer = MultilabelScorer()
        >>> labels = np.array([[0, 1, 0], [1, 0, 1]])
        >>> pred_probs = np.array([[0.1, 0.9, 0.1], [0.4, 0.1, 0.9]])
        >>> scores = scorer(labels, pred_probs)
        >>> scores
        """
        if self.strict:
            self._validate_labels_and_pred_probs(labels, pred_probs)
        scores = np.zeros(shape=labels.shape)
        for i, (label_i, pred_prob_i) in enumerate(zip(labels.T, pred_probs.T)):
            pred_prob_i_two_columns = self._stack_complement(pred_prob_i)
            scores[:, i] = self.base_scorer(label_i, pred_prob_i_two_columns, **kwargs)

        return self.aggregator(scores, axis=-1)

    @staticmethod
    def _stack_complement(pred_prob_slice: np.ndarray) -> np.ndarray:
        """
        Extends predicted probabilities of a single class to two columns.

        Parameters
        ----------
        pred_prob_slice:
            A 1D array with predicted probabilities for a single class.

        Example
        -------
        >>> pred_prob_slice = np.array([0.1, 0.9, 0.3, 0.8])
        >>> MultilabelScorer._stack_complement(pred_prob_slice)
        array([[0.9, 0.1],
                [0.1, 0.9],
                [0.7, 0.3],
                [0.2, 0.8]])
        """
        return np.vstack((1 - pred_prob_slice, pred_prob_slice)).T

    @staticmethod
    def _validate_labels_and_pred_probs(labels: np.ndarray, pred_probs: np.ndarray) -> None:
        """
        Checks that (multi-)labels are in the proper binary indicator format and that
        they are compatible with the predicted probabilities.
        """
        if not is_multilabel(labels):
            raise ValueError("Labels must be in multi-label format.")
        if labels.shape != pred_probs.shape:
            raise ValueError("Labels and predicted probabilities must have the same shape.")


def get_label_quality_scores(labels, pred_probs, *, method: MultilabelScorer):
    return method(labels, pred_probs)
