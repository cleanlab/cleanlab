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
Helper classes and functions used internally to compute label quality scores in multi-label classification.
"""

from enum import Enum
import itertools
from typing import Callable, Optional

import numpy as np
from sklearn.model_selection import cross_val_predict

from cleanlab.rank import (
    get_self_confidence_for_each_label,
    get_normalized_margin_for_each_label,
    get_confidence_weighted_entropy_for_each_label,
)


def _is_multilabel(y: np.ndarray) -> bool:
    """Checks whether `y` is in a multi-label indicator matrix format.

    Sparse matrices are not supported.
    """
    if not (isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] > 1):
        return False
    return np.array_equal(np.unique(y), [0, 1])


class _Wrapper:
    """Helper class for wrapping callable functions as attributes of an Enum instead of
    setting them as methods of the Enum class.


    This class is only intended to be used internally for the ClassLabelScorer or
    other cases where functions are used for enumeration values.
    """

    def __init__(self, f: Callable) -> None:
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __repr__(self):
        return self.f.__name__


class ClassLabelScorer(Enum):
    """Enum for the different methods to compute label quality scores."""

    SELF_CONFIDENCE = _Wrapper(get_self_confidence_for_each_label)
    NORMALIZED_MARGIN = _Wrapper(get_normalized_margin_for_each_label)
    CONFIDENCE_WEIGHTED_ENTROPY = _Wrapper(get_confidence_weighted_entropy_for_each_label)

    def __call__(self, labels: np.ndarray, pred_probs: np.ndarray, **kwargs) -> np.ndarray:
        """Returns the label-quality scores for each datapoint based on the given labels and predicted probabilities."""
        return self.value(labels, pred_probs, **kwargs)


class Aggregator:
    """Helper class for aggregating the label quality scores for each class into a single score for each datapoint.

    Parameters
    ----------
    method:
        The method to compute the label quality scores for each class.

    kwargs:
        Additional keyword arguments to pass to the method when called.
    """

    def __init__(self, method, **kwargs):
        self._validate_method(method)
        self.method = method
        self.kwargs = kwargs

    @staticmethod
    def _validate_method(method) -> None:
        assert callable(method), f"Expected callable method, got {type(method)}"

    @staticmethod
    def _validate_scores(scores: np.ndarray) -> None:
        if not (isinstance(scores, np.ndarray) and scores.ndim == 2):
            raise ValueError(
                f"Expected 2D array for scores, got {type(scores)} with shape {scores.shape}"
            )

    def __call__(self, scores: np.ndarray, **kwargs) -> np.ndarray:
        """Returns the label quality scores for each datapoint based on the given label quality scores for each class.

        Parameters
        ----------
        scores:
            The label quality scores for each class.

        Returns
        -------
            A single label quality score for each datapoint.
        """
        self._validate_scores(scores)
        kwargs["axis"] = 1
        return self.method(scores, **{**kwargs, **self.kwargs})

    def __repr__(self):
        return f"Aggregator(method={self.method.__name__})"


def exponential_moving_average(
    s: np.ndarray,
    *,
    alpha: Optional[float] = None,
    axis: int = 1,
    **_,
) -> np.ndarray:
    """Exponential moving average (EMA) score function.

    For a score vector s = (s_1, ..., s_K) with K scores, the values
    are sorted in *descending* order and the exponential moving average
    of the last score is calculated.

    Parameters
    ----------
    s :
        Scores to be transformed.

    alpha :
        Discount factor that determines the weight of the previous EMA score.
        Higher alpha means that the previous EMA score has a lower weight while
        the current score has a higher weight.

        If alpha is None, it is set to 2 / (K + 1) where K is the number of scores.

    axis :
        Axis along which the scores are sorted.

    Returns
    -------
        Exponential moving average score.

    Examples
    --------
    >>> from cleanlab.internal.multilabel_utils import exponential_moving_average
    >>> import numpy as np
    >>> s = np.array([0.1, 0.2, 0.3])
    >>> exponential_moving_average(s, alpha=0.5)
    np.array([0.175])
    """
    K = s.shape[1]
    s_sorted = np.fliplr(np.sort(s, axis=axis))
    if alpha is None:
        # One conventional choice for alpha is 2/(K + 1), where K is the number of periods in the moving average.
        alpha = float(2 / (K + 1))
    s_T = s_sorted.T
    s_ema, s_next = s_T[0], s_T[1:]
    for s_i in s_next:
        s_ema = alpha * s_i + (1 - alpha) * s_ema
    return s_ema


class MultilabelScorer:
    """Aggregates label quality scores across different classes to produce one score per example in multi-label classification tasks."""

    def __init__(
        self,
        base_scorer: ClassLabelScorer = ClassLabelScorer.SELF_CONFIDENCE,
        aggregator: Aggregator = Aggregator(exponential_moving_average, alpha=0.8),
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
        >>> from cleanlab.internal.multilabel_utils import MultilabelScorer, ClassLabelScorer
        >>> import numpy as np
        >>> scorer = MultilabelScorer(
        ...     base_scorer = ClassLabelScorer.NORMALIZED_MARGIN,
        ...     aggregator = np.min,  # Use the "worst" label quality score for each example.
        ... )
        >>> labels = np.array([[0, 1, 0], [1, 0, 1]])
        >>> pred_probs = np.array([[0.1, 0.9, 0.1], [0.4, 0.1, 0.9]])
        >>> scores = scorer(labels, pred_probs)
        >>> scores
        array([0.9, 0.4])
        """
        self.base_scorer = base_scorer
        if not isinstance(aggregator, Aggregator):
            self.aggregator = Aggregator(aggregator)
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
            Additional keyword arguments to pass to the base_scorer and the aggregator.

        Returns
        -------
        scores:
            A 1D array of shape (n_samples,) with the quality scores for each datapoint.

        Examples
        --------
        >>> from cleanlab.internal.multilabel_utils import MultilabelScorer
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

        return self.aggregator(scores, **kwargs)

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
        # Only allow dense matrices for labels for now
        if not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a numpy array.")
        if not _is_multilabel(labels):
            raise ValueError("Labels must be in multi-label format.")
        if labels.shape != pred_probs.shape:
            raise ValueError("Labels and predicted probabilities must have the same shape.")


def get_label_quality_scores(labels, pred_probs, *, method: MultilabelScorer):
    return method(labels, pred_probs)


# Probabilities


def multilabel_py(y: np.ndarray) -> np.ndarray:
    """Compute the prior probability of each label in a multi-label classification problem.

    Parameters
    ----------
    y :
        A 2d numpy array of binarized multi-labels of shape (N, K) where N is the number of samples and K is the number of classes.

    Returns
    -------
    py :
        A 1d numpy array of prior probabilities of shape (2**K,) where 2**K is the number of possible class-assignment configurations.

    Examples
    --------
    >>> y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    >>> multilabel_py(y)
    array([0.25, 0.25, 0.25, 0.25])
    >>> y = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 0]])
    >>> multilabel_py(y)
    array([0.2, 0.2, 0.4, 0.2])
    """
    # Count the number of unique class-assignment configurations/labels
    # and the number of times each configuration occurs.
    N, K = y.shape
    unique_labels, counts = np.unique(y, axis=0, return_counts=True)
    counts = _fix_missing_class_count(K, unique_labels, counts)
    py = counts / N
    return py


def _fix_missing_class_count(K: int, unique_labels: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """If there are missing configurations, i.e. fewer than 2**K unique label, add them with a count of 0."""
    if unique_labels.shape[0] < 2**K:
        # Get the missing labels.
        all_configurations = itertools.product([0, 1], repeat=K)
        missing_labels = np.array(list(set(all_configurations) - set(map(tuple, unique_labels))))
        # Add the missing labels with a count of 0.
        unique_labels = np.vstack((unique_labels, missing_labels))
        counts = np.hstack((counts, np.zeros(missing_labels.shape[0])))
        # Sort the labels and counts by binary representation in
        # 'big' bit order:  [0, 0] < [0, 1] < [1, 0] < [1, 1])
        sorted_ids = np.argsort(np.sum(unique_labels * 2 ** np.arange(K)[::-1], axis=1))
        counts = counts[sorted_ids]
    return counts


# Cross-validation helpers


def _get_split_generator(labels, cv):
    unique_labels = np.unique(labels, axis=0)
    label_to_index = {tuple(label): i for i, label in enumerate(unique_labels)}
    multilabel_ids = np.array([label_to_index[tuple(label)] for label in labels])
    split_generator = cv.split(X=multilabel_ids, y=multilabel_ids)
    return split_generator


def get_cross_validated_multilabel_pred_probs(X, labels, *, clf, cv):
    split_generator = _get_split_generator(labels, cv)
    pred_probs = cross_val_predict(clf, X, labels, cv=split_generator, method="predict_proba")
    return pred_probs
