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
Methods to rank/order data by cleanlab's `label quality score`.
Except for :py:func:`order_label_issues <cleanlab.rank.order_label_issues>`, which operates only on the subset of the data identified
as potential label issues/errors, the methods in this module can be used on whichever subset
of the dataset you choose (including the entire dataset) and provide a `label quality score` for
every example. You can then do something like: ``np.argsort(label_quality_score)`` to obtain ranked
indices of individual data.

CAUTION: These label quality scores are computed based on `pred_probs` from your model that must be out-of-sample!
You should never provide predictions on the same examples used to train the model,
as these will be overfit and unsuitable for finding label-errors.
To obtain out-of-sample predicted probabilities for every datapoint in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`.
Alternatively it is ok if your model was trained on a separate dataset and you are only evaluating
labels in data that was previously held-out.
"""


import numpy as np
from typing import List
import warnings
from cleanlab.internal.label_quality_utils import (
    _subtract_confident_thresholds,
    get_normalized_entropy,
)


def order_label_issues(
    label_issues_mask: np.array,
    labels: np.array,
    pred_probs: np.array,
    *,
    rank_by: str = "self_confidence",
    rank_by_kwargs: dict = {},
) -> np.array:
    """Sorts label issues by label quality score.

    Default label quality score is "self_confidence".

    Parameters
    ----------
    label_issues_mask : np.array
      A boolean mask for the entire dataset where ``True`` represents a label
      issue and ``False`` represents an example that is accurately labeled with
      high confidence.

    labels : np.array
      Labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    pred_probs : np.array (shape (N, K))
      Predicted-probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    rank_by : str, optional
      Score by which to order label error indices (in increasing order). See
      the `method` argument of :py:func:`get_label_quality_scores
      <cleanlab.rank.get_label_quality_scores>`.

    rank_by_kwargs : dict, optional
      Optional keyword arguments to pass into :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.
      Accepted args include `adjust_pred_probs`.

    Returns
    -------
    label_issues_idx : np.array
      Return an array of the indices of the label issues, ordered by the label-quality scoring method
      passed to `rank_by`.

    """

    assert len(pred_probs) == len(labels)

    # Convert bool mask to index mask
    label_issues_idx = np.arange(len(labels))[label_issues_mask]

    # Calculate label quality scores
    label_quality_scores = get_label_quality_scores(
        labels, pred_probs, method=rank_by, **rank_by_kwargs
    )

    # Get label quality scores for label issues
    label_quality_scores_issues = label_quality_scores[label_issues_mask]

    return label_issues_idx[np.argsort(label_quality_scores_issues)]


def get_label_quality_scores(
    labels: np.array,
    pred_probs: np.array,
    *,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
) -> np.array:
    """Returns label quality scores for each datapoint.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    Score is between 0 and 1.

    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels : np.array
      A discrete vector of noisy labels, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, labels must be in 0, 1, ..., K-1.

    pred_probs : np.array, optional
      An array of shape ``(N, K)`` of model-predicted probabilities,
      ``P(label=k|x)``. Each row of this matrix corresponds
      to an example `x` and contains the model-predicted probabilities that
      `x` belongs to each possible class, for each of the K classes. The
      columns must be ordered such that these probabilities correspond to
      class 0, 1, ..., K-1.

      **Caution**: `pred_probs` from your model must be out-of-sample!
      You should never provide predictions on the same examples used to train the model,
      as these will be overfit and unsuitable for finding label-errors.
      To obtain out-of-sample predicted probabilities for every datapoint in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`.
      Alternatively it is ok if your model was trained on a separate dataset and you are only evaluating
      data that was previously held-out.

    method : {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
      Label quality scoring method.

      Letting ``k = labels[i]`` and ``P = pred_probs[i]`` denote the given label and predicted class-probabilities
      for datapoint *i*, its score can either be:

      - ``'normalized_margin'``: ``P[k] - max_{k' != k}[ P[k'] ]``
      - ``'self_confidence'``: ``P[k]``
      - ``'confidence_weighted_entropy'``: ``entropy(P) / self_confidence``

      Let ``C = {0, 1, ..., K}`` denote the classification task's specified set of classes.

      The normalized_margin score works better for identifying class conditional label errors,
      i.e. examples for which another label in C is appropriate but the given label is not.

      The self_confidence score works better for identifying alternative label issues corresponding
      to bad examples that are: not from any of the classes in C, well-described by 2 or more labels in C,
      or generally just out-of-distribution (ie. anomalous outliers).

    adjust_pred_probs : bool, optional
      Account for class imbalance in the label-quality scoring by adjusting predicted probabilities
      via subtraction of class confident thresholds and renormalization.
      Set this to ``True`` if you prefer to account for class-imbalance.
      See `Northcutt et al., 2021 <https://jair.org/index.php/jair/article/view/12125>`_.

    Returns
    -------
    label_quality_scores : np.array
      Scores are between 0 and 1 where lower scores indicate labels less likely to be correct.

    See Also
    --------
    get_self_confidence_for_each_label
    get_normalized_margin_for_each_label
    get_confidence_weighted_entropy_for_each_label

    """

    # Available scoring functions to choose from
    scoring_funcs = {
        "self_confidence": get_self_confidence_for_each_label,
        "normalized_margin": get_normalized_margin_for_each_label,
        "confidence_weighted_entropy": get_confidence_weighted_entropy_for_each_label,
    }

    # Select scoring function
    try:
        scoring_func = scoring_funcs[method]
    except KeyError:
        raise ValueError(
            f"""
            {method} is not a valid scoring method for rank_by!
            Please choose a valid rank_by: self_confidence, normalized_margin, confidence_weighted_entropy
            """
        )

    # Adjust predicted probabilities
    if adjust_pred_probs:

        # Check if adjust_pred_probs is supported for the chosen method
        if method == "confidence_weighted_entropy":
            raise ValueError(f"adjust_pred_probs is not currently supported for {method}.")

        pred_probs = _subtract_confident_thresholds(labels, pred_probs)

    # Pass keyword arguments for scoring function
    input = {"labels": labels, "pred_probs": pred_probs}

    # Calculate scores
    label_quality_scores = scoring_func(**input)

    return label_quality_scores


def get_label_quality_ensemble_scores(
    labels: np.array,
    pred_probs_list: List[np.array],
    *,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
    weight_ensemble_members_by: str = "accuracy",
    verbose: bool = True,
) -> np.array:
    """Returns label quality scores based on predictions from an ensemble of models.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    Ensemble scoring requires a list of pred_probs from each model in the ensemble.

    For each pred_probs in list, compute label quality score.
    Take the average of the scores with the chosen weighting scheme determined by `weight_ensemble_members_by`.

    Score is between 0 and 1:

    - 1 --- clean label (given label is likely correct).
    - 0 --- dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels : np.array
      Labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    pred_probs_list : List[np.array]
      Each element in this list should be an array of pred_probs in the same format
      expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.
      Each element of `pred_probs_list` corresponds to the predictions from one model for all examples.

    method : {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
      Label quality scoring method. See :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`
      for scenarios on when to use each method.

    adjust_pred_probs : bool, optional
      `adjust_pred_probs` in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    weight_ensemble_members_by : {"uniform", "accuracy"}, default="accuracy"
      Weighting scheme used to aggregate scores from each model:

      - "uniform": take the simple average of scores
      - "accuracy": take weighted average of scores, weighted by model accuracy

    verbose : bool, default=True
      Set to ``False`` to suppress all print statements.

    Returns
    -------
    label_quality_scores : np.array

    See Also
    --------
    get_label_quality_scores

    """

    # Check pred_probs_list for errors
    assert isinstance(
        pred_probs_list, list
    ), f"pred_probs_list needs to be a list. Provided pred_probs_list is a {type(pred_probs_list)}"

    assert len(pred_probs_list) > 0, "pred_probs_list is empty."

    if len(pred_probs_list) == 1:
        warnings.warn(
            """
            pred_probs_list only has one element.
            Consider using get_label_quality_scores() if you only have a single array of pred_probs.
            """
        )

    # Generate scores for each model's pred_probs
    scores_list = []
    accuracy_list = []
    for pred_probs in pred_probs_list:

        # Calculate scores and accuracy
        scores = get_label_quality_scores(
            labels=labels,
            pred_probs=pred_probs,
            method=method,
            adjust_pred_probs=adjust_pred_probs,
        )
        scores_list.append(scores)

        # Only compute if weighting by accuracy
        if weight_ensemble_members_by == "accuracy":
            accuracy = (pred_probs.argmax(axis=1) == labels).mean()
            accuracy_list.append(accuracy)

    if verbose:
        print(f"Weighting scheme for ensemble: {weight_ensemble_members_by}")

    # Transform list of scores into an array of shape (N, M) where M is the number of models in the ensemble
    scores_ensemble = np.vstack(scores_list).T

    # Aggregate scores with chosen weighting scheme
    if weight_ensemble_members_by == "uniform":
        label_quality_scores = scores_ensemble.mean(axis=1)  # Uniform weights (simple average)

    elif weight_ensemble_members_by == "accuracy":
        weights = np.array(accuracy_list) / sum(accuracy_list)  # Weight by relative accuracy
        if verbose:
            print("Ensemble members will be weighted by: their relative accuracy")
            for i, acc in enumerate(accuracy_list):
                print(f"  Model {i} accuracy : {acc}")
                print(f"  Model {i} weights  : {weights[i]}")

        # Aggregate scores with weighted average
        label_quality_scores = (scores_ensemble * weights).sum(axis=1)

    else:
        raise ValueError(
            f"""
            {weight_ensemble_members_by} is not a valid weighting method for weight_ensemble_members_by!
            Please choose a valid weight_ensemble_members_by: uniform, accuracy
            """
        )

    return label_quality_scores


def get_self_confidence_for_each_label(
    labels: np.array,
    pred_probs: np.array,
) -> np.array:
    """Returns the self-confidence label-quality score for each datapoint.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    The self-confidence is the holdout probability that an example belongs to
    its given class label.

    Self-confidence works better for finding out-of-distribution (OOD) examples, weird examples, bad examples,
    multi-label, and other types of label errors.

    Parameters
    ----------
    labels : np.array
      Labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    pred_probs : np.array
      Predicted-probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    Returns
    -------
    label_quality_scores : np.array
      An array of holdout probabilities that each example in `pred_probs` belongs to its
      label.

    """

    # np.mean is used so that this works for multi-labels (list of lists)
    label_quality_scores = np.array([np.mean(pred_probs[i, l]) for i, l in enumerate(labels)])
    return label_quality_scores


def get_normalized_margin_for_each_label(
    labels: np.array,
    pred_probs: np.array,
) -> np.array:
    """Returns the "normalized margin" label-quality score for each datapoint.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    Letting k denote the given label for a datapoint, the normalized margin is
    ``(p(label = k) - max(p(label != k)))``, i.e. the probability
    of the given label minus the probability of the argmax label that is not
    the given label. This gives you an idea of how likely an example is BOTH
    its given label AND not another label, and therefore, scores its likelihood
    of being a good label or a label error.

    Normalized margin works better for finding class conditional label errors where
    there is another label in the class that is better than the given label.

    Parameters
    ----------
    labels : np.array
      Labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    pred_probs : np.array
      Predicted-probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    Returns
    -------
    label_quality_scores : np.array
      An array of scores (between 0 and 1) for each example of its likelihood of
      being correctly labeled. ``normalized_margin = prob_label - max_prob_not_label``
    """

    self_confidence = get_self_confidence_for_each_label(labels, pred_probs)
    max_prob_not_label = np.array(
        [max(np.delete(pred_probs[i], l, -1)) for i, l in enumerate(labels)]
    )
    label_quality_scores = (self_confidence - max_prob_not_label + 1) / 2
    return label_quality_scores


def get_confidence_weighted_entropy_for_each_label(
    labels: np.array, pred_probs: np.array
) -> np.array:
    """Returns the "confidence weighted entropy" label-quality score for each datapoint.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    "confidence weighted entropy" is the normalized entropy divided by "self-confidence".

    Parameters
    ----------
    labels : np.array
      Labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    pred_probs : np.array
      Predicted-probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    Returns
    -------
    label_quality_scores : np.array
      An array of scores (between 0 and 1) for each example of its likelihood of
      being correctly labeled.
    """

    MIN_ALLOWED_RAW_SCORE = 1e-6 - 1  # lower-bound clipping threshold to prevent 0 in logarithm
    self_confidence = get_self_confidence_for_each_label(labels, pred_probs)

    # Divide entropy by self confidence
    label_quality_scores = get_normalized_entropy(**{"pred_probs": pred_probs}) / self_confidence

    # Rescale
    label_quality_scores = np.clip(label_quality_scores, a_min=MIN_ALLOWED_RAW_SCORE, a_max=None)
    label_quality_scores = np.log(label_quality_scores + 1) / label_quality_scores

    return label_quality_scores
