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


"""Rank module provides methods to rank/order data by cleanlab's `label quality score`.

Except for `order_label_issues`, which operates only on the subset of the data identified
as potential label issues/errors, the methods in the `rank` module can be used on whichever subset
of the dataset you choose (including the entire dataset) and provide a `label quality score` for
every example. You can then do something like: `np.argsort(label_quality_score)` to obtain ranked
indices of individual data.

If you aren't sure which method to use, try `get_normalized_margin_for_each_label()`.
"""


import numpy as np
from typing import List


def order_label_issues(
    label_issues_mask: np.array,
    labels: np.array,
    pred_probs: np.array,
    *,
    rank_by: str = "normalized_margin",
    rank_by_kwargs: dict = {},
) -> np.array:
    """Sorts label issues by label quality score.

    Default label quality score is "normalized margin".
    See https://arxiv.org/pdf/1810.05369.pdf (eqn 2.2)

    Parameters
    ----------
    label_issues_mask : np.array (bool)
      Contains True if the index of labels is an error, o.w. false

    labels : np.array
      A discrete vector of noisy labels, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    pred_probs : np.array (shape (N, K))
      P(label=k|x) is a matrix with K model-predicted probabilities.
      Each row of this matrix corresponds to an example x and contains the model-predicted
      probabilities that x belongs to each possible class.
      The columns must be ordered such that these probabilities correspond to class 0,1,2,...
      `pred_probs` should have been computed using 3 (or higher) fold cross-validation.

    rank_by : str ['normalized_margin', 'self_confidence', 'confidence_weighted_entropy']
      Method to order label error indices (instead of a bool mask), either:
        'normalized_margin' := normalized margin (p(label = k) - max(p(label != k)))
        'self_confidence' := [pred_probs[i][labels[i]] for i in label_issues_idx]
        'confidence_weighted_entropy' := entropy(pred_probs) / self_confidence

    rank_by_kwargs : dict
      Optional keyword arguments to pass into `score_label_quality()` method.
      Accepted args include:
        adj_pred_probs : bool, default = False

    Returns
    -------
    label_issues_idx : np.array (int)
      Return the index integers of the label issues, ordered by the label-quality scoring method
      passed to rank_by.

    """

    assert len(pred_probs) == len(labels)

    # Convert bool mask to index mask
    label_issues_idx = np.arange(len(labels))[label_issues_mask]

    # Calculate label quality scores
    label_quality_scores = score_label_quality(labels, pred_probs, method=rank_by, **rank_by_kwargs)

    # Get label quality scores for label issues
    label_quality_scores_issues = label_quality_scores[label_issues_mask]

    return label_issues_idx[np.argsort(label_quality_scores_issues)]


def score_label_quality(
    labels: np.array,
    pred_probs: np.array,
    method: str = "self_confidence",
    adj_pred_probs: bool = False,
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
      *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    pred_probs : np.array (shape (N, K))
      P(label=k|x) is a matrix with K model-predicted probabilities.
      Each row of this matrix corresponds to an example x and contains the model-predicted
      probabilities that x belongs to each possible class.
      The columns must be ordered such that these probabilities correspond to class 0,1,2,...
      `pred_probs` should have been computed using 3 (or higher) fold cross-validation.

    method : {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
      Label quality scoring method. Default is "self_confidence".

      .. seealso::
        :func:`self_confidence`
        :func:`normalized_margin`
        :func:`confidence_weighted_entropy`

    adj_pred_probs : bool, default = True
      Account for class-imbalance in the label-quality scoring (by adjusting predicted probabilities
      via subtraction of class confident thresholds and renormalization).
      Set this = False if you prefer to emphasize label issues in classes that are more common
      throughout the dataset
      See paper "Confident Learning: Estimating Uncertainty in Dataset Labels" by Northcutt et al.
      https://arxiv.org/abs/1911.00068

    Returns
    -------
    label_quality_scores : np.array (float)
      Scores are between 0 and 1 where lower scores indicate labels less likely to be correct.

    See Also
    --------
    self_confidence
    normalized_margin
    confidence_weighted_entropy
    subtract_confident_thresholds

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
    if adj_pred_probs:
        pred_probs = subtract_confident_thresholds(labels, pred_probs)

    # Pass keyword arguments for scoring function
    input = {"labels": labels, "pred_probs": pred_probs}

    # Calculate scores
    label_quality_scores = scoring_func(**input)

    return label_quality_scores


def score_label_quality_ensemble(
    labels: np.array,
    pred_probs_list: List[np.array],
    method: str = "self_confidence",
    adj_pred_probs: bool = False,
) -> np.array:
    """Returns label quality scores based on predictions from an ensemble of models.

    This requires a list of pred_probs from each model in the ensemble.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    Score is between 0 and 1.
    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels : np.array
      Labels in the same format expected by the `score_label_quality()` method.

    pred_probs_list : List of np.array (shape (N, K))
      Each element in this list should be an array of pred_probs in the same format
      expected by the `score_label_quality()` method

    method : {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
      Label quality scoring method. Default is "self_confidence".

      .. seealso::
        :func:`self_confidence`
        :func:`normalized_margin`
        :func:`confidence_weighted_entropy`

    adj_pred_probs : bool, default = True
      Adj_pred_probs in the same format expected by the `score_label_quality()` method.

    Returns
    -------
    label_quality_scores : np.array (float)

    See Also
    --------
    self_confidence
    normalized_margin
    confidence_weighted_entropy
    score_label_quality

    """

    # Generate scores for each model's pred_probs
    scores_list = []
    accuracy_list = []
    for pred_probs in pred_probs_list:

        # Calculate scores and accuracy
        scores = score_label_quality(
            labels=labels,
            pred_probs=pred_probs,
            method=method,
            adj_pred_probs=adj_pred_probs,
        )
        accuracy = (pred_probs.argmax(axis=1) == labels).mean()
        scores_list.append(scores)
        accuracy_list.append(accuracy)

    # Weight by accuracy
    weights = np.array(accuracy_list) / sum(accuracy_list)

    # Aggregate scores with weighted average
    label_quality_scores = (np.vstack(scores_list).T * weights).sum(axis=1)

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

    Parameters
    ----------
    labels : np.array
      Labels in the same format expected by the `score_label_quality()` method.

    pred_probs : np.array (shape (N, K))
      Predicted-probabilities in the same format expected by the `score_label_quality()` method.

    Returns
    -------
    label_quality_scores : np.array (float)
      Return the holdout probability that each example in pred_probs belongs to its
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

    The normalized margin is (p(label = k) - max(p(label != k))), i.e. the probability
    of the given label minus the probability of the argmax label that is not
    the given label. This gives you an idea of how likely an example is BOTH
    its given label AND not another label, and therefore, scores its likelihood
    of being a good label or a label error.

    Parameters
    ----------

    labels : np.array
      Labels in the same format expected by the `score_label_quality()` method.

    pred_probs : np.array (shape (N, K))
      Predicted-probabilities in the same format expected by the `score_label_quality()` method.

    Returns
    -------
    label_quality_scores : np.array (float)
      Return a score (between 0 and 1) for each example of its likelihood of
      being correctly labeled.
      normalized_margin = prob_label - max_prob_not_label

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
      Labels in the same format expected by the `score_label_quality()` method.

    pred_probs : np.array (shape (N, K))
      Predicted-probabilities in the same format expected by the `score_label_quality()` method.

    Returns
    -------
    label_quality_scores : np.array (float)
      Return a score (between 0 and 1) for each example of its likelihood of
      being correctly labeled.

    """

    self_confidence = get_self_confidence_for_each_label(labels, pred_probs)

    # Divide entropy by self confidence
    label_quality_scores = get_entropy(**{"pred_probs": pred_probs}) / self_confidence

    # Rescale
    label_quality_scores = np.log(label_quality_scores + 1) / label_quality_scores

    return label_quality_scores


def get_entropy(pred_probs: np.array) -> np.array:
    """Returns the normalized entropy of pred_probs.

    Read more about normalized entropy here: https://en.wikipedia.org/wiki/Entropy_(information_theory)

    Normalized entropy is used in active learning for uncertainty sampling: https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b

    Normalized entropy is between 0 and 1. Higher values of entropy indicate higher uncertainty in the model's prediction of the correct label.

    Parameters
    ----------
    pred_probs : np.array (shape (N, K))
      P(label=k|x) is a matrix with K model-predicted probabilities.
      Each row of this matrix corresponds to an example x and contains the model-predicted
      probabilities that x belongs to each possible class.
      The columns must be ordered such that these probabilities correspond to class 0,1,2,...
      `pred_probs` should have been computed using 3 (or higher) fold cross-validation.

    Returns
    -------
    entropy : np.array (float)

    """

    num_classes = pred_probs.shape[1]

    # Note that dividing by log(num_classes) changes the base of the log which rescales entropy to 0-1 range
    return -np.sum(pred_probs * np.log(pred_probs), axis=1) / np.log(num_classes)


def subtract_confident_thresholds(labels: np.array, pred_probs: np.array) -> np.array:
    """Returns adjusted predicted probabilities by subtracting the class confident thresholds and renormalizing.

    The confident class threshold for a class j is the expected (average) "self-confidence" for class j.

    The purpose of this adjustment is to handle class imbalance.

    Parameters
    ----------
    labels : np.array
      A discrete vector of noisy labels, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    pred_probs : np.array (shape (N, K))
      P(label=k|x) is a matrix with K model-predicted probabilities.
      Each row of this matrix corresponds to an example x and contains the model-predicted
      probabilities that x belongs to each possible class.
      The columns must be ordered such that these probabilities correspond to class 0,1,2,...
      `pred_probs` should have been computed using 3 (or higher) fold cross-validation.

    Returns
    -------
    pred_probs_adj : np.array (float)
    """

    # Get expected (average) self-confidence for each class
    confident_thresholds = get_confident_thresholds(labels, pred_probs)

    # Subtract the class confident thresholds
    pred_probs_adj = pred_probs - confident_thresholds

    # Renormalize by shifting data to take care of negative values from the subtraction
    pred_probs_adj += 1
    pred_probs_adj /= pred_probs_adj.sum(axis=1)[:, None]

    return pred_probs_adj


def get_confident_thresholds(labels: np.array, pred_probs: np.array) -> np.array:
    """Returns expected (average) "self-confidence" for each class.

    The confident class threshold for a class j is the expected (average) "self-confidence" for class j.

    Parameters
    ----------
    labels : np.array
      A discrete vector of noisy labels, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    pred_probs : np.array (shape (N, K))
      P(label=k|x) is a matrix with K model-predicted probabilities.
      Each row of this matrix corresponds to an example x and contains the model-predicted
      probabilities that x belongs to each possible class.
      The columns must be ordered such that these probabilities correspond to class 0,1,2,...
      `pred_probs` should have been computed using 3 (or higher) fold cross-validation.

    Returns
    -------
    confident_thresholds : np.array (shape (K,))

    """
    confident_thresholds = np.array(
        [np.mean(pred_probs[:, k][labels == k]) for k in range(pred_probs.shape[1])]
    )
    return confident_thresholds
