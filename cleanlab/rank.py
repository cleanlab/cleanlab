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
Methods to rank examples in standard (multi-class) classification datasets by cleanlab's `label quality score`.
Except for `~cleanlab.rank.order_label_issues`, which operates only on the subset of the data identified
as potential label issues/errors, the methods in this module can be used on whichever subset
of the dataset you choose (including the entire dataset) and provide a `label quality score` for
every example. You can then do something like: ``np.argsort(label_quality_score)`` to obtain ranked
indices of individual datapoints based on their quality.

Note: multi-label classification is not supported by most methods in this module,
each example must be labeled as belonging to a single class, e.g. format: ``labels = np.ndarray([1,0,2,1,1,0...])``.
For multi-label classification, instead see :py:func:`multilabel_classification.get_label_quality_scores <cleanlab.multilabel_classification.get_label_quality_scores>`.

Note: Label quality scores are most accurate when they are computed based on out-of-sample `pred_probs` from your model.
To obtain out-of-sample predicted probabilities for every datapoint in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`. This is encouraged to get better results.
"""

import numpy as np
from sklearn.metrics import log_loss
from typing import List, Optional
import warnings

from cleanlab.internal.validation import assert_valid_inputs
from cleanlab.internal.constants import (
    CLIPPING_LOWER_BOUND,
)  # lower-bound clipping threshold to prevents 0 in logs and division

from cleanlab.internal.label_quality_utils import (
    _subtract_confident_thresholds,
    get_normalized_entropy,
)


def get_label_quality_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
) -> np.ndarray:
    """Returns a label quality score for each datapoint.

    This is a function to compute label quality scores for standard (multi-class) classification datasets,
    where lower scores indicate labels less likely to be correct.

    Score is between 0 and 1.

    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels : np.ndarray
      A discrete vector of noisy labels, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, labels must be in 0, 1, ..., K-1.
      Note: multi-label classification is not supported by this method, each example must belong to a single class, e.g. format: ``labels = np.ndarray([1,0,2,1,1,0...])``.

    pred_probs : np.ndarray, optional
      An array of shape ``(N, K)`` of model-predicted probabilities,
      ``P(label=k|x)``. Each row of this matrix corresponds
      to an example `x` and contains the model-predicted probabilities that
      `x` belongs to each possible class, for each of the K classes. The
      columns must be ordered such that these probabilities correspond to
      class 0, 1, ..., K-1.

      **Note**: Returned label issues are most accurate when they are computed based on out-of-sample `pred_probs` from your model.
      To obtain out-of-sample predicted probabilities for every datapoint in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`.
      This is encouraged to get better results.

    method : {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
      Label quality scoring method.

      Letting ``k = labels[i]`` and ``P = pred_probs[i]`` denote the given label and predicted class-probabilities
      for datapoint *i*, its score can either be:

      - ``'normalized_margin'``: ``P[k] - max_{k' != k}[ P[k'] ]``
      - ``'self_confidence'``: ``P[k]``
      - ``'confidence_weighted_entropy'``: ``entropy(P) / self_confidence``

      Note: the actual label quality scores returned by this method
      may be transformed versions of the above, in order to ensure
      their values lie between 0-1 with lower values indicating more likely mislabeled data.

      Let ``C = {0, 1, ..., K-1}`` be the set of classes specified for our classification task.

      The `normalized_margin` score works better for identifying class conditional label errors,
      i.e. examples for which another label in ``C`` is appropriate but the given label is not.

      The `self_confidence` score works better for identifying alternative label issues
      corresponding to bad examples that are: not from any of the classes in ``C``,
      well-described by 2 or more labels in ``C``,
      or generally just out-of-distribution (i.e. anomalous outliers).

    adjust_pred_probs : bool, optional
      Account for class imbalance in the label-quality scoring by adjusting predicted probabilities
      via subtraction of class confident thresholds and renormalization.
      Set this to ``True`` if you prefer to account for class-imbalance.
      See `Northcutt et al., 2021 <https://jair.org/index.php/jair/article/view/12125>`_.

    Returns
    -------
    label_quality_scores : np.ndarray
      Contains one score (between 0 and 1) per example.
      Lower scores indicate more likely mislabeled examples.

    See Also
    --------
    get_self_confidence_for_each_label
    get_normalized_margin_for_each_label
    get_confidence_weighted_entropy_for_each_label
    """

    assert_valid_inputs(
        X=None, y=labels, pred_probs=pred_probs, multi_label=False, allow_one_class=True
    )
    return _compute_label_quality_scores(
        labels=labels, pred_probs=pred_probs, method=method, adjust_pred_probs=adjust_pred_probs
    )


def _compute_label_quality_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
    confident_thresholds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Internal implementation of get_label_quality_scores that assumes inputs
    have already been checked and are valid. This speeds things up.
    Can also take in pre-computed confident_thresholds to further accelerate things.
    """
    scoring_funcs = {
        "self_confidence": get_self_confidence_for_each_label,
        "normalized_margin": get_normalized_margin_for_each_label,
        "confidence_weighted_entropy": get_confidence_weighted_entropy_for_each_label,
    }
    try:
        scoring_func = scoring_funcs[method]
    except KeyError:
        raise ValueError(
            f"""
            {method} is not a valid scoring method for rank_by!
            Please choose a valid rank_by: self_confidence, normalized_margin, confidence_weighted_entropy
            """
        )
    if adjust_pred_probs:
        if method == "confidence_weighted_entropy":
            raise ValueError(f"adjust_pred_probs is not currently supported for {method}.")
        pred_probs = _subtract_confident_thresholds(
            labels=labels, pred_probs=pred_probs, confident_thresholds=confident_thresholds
        )

    scoring_inputs = {"labels": labels, "pred_probs": pred_probs}
    label_quality_scores = scoring_func(**scoring_inputs)
    return label_quality_scores


def get_label_quality_ensemble_scores(
    labels: np.ndarray,
    pred_probs_list: List[np.ndarray],
    *,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
    weight_ensemble_members_by: str = "accuracy",
    custom_weights: Optional[np.ndarray] = None,
    log_loss_search_T_values: List[float] = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 2e2],
    verbose: bool = True,
) -> np.ndarray:
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
    labels : np.ndarray
      Labels in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    pred_probs_list : List[np.ndarray]
      Each element in this list should be an array of pred_probs in the same format
      expected by the `~cleanlab.rank.get_label_quality_scores` function.
      Each element of `pred_probs_list` corresponds to the predictions from one model for all examples.

    method : {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
      Label quality scoring method. See `~cleanlab.rank.get_label_quality_scores`
      for scenarios on when to use each method.

    adjust_pred_probs : bool, optional
      `adjust_pred_probs` in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    weight_ensemble_members_by : {"uniform", "accuracy", "log_loss_search", "custom"}, default="accuracy"
      Weighting scheme used to aggregate scores from each model:

      - "uniform": Take the simple average of scores.
      - "accuracy": Take weighted average of scores, weighted by model accuracy.
      - "log_loss_search": Take weighted average of scores, weighted by exp(t * -log_loss) where t is selected from log_loss_search_T_values parameter and log_loss is the log-loss between a model's pred_probs and the given labels.
      - "custom": Take weighted average of scores using custom weights that the user passes to the custom_weights parameter.

    custom_weights : np.ndarray, default=None
      Weights used to aggregate scores from each model if weight_ensemble_members_by="custom".
      Length of this array must match the number of models: len(pred_probs_list).

    log_loss_search_T_values : List, default=[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 2e2]
      List of t values considered if weight_ensemble_members_by="log_loss_search".
      We will choose the value of t that leads to weights which produce the best log-loss when used to form a weighted average of pred_probs from the models.

    verbose : bool, default=True
      Set to ``False`` to suppress all print statements.

    Returns
    -------
    label_quality_scores : np.ndarray
      Contains one score (between 0 and 1) per example.
      Lower scores indicate more likely mislabeled examples.

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

    for pred_probs in pred_probs_list:
        assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs, multi_label=False)

    # Raise ValueError if user passed custom_weights array but did not choose weight_ensemble_members_by="custom"
    if custom_weights is not None and weight_ensemble_members_by != "custom":
        raise ValueError(
            f"""
            custom_weights provided but weight_ensemble_members_by is not "custom"!
            """
        )

    # This weighting scheme performs search of t in log_loss_search_T_values for "best" log loss
    if weight_ensemble_members_by == "log_loss_search":
        # Initialize variables for log loss search
        pred_probs_avg_log_loss_weighted = None
        neg_log_loss_weights = None
        best_eval_log_loss = float("inf")

        for t in log_loss_search_T_values:
            neg_log_loss_list = []

            # pred_probs for each model
            for pred_probs in pred_probs_list:
                pred_probs_clipped = np.clip(
                    pred_probs, a_min=CLIPPING_LOWER_BOUND, a_max=None
                )  # lower-bound clipping threshold to prevents 0 in logs when calculating log loss
                pred_probs_clipped /= pred_probs_clipped.sum(axis=1)[:, np.newaxis]  # renormalize

                neg_log_loss = np.exp(-t * log_loss(labels, pred_probs_clipped))
                neg_log_loss_list.append(neg_log_loss)

            # weights using negative log loss
            neg_log_loss_weights_temp = np.array(neg_log_loss_list) / sum(neg_log_loss_list)

            # weighted average using negative log loss
            pred_probs_avg_log_loss_weighted_temp = sum(
                [neg_log_loss_weights_temp[i] * p for i, p in enumerate(pred_probs_list)]
            )
            # evaluate log loss with this weighted average pred_probs
            eval_log_loss = log_loss(labels, pred_probs_avg_log_loss_weighted_temp)

            # check if eval_log_loss is the best so far (lower the better)
            if best_eval_log_loss > eval_log_loss:
                best_eval_log_loss = eval_log_loss
                pred_probs_avg_log_loss_weighted = pred_probs_avg_log_loss_weighted_temp
                neg_log_loss_weights = neg_log_loss_weights_temp.copy()

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
            print("Ensemble members will be weighted by their relative accuracy")
            for i, acc in enumerate(accuracy_list):
                print(f"  Model {i} accuracy : {acc}")
                print(f"  Model {i} weight   : {weights[i]}")

        # Aggregate scores with weighted average
        label_quality_scores = (scores_ensemble * weights).sum(axis=1)

    elif weight_ensemble_members_by == "log_loss_search":
        assert neg_log_loss_weights is not None
        weights = neg_log_loss_weights  # Weight by exp(t * -log_loss) where t is found by searching through log_loss_search_T_values
        if verbose:
            print(
                "Ensemble members will be weighted by log-loss between their predicted probabilities and given labels"
            )
            for i, weight in enumerate(weights):
                print(f"  Model {i} weight   : {weight}")

        # Aggregate scores with weighted average
        label_quality_scores = (scores_ensemble * weights).sum(axis=1)

    elif weight_ensemble_members_by == "custom":
        # Check custom_weights for errors
        assert (
            custom_weights is not None
        ), "custom_weights is None! Please pass a valid custom_weights."

        assert len(custom_weights) == len(
            pred_probs_list
        ), "Length of custom_weights array must match the number of models: len(pred_probs_list)."

        # Aggregate scores with custom weights
        label_quality_scores = (scores_ensemble * custom_weights).sum(axis=1)

    else:
        raise ValueError(
            f"""
            {weight_ensemble_members_by} is not a valid weighting method for weight_ensemble_members_by!
            Please choose a valid weight_ensemble_members_by: uniform, accuracy, custom
            """
        )

    return label_quality_scores


def find_top_issues(quality_scores: np.ndarray, *, top: int = 10) -> np.ndarray:
    """Returns the sorted indices of the `top` issues in `quality_scores`, ordered from smallest to largest quality score
    (i.e., from most to least likely to be an issue). For example, the first value returned is the index corresponding
    to the smallest value in `quality_scores` (most likely to be an issue). The second value in the returned array is
    the index corresponding to the second smallest value in `quality-scores` (second-most likely to be an issue), and so forth.

    This method assumes that `quality_scores` shares an index with some dataset such that the indices returned by this method
    map to the examples in that dataset.

    Parameters
    ----------
    quality_scores :
      Array of shape ``(N,)``, where N is the number of examples, containing one quality score for each example in the dataset.

    top :
      The number of indices to return.

    Returns
    -------
    top_issue_indices :
      Indices of top examples most likely to suffer from an issue (ranked by issue severity)."""

    if top is None or top > len(quality_scores):
        top = len(quality_scores)

    top_outlier_indices = quality_scores.argsort()[:top]
    return top_outlier_indices


def order_label_issues(
    label_issues_mask: np.ndarray,
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    rank_by: str = "self_confidence",
    rank_by_kwargs: dict = {},
) -> np.ndarray:
    """Sorts label issues by label quality score.

    Default label quality score is "self_confidence".

    Parameters
    ----------
    label_issues_mask : np.ndarray
      A boolean mask for the entire dataset where ``True`` represents a label
      issue and ``False`` represents an example that is accurately labeled with
      high confidence.

    labels : np.ndarray
      Labels in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    pred_probs : np.ndarray (shape (N, K))
      Predicted-probabilities in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    rank_by : str, optional
      Score by which to order label error indices (in increasing order). See
      the `method` argument of `~cleanlab.rank.get_label_quality_scores`.

    rank_by_kwargs : dict, optional
      Optional keyword arguments to pass into `~cleanlab.rank.get_label_quality_scores` function.
      Accepted args include `adjust_pred_probs`.

    Returns
    -------
    label_issues_idx : np.ndarray
      Return an array of the indices of the examples with label issues,
      ordered by the label-quality scoring method passed to `rank_by`.
    """

    allow_one_class = False
    if isinstance(labels, np.ndarray) or all(isinstance(lab, int) for lab in labels):
        if set(labels) == {0}:  # occurs with missing classes in multi-label settings
            allow_one_class = True
    assert_valid_inputs(
        X=None,
        y=labels,
        pred_probs=pred_probs,
        multi_label=False,
        allow_one_class=allow_one_class,
    )

    # Convert bool mask to index mask
    label_issues_idx = np.arange(len(labels))[label_issues_mask]

    # Calculate label quality scores
    label_quality_scores = get_label_quality_scores(
        labels, pred_probs, method=rank_by, **rank_by_kwargs
    )

    # Get label quality scores for label issues
    label_quality_scores_issues = label_quality_scores[label_issues_mask]

    return label_issues_idx[np.argsort(label_quality_scores_issues)]


def get_self_confidence_for_each_label(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> np.ndarray:
    """Returns the self-confidence label-quality score for each datapoint.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    The self-confidence is the classifier's predicted probability that an example belongs to
    its given class label.

    Self-confidence can work better than normalized-margin for detecting label errors due to out-of-distribution (OOD) or weird examples
    vs. label errors in which labels for random examples have been replaced by other classes.

    Parameters
    ----------
    labels : np.ndarray
      Labels in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    pred_probs : np.ndarray
      Predicted-probabilities in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    Returns
    -------
    label_quality_scores : np.ndarray
      Contains one score (between 0 and 1) per example.
      Lower scores indicate more likely mislabeled examples.
    """

    # To make this work for multi-label (but it will slow down runtime), return:
    # np.array([np.mean(pred_probs[i, l]) for i, l in enumerate(labels)])
    return pred_probs[np.arange(labels.shape[0]), labels]


def get_normalized_margin_for_each_label(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> np.ndarray:
    """Returns the "normalized margin" label-quality score for each datapoint.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    Letting ``k`` denote the given label for a datapoint, the margin is
    ``(p(label = k) - max(p(label != k)))``, i.e. the probability
    of the given label minus the probability of the argmax label that is not
    the given label (``margin = prob_label - max_prob_not_label``).
    This gives you an idea of how likely an example is BOTH its given label AND not another label,
    and therefore, scores its likelihood of being a good label or a label error.
    The normalized margin is simply a transformed version of the margin,
    to ensure values between 0-1 with lower values indicating more likely mislabeled data.

    Normalized margin works best for finding class conditional label errors where
    there is another label in the set of classes that is clearly better than the given label.

    Parameters
    ----------
    labels : np.ndarray
      Labels in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    pred_probs : np.ndarray
      Predicted-probabilities in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    Returns
    -------
    label_quality_scores : np.ndarray
      Contains one score (between 0 and 1) per example.
      Lower scores indicate more likely mislabeled examples.
    """

    self_confidence = get_self_confidence_for_each_label(labels, pred_probs)
    N, K = pred_probs.shape
    del_indices = np.arange(N) * K + labels
    max_prob_not_label = np.max(
        np.delete(pred_probs, del_indices, axis=None).reshape(N, K - 1), axis=-1
    )
    label_quality_scores = (self_confidence - max_prob_not_label + 1) / 2
    return label_quality_scores


def get_confidence_weighted_entropy_for_each_label(
    labels: np.ndarray, pred_probs: np.ndarray
) -> np.ndarray:
    """Returns the "confidence weighted entropy" label-quality score for each datapoint.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    "confidence weighted entropy" is defined as the normalized entropy divided by "self-confidence".
    The returned values are a transformed version of this score, in order to
    ensure values between 0-1 with lower values indicating more likely mislabeled data.

    Parameters
    ----------
    labels : np.ndarray
      Labels in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    pred_probs : np.ndarray
      Predicted-probabilities in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    Returns
    -------
    label_quality_scores : np.ndarray
      Contains one score (between 0 and 1) per example.
      Lower scores indicate more likely mislabeled examples.
    """

    self_confidence = get_self_confidence_for_each_label(labels, pred_probs)
    self_confidence = np.clip(self_confidence, a_min=CLIPPING_LOWER_BOUND, a_max=None)

    # Divide entropy by self confidence
    label_quality_scores = get_normalized_entropy(pred_probs) / self_confidence

    # Rescale
    clipped_scores = np.clip(label_quality_scores, a_min=CLIPPING_LOWER_BOUND, a_max=None)
    label_quality_scores = np.log(label_quality_scores + 1) / clipped_scores

    return label_quality_scores
