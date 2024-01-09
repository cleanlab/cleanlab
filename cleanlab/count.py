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
Methods to estimate latent structures used for confident learning, including:

* Latent prior of the unobserved, error-less labels: `py`: ``p(y)``
* Latent noisy channel (noise matrix) characterizing the flipping rates: `nm`: ``P(given label | true label)``
* Latent inverse noise matrix characterizing the flipping process: `inv`: ``P(true label | given label)``
* Latent `confident_joint`, an un-normalized matrix that counts the confident subset of label errors under the joint distribution for true/given label

These are estimated from a classification dataset. This module considers two types of datasets:

* standard (multi-class) classification where each example is labeled as belonging to exactly one of K classes (e.g. ``labels = np.array([0,0,1,0,2,1])``)
* multi-label classification where each example can be labeled as belonging to multiple classes (e.g. ``labels = [[1,2],[1],[0],[],...]``)
"""

from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import sklearn.base
import numpy as np
import warnings
from typing import Tuple, Union, Optional

from cleanlab.typing import LabelLike
from cleanlab.internal.multilabel_utils import stack_complement, get_onehot_num_classes
from cleanlab.internal.constants import (
    TINY_VALUE,
    CONFIDENT_THRESHOLDS_LOWER_BOUND,
    FLOATING_POINT_COMPARISON,
)

from cleanlab.internal.util import (
    value_counts_fill_missing_classes,
    clip_values,
    clip_noise_rates,
    round_preserving_row_totals,
    append_extra_datapoint,
    train_val_split,
    get_num_classes,
    get_unique_classes,
    is_torch_dataset,
    is_tensorflow_dataset,
)
from cleanlab.internal.latent_algebra import (
    compute_inv_noise_matrix,
    compute_py,
    compute_noise_matrix_from_inverse,
)
from cleanlab.internal.validation import (
    assert_valid_inputs,
    labels_to_array,
)


def num_label_issues(
    labels: LabelLike,
    pred_probs: np.ndarray,
    *,
    confident_joint: Optional[np.ndarray] = None,
    estimation_method: str = "off_diagonal",
    multi_label: bool = False,
) -> int:
    """Estimates the number of label issues in a classification dataset. Use this method to get the most accurate
    estimate of number of label issues when you don't need the indices of the examples with label issues.

    Parameters
    ----------
    labels : np.ndarray or list
      Given class labels for each example in the dataset, some of which may be erroneous,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    pred_probs :
      Model-predicted class probabilities for each example in the dataset,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    confident_joint :
      Array of estimated class label error statisics used for identifying label issues,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.
      The `confident_joint` can be computed using `~cleanlab.count.compute_confident_joint`.
      It is internally computed from the given (noisy) `labels` and `pred_probs`.

    estimation_method :
      Method for estimating the number of label issues in dataset by counting the examples in the off-diagonal of the `confident_joint` ``P(label=i, true_label=j)``.

      * ``'off_diagonal'``: Counts the number of examples in the off-diagonal of the `confident_joint`. Returns the same value as ``sum(find_label_issues(filter_by='confident_learning'))``

      * ``'off_diagonal_calibrated'``: Calibrates confident joint estimate ``P(label=i, true_label=j)`` such that
        ``np.sum(cj) == len(labels)`` and ``np.sum(cj, axis = 1) == np.bincount(labels)`` before counting the number
        of examples in the off-diagonal. Number will always be equal to or greater than
        ``estimate_issues='off_diagonal'``. You can use this value as the cutoff threshold used with ranking/scoring
        functions from :py:mod:`cleanlab.rank` with `num_label_issues` over ``estimation_method='off_diagonal'`` in
        two cases:

        #. As we add more label and data quality scoring functions in :py:mod:`cleanlab.rank`, this approach will always work.
        #. If you have a custom score to rank your data by label quality and you just need to know the cut-off of likely label issues.

      * ``'off_diagonal_custom'``: Counts the number of examples in the off-diagonal of a provided `confident_joint` matrix.

      TL;DR: Use this method to get the most accurate estimate of number of label issues when you don't need the indices of the label issues.

      Note: ``'off_diagonal'`` may sometimes underestimate issues for data with few classes, so consider using ``'off_diagonal_calibrated'`` instead if your data has < 4 classes.

    multi_label : bool, optional
      Set ``False`` if your dataset is for regular (multi-class) classification, where each example belongs to exactly one class.
      Set ``True`` if your dataset is for multi-label classification, where each example can belong to multiple classes.
      See documentation of `~cleanlab.count.compute_confident_joint` for details.

    Returns
    -------
    num_issues :
      The estimated number of examples with label issues in the dataset.
    """
    valid_methods = ["off_diagonal", "off_diagonal_calibrated", "off_diagonal_custom"]
    if isinstance(confident_joint, np.ndarray) and estimation_method != "off_diagonal_custom":
        warn_str = (
            "The supplied `confident_joint` is ignored as `confident_joint` is recomuputed internally using "
            "the supplied `labels` and `pred_probs`. If you still want to use custom `confident_joint` call function "
            "with `estimation_method='off_diagonal_custom'`."
        )
        warnings.warn(warn_str)

    if multi_label:
        return _num_label_issues_multilabel(
            labels=labels,
            pred_probs=pred_probs,
            confident_joint=confident_joint,
        )
    labels = labels_to_array(labels)
    assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs)

    if estimation_method == "off_diagonal":
        _, cl_error_indices = compute_confident_joint(
            labels=labels,
            pred_probs=pred_probs,
            calibrate=False,
            return_indices_of_off_diagonals=True,
        )

        label_issues_mask = np.zeros(len(labels), dtype=bool)
        label_issues_mask[cl_error_indices] = True

        # Remove label issues if model prediction is close to given label
        mask = _reduce_issues(pred_probs=pred_probs, labels=labels)
        label_issues_mask[mask] = False
        num_issues = np.sum(label_issues_mask)
    elif estimation_method == "off_diagonal_calibrated":
        calculated_confident_joint = compute_confident_joint(
            labels=labels,
            pred_probs=pred_probs,
            calibrate=True,
        )
        assert isinstance(calculated_confident_joint, np.ndarray)
        # Estimate_joint calibrates the row sums to match the prior distribution of given labels and normalizes to sum to 1
        joint = estimate_joint(labels, pred_probs, confident_joint=calculated_confident_joint)
        frac_issues = 1.0 - joint.trace()
        num_issues = np.rint(frac_issues * len(labels)).astype(int)
    elif estimation_method == "off_diagonal_custom":
        if not isinstance(confident_joint, np.ndarray):
            raise ValueError(
                f"""
                No `confident_joint` provided. For 'estimation_method' = {estimation_method} you need to provide pre-calculated
                `confident_joint` matrix. Use a different `estimation_method` if you want the `confident_joint` matrix to
                be calculated for you.
                """
            )
        else:
            joint = estimate_joint(labels, pred_probs, confident_joint=confident_joint)
            frac_issues = 1.0 - joint.trace()
            num_issues = np.rint(frac_issues * len(labels)).astype(int)
    else:
        raise ValueError(
            f"""
                {estimation_method} is not a valid estimation method!
                Please choose a valid estimation method: {valid_methods}
                """
        )

    return num_issues


def _num_label_issues_multilabel(
    labels: LabelLike,
    pred_probs: np.ndarray,
    confident_joint: Optional[np.ndarray] = None,
) -> int:
    """
    Parameters
    ----------
    labels: list
       Refer to documentation for this argument in ``count.calibrate_confident_joint()`` with `multi_label=True` for details.

    pred_probs : np.ndarray
       Predicted-probabilities in the same format expected by the `~cleanlab.count.get_confident_thresholds` function.

    Returns
    -------
    num_issues : int
       The estimated number of examples with label issues in the multi-label dataset.

    Note: We set the filter_by method as 'confident_learning' to match the non-multilabel case
    (analog to the off_diagonal estimation method)
    """

    from cleanlab.filter import find_label_issues

    issues_idx = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        confident_joint=confident_joint,
        multi_label=True,
        filter_by="confident_learning",  # specified to match num_label_issues
    )
    return sum(issues_idx)


def _reduce_issues(pred_probs, labels):
    """Returns a boolean mask denoting correct predictions or predictions within a margin around 0.5 for binary classification, suitable for filtering out indices in 'is_label_issue'."""
    pred_probs_copy = np.copy(pred_probs)  # Make a copy of the original array
    pred_probs_copy[np.arange(len(labels)), labels] += FLOATING_POINT_COMPARISON
    pred = pred_probs_copy.argmax(axis=1)
    mask = pred == labels
    del pred_probs_copy  # Delete copy
    return mask


def calibrate_confident_joint(
    confident_joint: np.ndarray, labels: LabelLike, *, multi_label: bool = False
) -> np.ndarray:
    """Calibrates any confident joint estimate ``P(label=i, true_label=j)`` such that
    ``np.sum(cj) == len(labels)`` and ``np.sum(cj, axis = 1) == np.bincount(labels)``.

    In other words, this function forces the confident joint to have the
    true noisy prior ``p(labels)`` (summed over columns for each row) and also
    forces the confident joint to add up to the total number of examples.

    This method makes the confident joint a valid counts estimate
    of the actual joint of noisy and true labels.

    Parameters
    ----------
    confident_joint : np.ndarray
      An array of shape ``(K, K)`` representing the confident joint, the matrix used for identifying label issues, which
      estimates a confident subset of the joint distribution of the noisy and true labels, ``P_{noisy label, true label}``.
      Entry ``(j, k)`` in the matrix is the number of examples confidently counted into the pair of ``(noisy label=j, true label=k)`` classes.
      The `confident_joint` can be computed using `~cleanlab.count.compute_confident_joint`.
      If not provided, it is computed from the given (noisy) `labels` and `pred_probs`.
      If `multi_label` is True, then the `confident_joint` should be a one-vs-rest array of shape ``(K, 2, 2)``, and an array of the same shape will be returned.

    labels : np.ndarray or list
      Given class labels for each example in the dataset, some of which may be erroneous,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    multi_label : bool, optional
      If ``False``, dataset is for regular (multi-class) classification, where each example belongs to exactly one class.
      If ``True``, dataset is for multi-label classification, where each example can belong to multiple classes.
      See documentation of `~cleanlab.count.compute_confident_joint` for details.
      In multi-label classification, the confident/calibrated joint arrays have shape ``(K, 2, 2)``
      formatted in a one-vs-rest fashion such that they contain a 2x2 matrix for each class
      that counts examples which are correctly/incorrectly labeled as belonging to that class.
      After calibration, the entries in each class-specific 2x2 matrix will sum to the number of examples.

    Returns
    -------
    calibrated_cj : np.ndarray
      An array of shape ``(K, K)`` representing a valid estimate of the joint *counts* of noisy and true labels (if `multi_label` is False).
      If `multi_label` is True, the returned `calibrated_cj` is instead an one-vs-rest array of shape ``(K, 2, 2)``,
      where for class `c`: entry ``(c, 0, 0)`` in this one-vs-rest  array is the number of examples whose noisy label contains `c` confidently identified as truly belonging to class `c` as well.
      Entry ``(c, 1, 0)`` in this one-vs-rest  array is the number of examples whose noisy label contains `c` confidently identified as not actually belonging to class `c`.
      Entry ``(c, 0, 1)`` in this one-vs-rest array is the number of examples whose noisy label does not contain `c` confidently identified as truly belonging to class `c`.
      Entry ``(c, 1, 1)`` in this one-vs-rest array is the number of examples whose noisy label does not contain `c` confidently identified as actually not belonging to class `c` as well.

    """

    if multi_label:
        if not isinstance(labels, list):
            raise TypeError("`labels` must be list when `multi_label=True`.")
        else:
            return _calibrate_confident_joint_multilabel(confident_joint, labels)
    else:
        num_classes = len(confident_joint)
        label_counts = value_counts_fill_missing_classes(labels, num_classes, multi_label=False)
    # Calibrate confident joint to have correct p(labels) prior on noisy labels.
    calibrated_cj = (
        confident_joint.T
        / np.clip(confident_joint.sum(axis=1), a_min=TINY_VALUE, a_max=None)
        * label_counts
    ).T
    # Calibrate confident joint to sum to:
    # The number of examples (for single labeled datasets)
    # The number of total labels (for multi-labeled datasets)
    calibrated_cj = (
        calibrated_cj
        / np.clip(np.sum(calibrated_cj), a_min=TINY_VALUE, a_max=None)
        * sum(label_counts)
    )
    return round_preserving_row_totals(calibrated_cj)


def _calibrate_confident_joint_multilabel(confident_joint: np.ndarray, labels: list) -> np.ndarray:
    """Calibrates the confident joint for multi-label classification data. Here
        input `labels` is a list of lists (or list of iterable).
        This is intended as a helper function. You should probably
        be using `calibrate_confident_joint(multi_label=True)` instead.


        See `calibrate_confident_joint` docstring for more info.

    Parameters
    ----------
    confident_joint : np.ndarray
        Refer to documentation for this argument in count.calibrate_confident_joint() for details.

    labels : list
        Refer to documentation for this argument in count.calibrate_confident_joint() for details.

    multi_label : bool, optional
        Refer to documentation for this argument in count.calibrate_confident_joint() for details.

    Returns
    -------
    calibrated_cj : np.ndarray
      An array of shape ``(K, 2, 2)`` of type float representing a valid
      estimate of the joint *counts* of noisy and true labels in a one-vs-rest fashion."""
    y_one, num_classes = get_onehot_num_classes(labels)
    calibrate_confident_joint_list: np.ndarray = np.ndarray(
        shape=(num_classes, 2, 2), dtype=np.int64
    )
    for class_num, (cj, y) in enumerate(zip(confident_joint, y_one.T)):
        calibrate_confident_joint_list[class_num] = calibrate_confident_joint(cj, labels=y)

    return calibrate_confident_joint_list


def estimate_joint(
    labels: LabelLike,
    pred_probs: np.ndarray,
    *,
    confident_joint: Optional[np.ndarray] = None,
    multi_label: bool = False,
) -> np.ndarray:
    """
    Estimates the joint distribution of label noise ``P(label=i, true_label=j)`` guaranteed to:

    * Sum to 1
    * Satisfy ``np.sum(joint_estimate, axis = 1) == p(labels)``

    Parameters
    ----------
    labels : np.ndarray or list
      Given class labels for each example in the dataset, some of which may be erroneous,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    pred_probs : np.ndarray
      Model-predicted class probabilities for each example in the dataset,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    confident_joint : np.ndarray, optional
      Array of estimated class label error statisics used for identifying label issues,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.
      The `confident_joint` can be computed using `~cleanlab.count.compute_confident_joint`.
      If not provided, it is internally computed from the given (noisy) `labels` and `pred_probs`.

    multi_label : bool, optional
      If ``False``, dataset is for regular (multi-class) classification, where each example belongs to exactly one class.
      If ``True``, dataset is for multi-label classification, where each example can belong to multiple classes.
      See documentation of `~cleanlab.count.compute_confident_joint` for details.

    Returns
    -------
    confident_joint_distribution : np.ndarray
      An array of shape ``(K, K)`` representing an
      estimate of the true joint distribution of noisy and true labels (if `multi_label` is False).
      If `multi_label` is True, an array of shape ``(K, 2, 2)`` representing an
      estimate of the true joint distribution of noisy and true labels for each class in a one-vs-rest fashion.
      Entry ``(c, i, j)`` in this array is the number of examples confidently counted into a ``(class c, noisy label=i, true label=j)`` bin,
      where `i, j` are either 0 or 1 to denote whether this example belongs to class `c` or not
      (recall examples can belong to multiple classes in multi-label classification).
    """

    if confident_joint is None:
        calibrated_cj = compute_confident_joint(
            labels,
            pred_probs,
            calibrate=True,
            multi_label=multi_label,
        )
    else:
        if labels is not None:
            calibrated_cj = calibrate_confident_joint(
                confident_joint, labels, multi_label=multi_label
            )
        else:
            calibrated_cj = confident_joint

    assert isinstance(calibrated_cj, np.ndarray)
    if multi_label:
        if not isinstance(labels, list):
            raise TypeError("`labels` must be list when `multi_label=True`.")
        else:
            return _estimate_joint_multilabel(
                labels=labels, pred_probs=pred_probs, confident_joint=confident_joint
            )
    else:
        return calibrated_cj / np.clip(float(np.sum(calibrated_cj)), a_min=TINY_VALUE, a_max=None)


def _estimate_joint_multilabel(
    labels: list, pred_probs: np.ndarray, *, confident_joint: Optional[np.ndarray] = None
) -> np.ndarray:
    """Parameters
     ----------
     labels : list
      Refer to documentation for this argument in filter.find_label_issues() for details.

     pred_probs : np.ndarray
       Refer to documentation for this argument in count.estimate_joint() for details.

    confident_joint : np.ndarray, optional
       Refer to documentation for this argument in filter.find_label_issues() with multi_label=True for details.

     Returns
     -------
     confident_joint_distribution : np.ndarray
       An array of shape ``(K, 2, 2)`` representing an
       estimate of the true joint distribution of noisy and true labels for each class, in a one-vs-rest format employed for multi-label settings.
    """
    y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
    if confident_joint is None:
        calibrated_cj = compute_confident_joint(
            labels,
            pred_probs,
            calibrate=True,
            multi_label=True,
        )
    else:
        calibrated_cj = confident_joint
    assert isinstance(calibrated_cj, np.ndarray)
    calibrated_cf: np.ndarray = np.ndarray((num_classes, 2, 2))
    for class_num, (label, pred_prob_for_class) in enumerate(zip(y_one.T, pred_probs.T)):
        pred_probs_binary = stack_complement(pred_prob_for_class)
        calibrated_cf[class_num] = estimate_joint(
            labels=label,
            pred_probs=pred_probs_binary,
            confident_joint=calibrated_cj[class_num],
        )

    return calibrated_cf


def compute_confident_joint(
    labels: LabelLike,
    pred_probs: np.ndarray,
    *,
    thresholds: Optional[Union[np.ndarray, list]] = None,
    calibrate: bool = True,
    multi_label: bool = False,
    return_indices_of_off_diagonals: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    """Estimates the confident counts of latent true vs observed noisy labels
    for the examples in our dataset. This array of shape ``(K, K)`` is called the **confident joint**
    and contains counts of examples in every class, confidently labeled as every other class.
    These counts may subsequently be used to estimate the joint distribution of true and noisy labels
    (by normalizing them to frequencies).

    Important: this function assumes that `pred_probs` are out-of-sample
    holdout probabilities. This can be :ref:`done with cross validation <pred_probs_cross_val>`. If
    the probabilities are not computed out-of-sample, overfitting may occur.

    Parameters
    ----------
    labels : np.ndarray or list
      Given class labels for each example in the dataset, some of which may be erroneous,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    pred_probs : np.ndarray
      Model-predicted class probabilities for each example in the dataset,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    thresholds : array_like, optional
      An array of shape ``(K, 1)`` or ``(K,)`` of per-class threshold
      probabilities, used to determine the cutoff probability necessary to
      consider an example as a given class label (see `Northcutt et al.,
      2021 <https://jair.org/index.php/jair/article/view/12125>`_, Section
      3.1, Equation 2).

      This is for advanced users only. If not specified, these are computed
      for you automatically. If an example has a predicted probability
      greater than this threshold, it is counted as having true_label =
      k. This is not used for pruning/filtering, only for estimating the
      noise rates using confident counts.

    calibrate : bool, default=True
        Calibrates confident joint estimate ``P(label=i, true_label=j)`` such that
        ``np.sum(cj) == len(labels)`` and ``np.sum(cj, axis = 1) == np.bincount(labels)``.
        When ``calibrate=True``, this method returns an estimate of
        the latent true joint counts of noisy and true labels.

    multi_label : bool, optional
      If ``True``, this is multi-label classification dataset (where each example can belong to more than one class)
      rather than a regular (multi-class) classifiction dataset.
      In this case, `labels` should be an iterable (e.g. list) of iterables (e.g. ``List[List[int]]``),
      containing the list of classes to which each example belongs, instead of just a single class.
      Example of `labels` for a multi-label classification dataset: ``[[0,1], [1], [0,2], [0,1,2], [0], [1], [], ...]``.

    return_indices_of_off_diagonals : bool, optional
      If ``True``, returns indices of examples that were counted in off-diagonals
      of confident joint as a baseline proxy for the label issues. This
      sometimes works as well as ``filter.find_label_issues(confident_joint)``.


    Returns
    -------
    confident_joint_counts : np.ndarray
      An array of shape ``(K, K)`` representing counts of examples
      for which we are confident about their given and true label (if `multi_label` is False).
      If `multi_label` is True,
      this array instead has shape ``(K, 2, 2)`` representing a one-vs-rest format for the  confident joint, where for each class `c`:
      Entry ``(c, 0, 0)`` in this one-vs-rest array is the number of examples whose noisy label contains `c` confidently identified as truly belonging to class `c` as well.
      Entry ``(c, 1, 0)`` in this one-vs-rest array is the number of examples whose noisy label contains `c` confidently identified as not actually belonging to class `c`.
      Entry ``(c, 0, 1)`` in this one-vs-rest array is the number of examples whose noisy label does not contain `c` confidently identified as truly belonging to class `c`.
      Entry ``(c, 1, 1)`` in this one-vs-rest array is the number of examples whose noisy label does not contain `c` confidently identified as actually not belonging to class `c` as well.


      Note
      ----
      If `return_indices_of_off_diagonals` is set as True, this function instead returns a tuple `(confident_joint, indices_off_diagonal)`
      where `indices_off_diagonal` is a list of arrays and each array contains the indices of examples counted in off-diagonals of confident joint.

    Note
    ----
    We provide a for-loop based simplification of the confident joint
    below. This implementation is not efficient, not used in practice, and
    not complete, but covers the gist of how the confident joint is computed:

    .. code:: python

        # Confident examples are those that we are confident have true_label = k
        # Estimate (K, K) matrix of confident examples with label = k_s and true_label = k_y
        cj_ish = np.zeros((K, K))
        for k_s in range(K): # k_s is the class value k of noisy labels `s`
            for k_y in range(K): # k_y is the (guessed) class k of true_label k_y
                cj_ish[k_s][k_y] = sum((pred_probs[:,k_y] >= (thresholds[k_y] - 1e-8)) & (labels == k_s))

    The following is a vectorized (but non-parallelized) implementation of the
    confident joint, again slow, using for-loops/simplified for understanding.
    This implementation is 100% accurate, it's just not optimized for speed.

    .. code:: python

        confident_joint = np.zeros((K, K), dtype = int)
        for i, row in enumerate(pred_probs):
            s_label = labels[i]
            confident_bins = row >= thresholds - 1e-6
            num_confident_bins = sum(confident_bins)
            if num_confident_bins == 1:
                confident_joint[s_label][np.argmax(confident_bins)] += 1
            elif num_confident_bins > 1:
                confident_joint[s_label][np.argmax(row)] += 1
    """

    if multi_label:
        if not isinstance(labels, list):
            raise TypeError("`labels` must be list when `multi_label=True`.")

        return _compute_confident_joint_multi_label(
            labels=labels,
            pred_probs=pred_probs,
            thresholds=thresholds,
            calibrate=calibrate,
            return_indices_of_off_diagonals=return_indices_of_off_diagonals,
        )

    # labels needs to be a numpy array
    labels = np.asarray(labels)

    # Estimate the probability thresholds for confident counting
    if thresholds is None:
        # P(we predict the given noisy label is k | given noisy label is k)
        thresholds = get_confident_thresholds(labels, pred_probs, multi_label=multi_label)
    thresholds = np.asarray(thresholds)

    # Compute confident joint (vectorized for speed).

    # pred_probs_bool is a bool matrix where each row represents a training example as a boolean vector of
    # size num_classes, with True if the example confidently belongs to that class and False if not.
    pred_probs_bool = pred_probs >= thresholds - 1e-6
    num_confident_bins = pred_probs_bool.sum(axis=1)
    at_least_one_confident = num_confident_bins > 0
    more_than_one_confident = num_confident_bins > 1
    pred_probs_argmax = pred_probs.argmax(axis=1)
    # Note that confident_argmax is meaningless for rows of all False
    confident_argmax = pred_probs_bool.argmax(axis=1)
    # For each example, choose the confident class (greater than threshold)
    # When there is 2+ confident classes, choose the class with largest prob.
    true_label_guess = np.where(
        more_than_one_confident,
        pred_probs_argmax,
        confident_argmax,
    )
    # true_labels_confident omits meaningless all-False rows
    true_labels_confident = true_label_guess[at_least_one_confident]
    labels_confident = labels[at_least_one_confident]
    confident_joint = confusion_matrix(
        y_true=true_labels_confident,
        y_pred=labels_confident,
        labels=range(pred_probs.shape[1]),
    ).T  # Guarantee at least one correctly labeled example is represented in every class
    np.fill_diagonal(confident_joint, confident_joint.diagonal().clip(min=1))
    if calibrate:
        confident_joint = calibrate_confident_joint(confident_joint, labels)

    if return_indices_of_off_diagonals:
        true_labels_neq_given_labels = true_labels_confident != labels_confident
        indices = np.arange(len(labels))[at_least_one_confident][true_labels_neq_given_labels]

        return confident_joint, indices

    return confident_joint


def _compute_confident_joint_multi_label(
    labels: list,
    pred_probs: np.ndarray,
    *,
    thresholds: Optional[Union[np.ndarray, list]] = None,
    calibrate: bool = True,
    return_indices_of_off_diagonals: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    """Computes the confident joint for multi_labeled data. Thus,
    input `labels` is a list of lists (or list of iterable).
    This is intended as a helper function. You should probably
    be using `compute_confident_joint(multi_label=True)` instead.

    The MAJOR DIFFERENCE in how this is computed versus single_label,
    is the total number of errors considered is based on the number
    of labels, not the number of examples. So, the confident_joint
    will have larger values.

    See `compute_confident_joint` docstring for more info.

    Parameters
    ----------
    labels : list of list/iterable (length N)
        Given noisy labels for multi-label classification.
        Must be a list of lists (or a list of np.ndarrays or iterable).
        The i-th element is a list containing the classes that the i-th example belongs to.

    pred_probs : np.ndarray (shape (N, K))
        P(label=k|x) is a matrix with K model-predicted probabilities.
        Each row of this matrix corresponds to an example `x` and contains the model-predicted
        probabilities that `x` belongs to each possible class.
        The columns must be ordered such that these probabilities correspond to class 0, 1, 2,..., K-1.
        `pred_probs` must be out-of-sample (ideally should have been computed using 3+ fold cross-validation).

    thresholds : iterable (list or np.ndarray) of shape (K, 1)  or (K,)
        P(label^=k|label=k). If an example has a predicted probability "greater" than
        this threshold, it is counted as having true_label = k. This is
        not used for filtering/pruning, only for estimating the noise rates using
        confident counts. This value should be between 0 and 1. Default is None.

    calibrate : bool, default = True
        Calibrates confident joint estimate P(label=i, true_label=j) such that
        ``np.sum(cj) == len(labels) and np.sum(cj, axis = 1) == np.bincount(labels)``.

    return_indices_of_off_diagonals: bool, default = False
        If true returns indices of examples that were counted in off-diagonals
        of confident joint as a baseline proxy for the label issues. This
        sometimes works as well as filter.find_label_issues(confident_joint).

    Returns
    -------
    confident_joint_counts : np.ndarray
      An array of shape ``(K, 2, 2)`` representing the confident joint of noisy and true labels for each class, in a one-vs-rest format employed for multi-label settings.

    Note: if `return_indices_of_off_diagonals` is set as True, this function instead returns a tuple `(confident_joint_counts, indices_off_diagonal)`
    where `indices_off_diagonal` is a list of arrays (one per class) and each array contains the indices of examples counted in off-diagonals of confident joint for that class.
    """

    y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
    confident_joint_list: np.ndarray = np.ndarray(shape=(num_classes, 2, 2), dtype=np.int64)
    indices_off_diagonal = []
    for class_num, (label, pred_prob_for_class) in enumerate(zip(y_one.T, pred_probs.T)):
        pred_probs_binary = stack_complement(pred_prob_for_class)
        if return_indices_of_off_diagonals:
            cj, ind = compute_confident_joint(
                labels=label,
                pred_probs=pred_probs_binary,
                multi_label=False,
                thresholds=thresholds,
                calibrate=calibrate,
                return_indices_of_off_diagonals=return_indices_of_off_diagonals,
            )
            indices_off_diagonal.append(ind)
        else:
            cj = compute_confident_joint(
                labels=label,
                pred_probs=pred_probs_binary,
                multi_label=False,
                thresholds=thresholds,
                calibrate=calibrate,
                return_indices_of_off_diagonals=return_indices_of_off_diagonals,
            )
        confident_joint_list[class_num] = cj

    if return_indices_of_off_diagonals:
        return confident_joint_list, indices_off_diagonal

    return confident_joint_list


def estimate_latent(
    confident_joint: np.ndarray,
    labels: np.ndarray,
    *,
    py_method: str = "cnt",
    converge_latent_estimates: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the latent prior ``p(y)``, the noise matrix ``P(labels|y)`` and the
    inverse noise matrix ``P(y|labels)`` from the `confident_joint` ``count(labels, y)``. The
    `confident_joint` can be estimated by `~cleanlab.count.compute_confident_joint`
    which counts confident examples.

    Parameters
    ----------
    confident_joint : np.ndarray
      An array of shape ``(K, K)`` representing the confident joint, the matrix used for identifying label issues, which
      estimates a confident subset of the joint distribution of the noisy and true labels, ``P_{noisy label, true label}``.
      Entry ``(j, k)`` in the matrix is the number of examples confidently counted into the pair of ``(noisy label=j, true label=k)`` classes.
      The `confident_joint` can be computed using `~cleanlab.count.compute_confident_joint`.
      If not provided, it is computed from the given (noisy) `labels` and `pred_probs`.

    labels : np.ndarray
      A 1D array of shape ``(N,)`` containing class labels for a standard (multi-class) classification dataset. Some given labels may be erroneous.
      Elements must be integers in the set 0, 1, ..., K-1, where K is the number of classes.

    py_method : {"cnt", "eqn", "marginal", "marginal_ps"}, default="cnt"
      `py` is shorthand for the "class proportions (a.k.a prior) of the true labels".
      This method defines how to compute the latent prior ``p(true_label=k)``. Default is ``"cnt"``,
      which works well even when the noise matrices are estimated poorly by using
      the matrix diagonals instead of all the probabilities.

    converge_latent_estimates : bool, optional
      If ``True``, forces numerical consistency of estimates. Each is estimated
      independently, but they are related mathematically with closed form
      equivalences. This will iteratively make them mathematically consistent.

    Returns
    ------
    tuple
      A tuple containing (py, noise_matrix, inv_noise_matrix).

    Note
    ----
    Multi-label classification is not supported in this method.
    """

    num_classes = len(confident_joint)
    label_counts = value_counts_fill_missing_classes(labels, num_classes)
    # 'ps' is p(labels=k)
    ps = label_counts / float(len(labels))
    # Number of training examples confidently counted from each noisy class
    labels_class_counts = confident_joint.sum(axis=1).astype(float)
    # Number of training examples confidently counted into each true class
    true_labels_class_counts = confident_joint.sum(axis=0).astype(float)
    # p(label=k_s|true_label=k_y) ~ |label=k_s and true_label=k_y| / |true_label=k_y|
    noise_matrix = confident_joint / np.clip(true_labels_class_counts, a_min=TINY_VALUE, a_max=None)
    # p(true_label=k_y|label=k_s) ~ |true_label=k_y and label=k_s| / |label=k_s|
    inv_noise_matrix = confident_joint.T / np.clip(
        labels_class_counts, a_min=TINY_VALUE, a_max=None
    )
    # Compute the prior p(y), the latent (uncorrupted) class distribution.
    py = compute_py(
        ps,
        noise_matrix,
        inv_noise_matrix,
        py_method=py_method,
        true_labels_class_counts=true_labels_class_counts,
    )
    # Clip noise rates to be valid probabilities.
    noise_matrix = clip_noise_rates(noise_matrix)
    inv_noise_matrix = clip_noise_rates(inv_noise_matrix)
    # Make latent estimates mathematically agree in their algebraic relations.
    if converge_latent_estimates:
        py, noise_matrix, inv_noise_matrix = _converge_estimates(
            ps, py, noise_matrix, inv_noise_matrix
        )
        # Again clip py and noise rates into proper range [0,1)
        py = clip_values(py, low=1e-5, high=1.0, new_sum=1.0)
        noise_matrix = clip_noise_rates(noise_matrix)
        inv_noise_matrix = clip_noise_rates(inv_noise_matrix)

    return py, noise_matrix, inv_noise_matrix


def estimate_py_and_noise_matrices_from_probabilities(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    thresholds: Optional[Union[np.ndarray, list]] = None,
    converge_latent_estimates: bool = True,
    py_method: str = "cnt",
    calibrate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the confident counts
    estimate of latent variables `py` and the noise rates
    using observed labels and predicted probabilities, `pred_probs`.

    Important: this function assumes that `pred_probs` are out-of-sample
    holdout probabilities. This can be :ref:`done with cross validation <pred_probs_cross_val>`. If
    the probabilities are not computed out-of-sample, overfitting may occur.

    This function estimates the `noise_matrix` of shape ``(K, K)``. This is the
    fraction of examples in every class, labeled as every other class. The
    `noise_matrix` is a conditional probability matrix for ``P(label=k_s|true_label=k_y)``.

    Under certain conditions, estimates are exact, and in most
    conditions, estimates are within one percent of the actual noise rates.

    Parameters
    ----------
    labels : np.ndarray
      A 1D array of shape ``(N,)`` containing class labels for a standard (multi-class) classification dataset. Some given labels may be erroneous.
      Elements must be integers in the set 0, 1, ..., K-1, where K is the number of classes.

    pred_probs : np.ndarray
      Model-predicted class probabilities for each example in the dataset,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    thresholds : array_like, optional
      An array of shape ``(K, 1)`` or ``(K,)`` of per-class threshold
      probabilities, used to determine the cutoff probability necessary to
      consider an example as a given class label (see `Northcutt et al.,
      2021 <https://jair.org/index.php/jair/article/view/12125>`_, Section
      3.1, Equation 2).

      This is for advanced users only. If not specified, these are computed
      for you automatically. If an example has a predicted probability
      greater than this threshold, it is counted as having true_label =
      k. This is not used for pruning/filtering, only for estimating the
      noise rates using confident counts.

    converge_latent_estimates : bool, optional
      If ``True``, forces numerical consistency of estimates. Each is estimated
      independently, but they are related mathematically with closed form
      equivalences. This will iteratively make them mathematically consistent.

    py_method : {"cnt", "eqn", "marginal", "marginal_ps"}, default="cnt"
      How to compute the latent prior ``p(true_label=k)``. Default is ``"cnt"`` as it often
      works well even when the noise matrices are estimated poorly by using
      the matrix diagonals instead of all the probabilities.

    calibrate : bool, default=True
      Calibrates confident joint estimate ``P(label=i, true_label=j)`` such that
      ``np.sum(cj) == len(labels)`` and ``np.sum(cj, axis = 1) == np.bincount(labels)``.

    Returns
    ------
    estimates : tuple
        A tuple of arrays: (`py`, `noise_matrix`, `inverse_noise_matrix`, `confident_joint`).

    Note
    ----
    Multi-label classification is not supported in this method.
    """

    confident_joint = compute_confident_joint(
        labels=labels,
        pred_probs=pred_probs,
        thresholds=thresholds,
        calibrate=calibrate,
    )
    assert isinstance(confident_joint, np.ndarray)
    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        labels=labels,
        py_method=py_method,
        converge_latent_estimates=converge_latent_estimates,
    )
    assert isinstance(confident_joint, np.ndarray)

    return py, noise_matrix, inv_noise_matrix, confident_joint


def estimate_confident_joint_and_cv_pred_proba(
    X,
    labels,
    clf=LogReg(multi_class="auto", solver="lbfgs"),
    *,
    cv_n_folds=5,
    thresholds=None,
    seed=None,
    calibrate=True,
    clf_kwargs={},
    validation_func=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimates ``P(labels, y)``, the confident counts of the latent
    joint distribution of true and noisy labels
    using observed `labels` and predicted probabilities `pred_probs`.

    The output of this function is an array of shape ``(K, K)``.

    Under certain conditions, estimates are exact, and in many
    conditions, estimates are within one percent of actual.

    Notes: There are two ways to compute the confident joint with pros/cons.
    (1) For each holdout set, we compute the confident joint, then sum them up.
    (2) Compute pred_proba for each fold, combine, compute the confident joint.
    (1) is more accurate because it correctly computes thresholds for each fold
    (2) is more accurate when you have only a little data because it computes
    the confident joint using all the probabilities. For example if you had 100
    examples, with 5-fold cross validation + uniform p(y) you would only have 20
    examples to compute each confident joint for (1). Such small amounts of data
    is bound to result in estimation errors. For this reason, we implement (2),
    but we implement (1) as a commented out function at the end of this file.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
      Input feature matrix of shape ``(N, ...)``, where N is the number of
      examples. The classifier that this instance was initialized with,
          ``clf``, must be able to fit() and predict() data with this format.

    labels : np.ndarray or pd.Series
      A 1D array of shape ``(N,)`` containing class labels for a standard (multi-class) classification dataset.
      Some given labels may be erroneous.
      Elements must be integers in the set 0, 1, ..., K-1, where K is the number of classes.
      All classes must be present in the dataset.

    clf : estimator instance, optional
      A classifier implementing the `sklearn estimator API
      <https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator>`_.

    cv_n_folds : int, default=5
      The number of cross-validation folds used to compute
      out-of-sample predicted probabilities for each example in `X`.

    thresholds : array_like, optional
      An array of shape ``(K, 1)`` or ``(K,)`` of per-class threshold
      probabilities, used to determine the cutoff probability necessary to
      consider an example as a given class label (see `Northcutt et al.,
      2021 <https://jair.org/index.php/jair/article/view/12125>`_, Section
      3.1, Equation 2).

      This is for advanced users only. If not specified, these are computed
      for you automatically. If an example has a predicted probability
      greater than this threshold, it is counted as having true_label =
      k. This is not used for pruning/filtering, only for estimating the
      noise rates using confident counts.

    seed : int, optional
        Set the default state of the random number generator used to split
        the cross-validated folds. If None, uses np.random current random state.

    calibrate : bool, default=True
        Calibrates confident joint estimate ``P(label=i, true_label=j)`` such that
        ``np.sum(cj) == len(labels)`` and ``np.sum(cj, axis = 1) == np.bincount(labels)``.

    clf_kwargs : dict, optional
      Optional keyword arguments to pass into `clf`'s ``fit()`` method.

    validation_func : callable, optional
      Specifies how to map the validation data split in cross-validation as input for ``clf.fit()``.
      For details, see the documentation of :py:meth:`CleanLearning.fit<cleanlab.classification.CleanLearning.fit>`

    Returns
    ------
    estimates : tuple
      Tuple of two numpy arrays in the form:
      (joint counts matrix, predicted probability matrix)

    Note
    ----
    Multi-label classification is not supported in this method.
    """

    assert_valid_inputs(X, labels)
    labels = labels_to_array(labels)
    num_classes = get_num_classes(
        labels=labels
    )  # This method definitely only works if all classes are present.

    # Create cross-validation object for out-of-sample predicted probabilities.
    # CV folds preserve the fraction of noisy positive and
    # noisy negative examples in each class.
    kf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=seed)

    # Initialize pred_probs array
    pred_probs = np.zeros(shape=(len(labels), num_classes))

    # Split X and labels into "cv_n_folds" stratified folds.
    # CV indices only require labels: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    # Only split based on labels because X may have various formats:
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(X=labels, y=labels)):
        try:
            clf_copy = sklearn.base.clone(clf)  # fresh untrained copy of the model
        except Exception:
            raise ValueError(
                "`clf` must be clonable via: sklearn.base.clone(clf). "
                "You can either implement instance method `clf.get_params()` to produce a fresh untrained copy of this model, "
                "or you can implement the cross-validation outside of cleanlab "
                "and pass in the obtained `pred_probs` to skip cleanlab's internal cross-validation"
            )
        # Select the training and holdout cross-validated sets.
        X_train_cv, X_holdout_cv, s_train_cv, s_holdout_cv = train_val_split(
            X, labels, cv_train_idx, cv_holdout_idx
        )

        # dict with keys: which classes missing, values: index of holdout data from this class that is duplicated:
        missing_class_inds = {}
        is_tf_or_torch_dataset = is_torch_dataset(X) or is_tensorflow_dataset(X)
        if not is_tf_or_torch_dataset:
            # Ensure no missing classes in training set.
            train_cv_classes = set(s_train_cv)
            all_classes = set(range(num_classes))
            if len(train_cv_classes) != len(all_classes):
                missing_classes = all_classes.difference(train_cv_classes)
                warnings.warn(
                    "Duplicated some data across multiple folds to ensure training does not fail "
                    f"because these classes do not have enough data for proper cross-validation: {missing_classes}."
                )
                for missing_class in missing_classes:
                    # Duplicate one instance of missing_class from holdout data to the training data:
                    holdout_inds = np.where(s_holdout_cv == missing_class)[0]
                    dup_idx = holdout_inds[0]
                    s_train_cv = np.append(s_train_cv, s_holdout_cv[dup_idx])
                    # labels are always np.ndarray so don't have to consider .iloc above
                    X_train_cv = append_extra_datapoint(
                        to_data=X_train_cv, from_data=X_holdout_cv, index=dup_idx
                    )
                    missing_class_inds[missing_class] = dup_idx

        # Map validation data into appropriate format to pass into classifier clf
        if validation_func is None:
            validation_kwargs = {}
        elif callable(validation_func):
            validation_kwargs = validation_func(X_holdout_cv, s_holdout_cv)
        else:
            raise TypeError("validation_func must be callable function with args: X_val, y_val")

        # Fit classifier clf to training set, predict on holdout set, and update pred_probs.
        clf_copy.fit(X_train_cv, s_train_cv, **clf_kwargs, **validation_kwargs)
        pred_probs_cv = clf_copy.predict_proba(X_holdout_cv)  # P(labels = k|x) # [:,1]

        # Replace predictions for duplicated indices with dummy predictions:
        for missing_class in missing_class_inds:
            dummy_pred = np.zeros(pred_probs_cv[0].shape)
            dummy_pred[missing_class] = 1.0  # predict given label with full confidence
            dup_idx = missing_class_inds[missing_class]
            pred_probs_cv[dup_idx] = dummy_pred

        pred_probs[cv_holdout_idx] = pred_probs_cv

    # Compute the confident counts, a num_classes x num_classes matrix for all pairs of labels.
    confident_joint = compute_confident_joint(
        labels=labels,
        pred_probs=pred_probs,  # P(labels = k|x)
        thresholds=thresholds,
        calibrate=calibrate,
    )
    assert isinstance(confident_joint, np.ndarray)
    assert isinstance(pred_probs, np.ndarray)

    return confident_joint, pred_probs


def estimate_py_noise_matrices_and_cv_pred_proba(
    X,
    labels,
    clf=LogReg(multi_class="auto", solver="lbfgs"),
    *,
    cv_n_folds=5,
    thresholds=None,
    converge_latent_estimates=False,
    py_method="cnt",
    seed=None,
    clf_kwargs={},
    validation_func=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """This function computes the out-of-sample predicted
    probability ``P(label=k|x)`` for every example x in `X` using cross
    validation while also computing the confident counts noise
    rates within each cross-validated subset and returning
    the average noise rate across all examples.

    This function estimates the `noise_matrix` of shape ``(K, K)``. This is the
    fraction of examples in every class, labeled as every other class. The
    `noise_matrix` is a conditional probability matrix for ``P(label=k_s|true_label=k_y)``.

    Under certain conditions, estimates are exact, and in most
    conditions, estimates are within one percent of the actual noise rates.

    Parameters
    ----------
    X : np.ndarray
      Input feature matrix of shape ``(N, ...)``, where N is the number of
      examples. The classifier that this instance was initialized with,
      `clf`, must be able to handle data with this shape.

    labels : np.ndarray
      A 1D array of shape ``(N,)`` containing class labels for a standard (multi-class) classification dataset.
      Some given labels may be erroneous.
      Elements must be integers in the set 0, 1, ..., K-1, where K is the number of classes.
      All classes must be present in the dataset.

    clf : estimator instance, optional
      A classifier implementing the `sklearn estimator API
      <https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator>`_.

    cv_n_folds : int, default=5
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in `X`.

    thresholds : array_like, optional
      An array of shape ``(K, 1)`` or ``(K,)`` of per-class threshold
      probabilities, used to determine the cutoff probability necessary to
      consider an example as a given class label (see `Northcutt et al.,
      2021 <https://jair.org/index.php/jair/article/view/12125>`_, Section
      3.1, Equation 2).

      This is for advanced users only. If not specified, these are computed
      for you automatically. If an example has a predicted probability
      greater than this threshold, it is counted as having true_label =
      k. This is not used for pruning/filtering, only for estimating the
      noise rates using confident counts.

    converge_latent_estimates : bool, optional
      If ``True``, forces numerical consistency of estimates. Each is estimated
      independently, but they are related mathematically with closed form
      equivalences. This will iteratively make them mathematically consistent.

    py_method : {"cnt", "eqn", "marginal", "marginal_ps"}, default="cnt"
      How to compute the latent prior ``p(true_label=k)``. Default is ``"cnt"`` as it often
      works well even when the noise matrices are estimated poorly by using
      the matrix diagonals instead of all the probabilities.

    seed : int, optional
      Set the default state of the random number generator used to split
      the cross-validated folds. If ``None``, uses ``np.random`` current random state.

    clf_kwargs : dict, optional
      Optional keyword arguments to pass into `clf`'s ``fit()`` method.

    validation_func : callable, optional
      Specifies how to map the validation data split in cross-validation as input for ``clf.fit()``.
      For details, see the documentation of :py:meth:`CleanLearning.fit<cleanlab.classification.CleanLearning.fit>`

    Returns
    ------
    estimates: tuple
      A tuple of five arrays (py, noise matrix, inverse noise matrix, confident joint, predicted probability matrix).

    Note
    ----
    Multi-label classification is not supported in this method.
    """
    confident_joint, pred_probs = estimate_confident_joint_and_cv_pred_proba(
        X=X,
        labels=labels,
        clf=clf,
        cv_n_folds=cv_n_folds,
        thresholds=thresholds,
        seed=seed,
        clf_kwargs=clf_kwargs,
        validation_func=validation_func,
    )

    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        labels=labels,
        py_method=py_method,
        converge_latent_estimates=converge_latent_estimates,
    )

    return py, noise_matrix, inv_noise_matrix, confident_joint, pred_probs


def estimate_cv_predicted_probabilities(
    X,
    labels,
    clf=LogReg(multi_class="auto", solver="lbfgs"),
    *,
    cv_n_folds=5,
    seed=None,
    clf_kwargs={},
    validation_func=None,
) -> np.ndarray:
    """This function computes the out-of-sample predicted
    probability [P(label=k|x)] for every example in X using cross
    validation. Output is a np.ndarray of shape ``(N, K)`` where N is
    the number of training examples and K is the number of classes.

    Parameters
    ----------
    X : np.ndarray
      Input feature matrix of shape ``(N, ...)``, where N is the number of
      examples. The classifier that this instance was initialized with,
      `clf`, must be able to handle data with this shape.

    labels : np.ndarray
      A 1D array of shape ``(N,)`` containing class labels for a standard (multi-class) classification dataset.
      Some given labels may be erroneous.
      Elements must be integers in the set 0, 1, ..., K-1, where K is the number of classes.
      All classes must be present in the dataset.

    clf : estimator instance, optional
      A classifier implementing the `sklearn estimator API
      <https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator>`_.

    cv_n_folds : int, default=5
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in `X`.

    seed : int, optional
      Set the default state of the random number generator used to split
      the cross-validated folds. If ``None``, uses ``np.random`` current random state.

    clf_kwargs : dict, optional
      Optional keyword arguments to pass into `clf`'s ``fit()`` method.

    validation_func : callable, optional
      Specifies how to map the validation data split in cross-validation as input for ``clf.fit()``.
      For details, see the documentation of :py:meth:`CleanLearning.fit<cleanlab.classification.CleanLearning.fit>`

    Returns
    --------
    pred_probs : np.ndarray
      An array of shape ``(N, K)`` representing ``P(label=k|x)``, the model-predicted probabilities.
      Each row of this matrix corresponds to an example `x` and contains the model-predicted
      probabilities that `x` belongs to each possible class.
    """

    return estimate_py_noise_matrices_and_cv_pred_proba(
        X=X,
        labels=labels,
        clf=clf,
        cv_n_folds=cv_n_folds,
        seed=seed,
        clf_kwargs=clf_kwargs,
        validation_func=validation_func,
    )[-1]


def estimate_noise_matrices(
    X,
    labels,
    clf=LogReg(multi_class="auto", solver="lbfgs"),
    *,
    cv_n_folds=5,
    thresholds=None,
    converge_latent_estimates=True,
    seed=None,
    clf_kwargs={},
    validation_func=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimates the `noise_matrix` of shape ``(K, K)``. This is the
    fraction of examples in every class, labeled as every other class. The
    `noise_matrix` is a conditional probability matrix for ``P(label=k_s|true_label=k_y)``.

    Under certain conditions, estimates are exact, and in most
    conditions, estimates are within one percent of the actual noise rates.

    Parameters
    ----------
    X : np.ndarray
      Input feature matrix of shape ``(N, ...)``, where N is the number of
      examples. The classifier that this instance was initialized with,
      `clf`, must be able to handle data with this shape.

    labels : np.ndarray
      An array of shape ``(N,)`` of noisy labels, i.e. some labels may be erroneous.
      Elements must be integers in the set 0, 1, ..., K-1, where K is the number of classes.

    clf : estimator instance, optional
      A classifier implementing the `sklearn estimator API
      <https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator>`_.

    cv_n_folds : int, default=5
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in `X`.

    thresholds : array_like, optional
      An array of shape ``(K, 1)`` or ``(K,)`` of per-class threshold
      probabilities, used to determine the cutoff probability necessary to
      consider an example as a given class label (see `Northcutt et al.,
      2021 <https://jair.org/index.php/jair/article/view/12125>`_, Section
      3.1, Equation 2).

      This is for advanced users only. If not specified, these are computed
      for you automatically. If an example has a predicted probability
      greater than this threshold, it is counted as having true_label =
      k. This is not used for pruning/filtering, only for estimating the
      noise rates using confident counts.

    converge_latent_estimates : bool, optional
      If ``True``, forces numerical consistency of estimates. Each is estimated
      independently, but they are related mathematically with closed form
      equivalences. This will iteratively make them mathematically consistent.

    seed : int, optional
        Set the default state of the random number generator used to split
        the cross-validated folds. If None, uses np.random current random state.

    clf_kwargs : dict, optional
      Optional keyword arguments to pass into `clf`'s ``fit()`` method.

    validation_func : callable, optional
      Specifies how to map the validation data split in cross-validation as input for ``clf.fit()``.
      For details, see the documentation of :py:meth:`CleanLearning.fit<cleanlab.classification.CleanLearning.fit>`

    Returns
    ------
    estimates : tuple
      A tuple containing arrays (`noise_matrix`, `inv_noise_matrix`)."""

    return estimate_py_noise_matrices_and_cv_pred_proba(
        X=X,
        labels=labels,
        clf=clf,
        cv_n_folds=cv_n_folds,
        thresholds=thresholds,
        converge_latent_estimates=converge_latent_estimates,
        seed=seed,
        clf_kwargs=clf_kwargs,
        validation_func=validation_func,
    )[1:-2]


def _converge_estimates(
    ps: np.ndarray,
    py: np.ndarray,
    noise_matrix: np.ndarray,
    inverse_noise_matrix: np.ndarray,
    *,
    inv_noise_matrix_iterations: int = 5,
    noise_matrix_iterations: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Updates py := P(true_label=k) and both `noise_matrix` and `inverse_noise_matrix`
    to be numerically consistent with each other, by iteratively updating their estimates based on
    the mathematical relationships between them.

    Forces numerical consistency of estimates. Each is estimated
    independently, but they are related mathematically with closed form
    equivalences. This will iteratively make them mathematically consistent.

    py := P(true_label=k) and the inverse noise matrix P(true_label=k_y|label=k_s) specify one
    another, meaning one can be computed from the other and vice versa.
    When numerical discrepancy exists due to poor estimation, they can be made
    to agree by repeatedly computing one from the other,
    for some a certain number of iterations (3-10 works fine.)

    Do not set iterations too high or performance will decrease as small
    deviations will get perturbed over and over and potentially magnified.

    Note that we have to first converge the inverse_noise_matrix and py,
    then we can update the noise_matrix, then repeat. This is because the
    inverse noise matrix depends on py (which is unknown/latent), but the
    noise matrix depends on ps (which is known), so there will be no change in
    the noise matrix if we recompute it when py and inverse_noise_matrix change.


    Parameters
    ----------
    ps : np.ndarray (shape (K, ) or (1, K))
        The fraction (prior probability) of each observed, NOISY class P(labels = k).

    py : np.ndarray (shape (K, ) or (1, K))
        The estimated fraction (prior probability) of each TRUE class P(true_label = k).

    noise_matrix : np.ndarray of shape (K, K), K = number of classes
        A conditional probability matrix of the form P(label=k_s|true_label=k_y) containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1.

    inverse_noise_matrix : np.ndarray of shape (K, K), K = number of classes
        A conditional probability matrix of the form P(true_label=k_y|labels=k_s) representing
        the estimated fraction observed examples in each class k_s, that are
        mislabeled examples from every other class k_y. If None, the
        inverse_noise_matrix will be computed from pred_probs and labels.
        Assumes columns of inverse_noise_matrix sum to 1.

    inv_noise_matrix_iterations : int, default = 5
        Number of times to converge inverse noise matrix with py and noise mat.

    noise_matrix_iterations : int, default = 3
        Number of times to converge noise matrix with py and inverse noise mat.

    Returns
    ------
    estimates: tuple
        Three arrays of the form (`py`, `noise_matrix`, `inverse_noise_matrix`) all
        having numerical agreement in terms of their mathematical relations."""

    for j in range(noise_matrix_iterations):
        for i in range(inv_noise_matrix_iterations):
            inverse_noise_matrix = compute_inv_noise_matrix(py=py, noise_matrix=noise_matrix, ps=ps)
            py = compute_py(ps, noise_matrix, inverse_noise_matrix)
        noise_matrix = compute_noise_matrix_from_inverse(
            ps=ps, inverse_noise_matrix=inverse_noise_matrix, py=py
        )

    return py, noise_matrix, inverse_noise_matrix


def get_confident_thresholds(
    labels: LabelLike,
    pred_probs: np.ndarray,
    multi_label: bool = False,
) -> np.ndarray:
    """Returns expected (average) "self-confidence" for each class.

    The confident class threshold for a class j is the expected (average) "self-confidence" for class j,
    i.e. the model-predicted probability of this class averaged amongst all examples labeled as class j.

    Parameters
    ----------
    labels : np.ndarray or list
      Given class labels for each example in the dataset, some of which may be erroneous,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    pred_probs : np.ndarray
      Model-predicted class probabilities for each example in the dataset,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    multi_label : bool, default = False
      Set ``False`` if your dataset is for regular (multi-class) classification, where each example belongs to exactly one class.
      Set ``True`` if your dataset is for multi-label classification, where each example can belong to multiple classes.
      See documentation of `~cleanlab.count.compute_confident_joint` for details.

    Returns
    -------
    confident_thresholds : np.ndarray
      An array of shape ``(K, )`` where K is the number of classes.
    """
    if multi_label:
        assert isinstance(labels, list)
        return _get_confident_thresholds_multilabel(labels=labels, pred_probs=pred_probs)
    else:
        # When all_classes != unique_classes the class threshold for the missing classes is set to
        # BIG_VALUE such that no valid prob >= BIG_VALUE (no example will be counted in missing classes)
        # REQUIRES: pred_probs.max() >= 1
        # TODO: if you want this to work for arbitrary softmax outputs where pred_probs.max()
        #  may exceed 1, change BIG_VALUE = 2 --> BIG_VALUE = 2 * pred_probs.max(). Downside of
        #  this approach is that there will be no standard value returned for missing classes.
        labels = labels_to_array(labels)
        all_classes = range(pred_probs.shape[1])
        unique_classes = get_unique_classes(labels)
        BIG_VALUE = 2
        confident_thresholds = [
            np.mean(pred_probs[:, k][labels == k]) if k in unique_classes else BIG_VALUE
            for k in all_classes
        ]
        confident_thresholds = np.clip(
            confident_thresholds, a_min=CONFIDENT_THRESHOLDS_LOWER_BOUND, a_max=None
        )
        return confident_thresholds


def _get_confident_thresholds_multilabel(
    labels: list,
    pred_probs: np.ndarray,
):
    """Returns expected (average) "self-confidence" for each class.

    The confident class threshold for a class j is the expected (average) "self-confidence" for class j in a one-vs-rest setting.

    Parameters
    ----------
    labels: list
       Refer to documentation for this argument in ``count.calibrate_confident_joint()`` with ``multi_label=True`` for details.

    pred_probs : np.ndarray
       Predicted class probabilities in the same format expected by the `~cleanlab.count.get_confident_thresholds` function.

    Returns
    -------
    confident_thresholds : np.ndarray
      An array of shape ``(K, 2, 2)`` where `K` is the number of classes, in a one-vs-rest format.
    """
    y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
    confident_thresholds: np.ndarray = np.ndarray((num_classes, 2))
    for class_num, (label_for_class, pred_prob_for_class) in enumerate(zip(y_one.T, pred_probs.T)):
        pred_probs_binary = stack_complement(pred_prob_for_class)
        confident_thresholds[class_num] = get_confident_thresholds(
            pred_probs=pred_probs_binary, labels=label_for_class
        )
    return confident_thresholds
