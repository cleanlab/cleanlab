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
Methods to identify which examples have label issues in a classification dataset.
The documentation below assumes a dataset with ``N`` examples and ``K`` classes.
This module is for standard (multi-class) classification where each example is labeled as belonging to exactly one of K classes (e.g. ``labels = np.array([0,0,1,0,2,1])``).
Some methods here also work for multi-label classification data where each example can be labeled as belonging to multiple classes (e.g. ``labels = [[1,2],[1],[0],[],...]``),
but we encourage using the methods in the ``cleanlab.multilabel_classification`` module instead for such data.
"""

import numpy as np
from sklearn.metrics import confusion_matrix
import multiprocessing
import sys
import warnings
from typing import Any, Dict, Optional, Tuple, List
from functools import reduce
import platform

from cleanlab.count import calibrate_confident_joint, num_label_issues, _reduce_issues
from cleanlab.rank import order_label_issues, get_label_quality_scores
import cleanlab.internal.multilabel_scorer as ml_scorer
from cleanlab.internal.validation import assert_valid_inputs
from cleanlab.internal.util import (
    value_counts_fill_missing_classes,
    round_preserving_row_totals,
    get_num_classes,
)
from cleanlab.internal.multilabel_utils import stack_complement, get_onehot_num_classes, int2onehot
from cleanlab.typing import LabelLike
from cleanlab.multilabel_classification.filter import find_multilabel_issues_per_class

# tqdm is a package to print time-to-complete when multiprocessing is used.
# This package is not necessary, but when installed improves user experience for large datasets.
try:
    import tqdm.auto as tqdm

    tqdm_exists = True
except ImportError as e:  # pragma: no cover
    tqdm_exists = False

    w = """To see estimated completion times for methods in cleanlab.filter, "pip install tqdm"."""
    warnings.warn(w)

# psutil is a package used to count physical cores for multiprocessing
# This package is not necessary, because we can always fall back to logical cores as the default
try:
    import psutil

    psutil_exists = True
except ImportError as e:  # pragma: no cover
    psutil_exists = False

# global variable for find_label_issues multiprocessing
pred_probs_by_class: Dict[int, np.ndarray]
prune_count_matrix_cols: Dict[int, np.ndarray]


def find_label_issues(
    labels: LabelLike,
    pred_probs: np.ndarray,
    *,
    return_indices_ranked_by: Optional[str] = None,
    rank_by_kwargs: Optional[Dict[str, Any]] = None,
    filter_by: str = "prune_by_noise_rate",
    frac_noise: float = 1.0,
    num_to_remove_per_class: Optional[List[int]] = None,
    min_examples_per_class=1,
    confident_joint: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    multi_label: bool = False,
) -> np.ndarray:
    """
    Identifies potentially bad labels in a classification dataset using confident learning.

    Returns a boolean mask for the entire dataset where ``True`` represents
    an example identified with a label issue and ``False`` represents an example that seems correctly labeled.

    Instead of a mask, you can obtain indices of the examples with label issues in your dataset
    (sorted by issue severity) by specifying the `return_indices_ranked_by` argument.
    This determines which label quality score is used to quantify severity,
    and is useful to view only the top-`J` most severe issues in your dataset.

    The number of indices returned as issues is controlled by `frac_noise`: reduce its
    value to identify fewer label issues. If you aren't sure, leave this set to 1.0.

    Tip: if you encounter the error "pred_probs is not defined", try setting
    ``n_jobs=1``.

    Parameters
    ----------
    labels : np.ndarray or list
      A discrete vector of noisy labels for a classification dataset, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, each label must be integer in 0, 1, ..., K-1.
      For a standard (multi-class) classification dataset where each example is labeled with one class,
      `labels` should be 1D array of shape ``(N,)``, for example: ``labels = [1,0,2,1,1,0...]``.

    pred_probs : np.ndarray, optional
      An array of shape ``(N, K)`` of model-predicted class probabilities,
      ``P(label=k|x)``. Each row of this matrix corresponds
      to an example `x` and contains the model-predicted probabilities that
      `x` belongs to each possible class, for each of the K classes. The
      columns must be ordered such that these probabilities correspond to
      class 0, 1, ..., K-1.

      **Note**: Returned label issues are most accurate when they are computed based on out-of-sample `pred_probs` from your model.
      To obtain out-of-sample predicted probabilities for every datapoint in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`.
      This is encouraged to get better results.

    return_indices_ranked_by : {None, 'self_confidence', 'normalized_margin', 'confidence_weighted_entropy'}, default=None
      Determines what is returned by this method: either a boolean mask or list of indices np.ndarray.
      If ``None``, this function returns a boolean mask (``True`` if example at index is label error).
      If not ``None``, this function returns a sorted array of indices of examples with label issues
      (instead of a boolean mask). Indices are sorted by label quality score which can be one of:

      - ``'normalized_margin'``: ``normalized margin (p(label = k) - max(p(label != k)))``
      - ``'self_confidence'``: ``[pred_probs[i][labels[i]] for i in label_issues_idx]``
      - ``'confidence_weighted_entropy'``: ``entropy(pred_probs) / self_confidence``

    rank_by_kwargs : dict, optional
      Optional keyword arguments to pass into scoring functions for ranking by
      label quality score (see :py:func:`rank.get_label_quality_scores
      <cleanlab.rank.get_label_quality_scores>`).

    filter_by : {'prune_by_class', 'prune_by_noise_rate', 'both', 'confident_learning', 'predicted_neq_given', 'low_normalized_margin', 'low_self_confidence'}, default='prune_by_noise_rate'
      Method to determine which examples are flagged as having label issue, so you can filter/prune them from the dataset. Options:

      - ``'prune_by_noise_rate'``: filters examples with *high probability* of being mislabeled for every non-diagonal in the confident joint (see `prune_counts_matrix` in `filter.py`). These are the examples where (with high confidence) the given label is unlikely to match the predicted label for the example.
      - ``'prune_by_class'``: filters the examples with *smallest probability* of belonging to their given class label for every class.
      - ``'both'``: filters only those examples that would be filtered by both ``'prune_by_noise_rate'`` and ``'prune_by_class'``.
      - ``'confident_learning'``: filters the examples counted as part of the off-diagonals of the confident joint. These are the examples that are confidently predicted to be a different label than their given label.
      - ``'predicted_neq_given'``: filters examples for which the predicted class (i.e. argmax of the predicted probabilities) does not match the given label.
      - ``'low_normalized_margin'``: filters the examples with *smallest* normalized margin label quality score. The number of issues returned matches :py:func:`count.num_label_issues <cleanlab.count.num_label_issues>`.
      - ``'low_self_confidence'``: filters the examples with *smallest* self confidence label quality score. The number of issues returned matches :py:func:`count.num_label_issues <cleanlab.count.num_label_issues>`.

    frac_noise : float, default=1.0
      Used to only return the "top" ``frac_noise * num_label_issues``. The choice of which "top"
      label issues to return is dependent on the `filter_by` method used. It works by reducing the
      size of the off-diagonals of the `joint` distribution of given labels and true labels
      proportionally by `frac_noise` prior to estimating label issues with each method.
      This parameter only applies for `filter_by=both`, `filter_by=prune_by_class`, and
      `filter_by=prune_by_noise_rate` methods and currently is unused by other methods.
      When ``frac_noise=1.0``, return all "confident" estimated noise indices (recommended).

      frac_noise * number_of_mislabeled_examples_in_class_k.

    num_to_remove_per_class : array_like
      An iterable of length K, the number of classes.
      E.g. if K = 3, ``num_to_remove_per_class=[5, 0, 1]`` would return
      the indices of the 5 most likely mislabeled examples in class 0,
      and the most likely mislabeled example in class 2.

      Note
      ----
      Only set this parameter if ``filter_by='prune_by_class'``.
      You may use with ``filter_by='prune_by_noise_rate'``, but
      if ``num_to_remove_per_class=k``, then either k-1, k, or k+1
      examples may be removed for any class due to rounding error. If you need
      exactly 'k' examples removed from every class, you should use
      ``filter_by='prune_by_class'``.

    min_examples_per_class : int, default=1
      Minimum number of examples per class to avoid flagging as label issues.
      This is useful to avoid deleting too much data from one class
      when pruning noisy examples in datasets with rare classes.

    confident_joint : np.ndarray, optional
      An array of shape ``(K, K)`` representing the confident joint, the matrix used for identifying label issues, which
      estimates a confident subset of the joint distribution of the noisy and true labels, ``P_{noisy label, true label}``.
      Entry ``(j, k)`` in the matrix is the number of examples confidently counted into the pair of ``(noisy label=j, true label=k)`` classes.
      The `confident_joint` can be computed using :py:func:`count.compute_confident_joint <cleanlab.count.compute_confident_joint>`.
      If not provided, it is computed from the given (noisy) `labels` and `pred_probs`.

    n_jobs : optional
      Number of processing threads used by multiprocessing. Default ``None``
      sets to the number of cores on your CPU (physical cores if you have ``psutil`` package installed, otherwise logical cores).
      Set this to 1 to *disable* parallel processing (if its causing issues).
      Windows users may see a speed-up with ``n_jobs=1``.

    verbose : optional
      If ``True``, prints when multiprocessing happens.

    Returns
    -------
    label_issues : np.ndarray
      If `return_indices_ranked_by` left unspecified, returns a boolean **mask** for the entire dataset
      where ``True`` represents a label issue and ``False`` represents an example that is
      accurately labeled with high confidence.
      If `return_indices_ranked_by` is specified, returns a shorter array of **indices** of examples identified to have
      label issues (i.e. those indices where the mask would be ``True``), sorted by likelihood that the corresponding label is correct.

      Note
      ----
      Obtain the *indices* of examples with label issues in your dataset by setting `return_indices_ranked_by`.
    """
    if not rank_by_kwargs:
        rank_by_kwargs = {}

    assert filter_by in [
        "low_normalized_margin",
        "low_self_confidence",
        "prune_by_noise_rate",
        "prune_by_class",
        "both",
        "confident_learning",
        "predicted_neq_given",
    ]  # TODO: change default to confident_learning ?
    allow_one_class = False
    if isinstance(labels, np.ndarray) or all(isinstance(lab, int) for lab in labels):
        if set(labels) == {0}:  # occurs with missing classes in multi-label settings
            allow_one_class = True
    assert_valid_inputs(
        X=None,
        y=labels,
        pred_probs=pred_probs,
        multi_label=multi_label,
        allow_one_class=allow_one_class,
    )

    if filter_by in [
        "confident_learning",
        "predicted_neq_given",
        "low_normalized_margin",
        "low_self_confidence",
    ] and (frac_noise != 1.0 or num_to_remove_per_class is not None):
        warn_str = (
            "frac_noise and num_to_remove_per_class parameters are only supported"
            " for filter_by 'prune_by_noise_rate', 'prune_by_class', and 'both'. They "
            "are not supported for methods 'confident_learning', 'predicted_neq_given', "
            "'low_normalized_margin' or 'low_self_confidence'."
        )
        warnings.warn(warn_str)
    if (num_to_remove_per_class is not None) and (
        filter_by
        in [
            "confident_learning",
            "predicted_neq_given",
            "low_normalized_margin",
            "low_self_confidence",
        ]
    ):
        # TODO - add support for these filters
        raise ValueError(
            "filter_by 'confident_learning', 'predicted_neq_given', 'low_normalized_margin' "
            "or 'low_self_confidence' is not supported (yet) when setting 'num_to_remove_per_class'"
        )
    if filter_by == "confident_learning" and isinstance(confident_joint, np.ndarray):
        warn_str = (
            "The supplied `confident_joint` is ignored when `filter_by = 'confident_learning'`; confident joint will be "
            "re-estimated from the given labels. To use your supplied `confident_joint`, please specify a different "
            "`filter_by` value."
        )
        warnings.warn(warn_str)

    K = get_num_classes(
        labels=labels, pred_probs=pred_probs, label_matrix=confident_joint, multi_label=multi_label
    )
    # Boolean set to true if dataset is large
    big_dataset = K * len(labels) > 1e8

    # Set-up number of multiprocessing threads
    # On Windows/macOS, when multi_label is True, multiprocessing is much slower
    # even for faily large input arrays, so we default to n_jobs=1 in this case
    os_name = platform.system()
    if n_jobs is None:
        if multi_label and os_name != "Linux":
            n_jobs = 1
        else:
            if psutil_exists:
                n_jobs = psutil.cpu_count(logical=False)  # physical cores
            elif big_dataset:
                print(
                    "To default `n_jobs` to the number of physical cores for multiprocessing in find_label_issues(), please: `pip install psutil`.\n"
                    "Note: You can safely ignore this message. `n_jobs` only affects runtimes, results will be the same no matter its value.\n"
                    "Since psutil is not installed, `n_jobs` was set to the number of logical cores by default.\n"
                    "Disable this message by either installing psutil or specifying the `n_jobs` argument."
                )  # pragma: no cover
            if not n_jobs:
                # either psutil does not exist
                # or psutil can return None when physical cores cannot be determined
                # switch to logical cores
                n_jobs = multiprocessing.cpu_count()
    else:
        assert n_jobs >= 1

    if multi_label:
        if not isinstance(labels, list):
            raise TypeError("`labels` must be list when `multi_label=True`.")
        warnings.warn(
            "The multi_label argument to filter.find_label_issues() is deprecated and will be removed in future versions. Please use `multilabel_classification.filter.find_label_issues()` instead.",
            DeprecationWarning,
        )
        return _find_label_issues_multilabel(
            labels,
            pred_probs,
            return_indices_ranked_by,
            rank_by_kwargs,
            filter_by,
            frac_noise,
            num_to_remove_per_class,
            min_examples_per_class,
            confident_joint,
            n_jobs,
            verbose,
        )

    # Else this is standard multi-class classification
    # Number of examples in each class of labels
    label_counts = value_counts_fill_missing_classes(labels, K, multi_label=multi_label)
    # Ensure labels are of type np.ndarray()
    labels = np.asarray(labels)
    if confident_joint is None or filter_by == "confident_learning":
        from cleanlab.count import compute_confident_joint

        confident_joint, cl_error_indices = compute_confident_joint(
            labels=labels,
            pred_probs=pred_probs,
            multi_label=multi_label,
            return_indices_of_off_diagonals=True,
        )

    if filter_by in ["low_normalized_margin", "low_self_confidence"]:
        # TODO: consider setting adjust_pred_probs to true based on benchmarks (or adding it kwargs, or ignoring and leaving as false by default)
        scores = get_label_quality_scores(
            labels,
            pred_probs,
            method=filter_by[4:],
            adjust_pred_probs=False,
        )
        num_errors = num_label_issues(
            labels, pred_probs, multi_label=multi_label  # TODO: Check usage of multilabel
        )
        # Find label issues O(nlogn) solution (mapped to boolean mask later in the method)
        cl_error_indices = np.argsort(scores)[:num_errors]
        # The following is the O(n) fastest solution (check for one-off errors), but the problem is if lots of the scores are identical you will overcount,
        # you can end up returning more or less and they aren't ranked in the boolean form so there's no way to drop the highest scores randomly
        #     boundary = np.partition(scores, num_errors)[num_errors]  # O(n) solution
        #     label_issues_mask = scores <= boundary

    if filter_by in ["prune_by_noise_rate", "prune_by_class", "both"]:
        # Create `prune_count_matrix` with the number of examples to remove in each class and
        # leave at least min_examples_per_class examples per class.
        # `prune_count_matrix` is transposed relative to the confident_joint.
        prune_count_matrix = _keep_at_least_n_per_class(
            prune_count_matrix=confident_joint.T,
            n=min_examples_per_class,
            frac_noise=frac_noise,
        )

        if num_to_remove_per_class is not None:
            # Estimate joint probability distribution over label issues
            psy = prune_count_matrix / np.sum(prune_count_matrix, axis=1)
            noise_per_s = psy.sum(axis=1) - psy.diagonal()
            # Calibrate labels.t. noise rates sum to num_to_remove_per_class
            tmp = (psy.T * num_to_remove_per_class / noise_per_s).T
            np.fill_diagonal(tmp, label_counts - num_to_remove_per_class)
            prune_count_matrix = round_preserving_row_totals(tmp)

        # Prepare multiprocessing shared data
        # On Linux, multiprocessing is started with fork,
        # so data can be shared with global vairables + COW
        # On Window/macOS, processes are started with spawn,
        # so data will need to be pickled to the subprocesses through input args
        chunksize = max(1, K // n_jobs)
        if n_jobs == 1 or os_name == "Linux":
            global pred_probs_by_class, prune_count_matrix_cols
            pred_probs_by_class = {k: pred_probs[labels == k] for k in range(K)}
            prune_count_matrix_cols = {k: prune_count_matrix[:, k] for k in range(K)}
            args = [[k, min_examples_per_class, None] for k in range(K)]
        else:
            args = [
                [k, min_examples_per_class, [pred_probs[labels == k], prune_count_matrix[:, k]]]
                for k in range(K)
            ]

    # Perform Pruning with threshold probabilities from BFPRT algorithm in O(n)
    # Operations are parallelized across all CPU processes
    if filter_by == "prune_by_class" or filter_by == "both":
        if n_jobs > 1:
            with multiprocessing.Pool(n_jobs) as p:
                if verbose:  # pragma: no cover
                    print("Parallel processing label issues by class.")
                sys.stdout.flush()
                if big_dataset and tqdm_exists:
                    label_issues_masks_per_class = list(
                        tqdm.tqdm(p.imap(_prune_by_class, args, chunksize=chunksize), total=K)
                    )
                else:
                    label_issues_masks_per_class = p.map(_prune_by_class, args, chunksize=chunksize)
        else:
            label_issues_masks_per_class = [_prune_by_class(arg) for arg in args]

        label_issues_mask = np.zeros(len(labels), dtype=bool)
        for k, mask in enumerate(label_issues_masks_per_class):
            if len(mask) > 1:
                label_issues_mask[labels == k] = mask

    if filter_by == "both":
        label_issues_mask_by_class = label_issues_mask

    if filter_by == "prune_by_noise_rate" or filter_by == "both":
        if n_jobs > 1:
            with multiprocessing.Pool(n_jobs) as p:
                if verbose:  # pragma: no cover
                    print("Parallel processing label issues by noise rate.")
                sys.stdout.flush()
                if big_dataset and tqdm_exists:
                    label_issues_masks_per_class = list(
                        tqdm.tqdm(p.imap(_prune_by_count, args, chunksize=chunksize), total=K)
                    )
                else:
                    label_issues_masks_per_class = p.map(_prune_by_count, args, chunksize=chunksize)
        else:
            label_issues_masks_per_class = [_prune_by_count(arg) for arg in args]

        label_issues_mask = np.zeros(len(labels), dtype=bool)
        for k, mask in enumerate(label_issues_masks_per_class):
            if len(mask) > 1:
                label_issues_mask[labels == k] = mask

    if filter_by == "both":
        label_issues_mask = label_issues_mask & label_issues_mask_by_class

    if filter_by in ["confident_learning", "low_normalized_margin", "low_self_confidence"]:
        label_issues_mask = np.zeros(len(labels), dtype=bool)
        label_issues_mask[cl_error_indices] = True

    if filter_by == "predicted_neq_given":
        label_issues_mask = find_predicted_neq_given(labels, pred_probs, multi_label=multi_label)

    if filter_by not in ["low_self_confidence", "low_normalized_margin"]:
        # Remove label issues if model prediction is close to given label
        mask = _reduce_issues(pred_probs=pred_probs, labels=labels)
        label_issues_mask[mask] = False

    if verbose:
        print("Number of label issues found: {}".format(sum(label_issues_mask)))

    # TODO: run count.num_label_issues() and adjust the total issues found here to match
    if return_indices_ranked_by is not None:
        er = order_label_issues(
            label_issues_mask=label_issues_mask,
            labels=labels,
            pred_probs=pred_probs,
            rank_by=return_indices_ranked_by,
            rank_by_kwargs=rank_by_kwargs,
        )
        return er
    return label_issues_mask


def _find_label_issues_multilabel(
    labels: list,
    pred_probs: np.ndarray,
    return_indices_ranked_by: Optional[str] = None,
    rank_by_kwargs={},
    filter_by: str = "prune_by_noise_rate",
    frac_noise: float = 1.0,
    num_to_remove_per_class: Optional[List[int]] = None,
    min_examples_per_class=1,
    confident_joint: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    low_memory: bool = False,
) -> np.ndarray:
    """
    Finds label issues in multi-label classification data where each example can belong to more than one class.
    This is done via a one-vs-rest reduction for each class and the results are subsequently aggregated across all classes.
    Here `labels` must be formatted as an iterable of iterables, e.g. ``List[List[int]]``.
    """
    if filter_by in ["low_normalized_margin", "low_self_confidence"] and not low_memory:
        num_errors = sum(
            find_label_issues(
                labels=labels,
                pred_probs=pred_probs,
                confident_joint=confident_joint,
                multi_label=True,
                filter_by="confident_learning",
            )
        )

        y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
        label_quality_scores = ml_scorer.get_label_quality_scores(
            labels=y_one,
            pred_probs=pred_probs,
        )

        cl_error_indices = np.argsort(label_quality_scores)[:num_errors]
        label_issues_mask = np.zeros(len(labels), dtype=bool)
        label_issues_mask[cl_error_indices] = True

        if return_indices_ranked_by is not None:
            label_quality_scores_issues = ml_scorer.get_label_quality_scores(
                labels=y_one[label_issues_mask],
                pred_probs=pred_probs[label_issues_mask],
                method=ml_scorer.MultilabelScorer(
                    base_scorer=ml_scorer.ClassLabelScorer.from_str(return_indices_ranked_by),
                ),
                base_scorer_kwargs=rank_by_kwargs,
            )
            return cl_error_indices[np.argsort(label_quality_scores_issues)]

        return label_issues_mask

    per_class_issues = find_multilabel_issues_per_class(
        labels,
        pred_probs,
        return_indices_ranked_by,
        rank_by_kwargs,
        filter_by,
        frac_noise,
        num_to_remove_per_class,
        min_examples_per_class,
        confident_joint,
        n_jobs,
        verbose,
        low_memory,
    )
    if return_indices_ranked_by is None:
        assert isinstance(per_class_issues, np.ndarray)
        return per_class_issues.sum(axis=1) >= 1
    else:
        label_issues_list, labels_list, pred_probs_list = per_class_issues
        label_issues_idx = reduce(np.union1d, label_issues_list)
        y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
        label_quality_scores = ml_scorer.get_label_quality_scores(
            labels=y_one,
            pred_probs=pred_probs,
            method=ml_scorer.MultilabelScorer(
                base_scorer=ml_scorer.ClassLabelScorer.from_str(return_indices_ranked_by),
            ),
            base_scorer_kwargs=rank_by_kwargs,
        )
        label_quality_scores_issues = label_quality_scores[label_issues_idx]
        return label_issues_idx[np.argsort(label_quality_scores_issues)]


def _keep_at_least_n_per_class(
    prune_count_matrix: np.ndarray, n: int, *, frac_noise: float = 1.0
) -> np.ndarray:
    """Make sure every class has at least n examples after removing noise.
    Functionally, increase each column, increases the diagonal term #(true_label=k,label=k)
    of prune_count_matrix until it is at least n, distributing the amount
    increased by subtracting uniformly from the rest of the terms in the
    column. When frac_noise = 1.0, return all "confidently" estimated
    noise indices, otherwise this returns frac_noise fraction of all
    the noise counts, with diagonal terms adjusted to ensure column
    totals are preserved.

    Parameters
    ----------
    prune_count_matrix : np.ndarray of shape (K, K), K = number of classes
        A counts of mislabeled examples in every class. For this function.
        NOTE prune_count_matrix is transposed relative to confident_joint.

    n : int
        Number of examples to make sure are left in each class.

    frac_noise : float, default=1.0
      Used to only return the "top" ``frac_noise * num_label_issues``. The choice of which "top"
      label issues to return is dependent on the `filter_by` method used. It works by reducing the
      size of the off-diagonals of the `prune_count_matrix` of given labels and true labels
      proportionally by `frac_noise` prior to estimating label issues with each method.
      When frac_noise=1.0, return all "confident" estimated noise indices (recommended).

    Returns
    -------
    prune_count_matrix : np.ndarray of shape (K, K), K = number of classes
        This the same as the confident_joint, but has been transposed and the counts are adjusted.
    """

    prune_count_matrix_diagonal = np.diagonal(prune_count_matrix)

    # Set diagonal terms less than n, to n.
    new_diagonal = np.maximum(prune_count_matrix_diagonal, n)

    # Find how much diagonal terms were increased.
    diff_per_col = new_diagonal - prune_count_matrix_diagonal

    # Count non-zero, non-diagonal items per column
    # np.maximum(*, 1) makes this never 0 (we divide by this next)
    num_noise_rates_per_col = np.maximum(
        np.count_nonzero(prune_count_matrix, axis=0) - 1.0,
        1.0,
    )

    # Uniformly decrease non-zero noise rates by the same amount
    # that the diagonal items were increased
    new_mat = prune_count_matrix - diff_per_col / num_noise_rates_per_col

    # Originally zero noise rates will now be negative, fix them back to zero
    new_mat[new_mat < 0] = 0

    # Round diagonal terms (correctly labeled examples)
    np.fill_diagonal(new_mat, new_diagonal)

    # Reduce (multiply) all noise rates (non-diagonal) by frac_noise and
    # increase diagonal by the total amount reduced in each column
    # to preserve column counts.
    new_mat = _reduce_prune_counts(new_mat, frac_noise)

    # These are counts, so return a matrix of ints.
    return round_preserving_row_totals(new_mat).astype(int)


def _reduce_prune_counts(prune_count_matrix: np.ndarray, frac_noise: float = 1.0) -> np.ndarray:
    """Reduce (multiply) all prune counts (non-diagonal) by frac_noise and
    increase diagonal by the total amount reduced in each column to
    preserve column counts.

    Parameters
    ----------
    prune_count_matrix : np.ndarray of shape (K, K), K = number of classes
        A counts of mislabeled examples in every class. For this function, it
        does not matter what the rows or columns are, but the diagonal terms
        reflect the number of correctly labeled examples.

    frac_noise : float
      Used to only return the "top" ``frac_noise * num_label_issues``. The choice of which "top"
      label issues to return is dependent on the `filter_by` method used. It works by reducing the
      size of the off-diagonals of the `prune_count_matrix` of given labels and true labels
      proportionally by `frac_noise` prior to estimating label issues with each method.
      When frac_noise=1.0, return all "confident" estimated noise indices (recommended).
    """

    new_mat = prune_count_matrix * frac_noise
    np.fill_diagonal(new_mat, prune_count_matrix.diagonal())
    np.fill_diagonal(
        new_mat,
        prune_count_matrix.diagonal() + np.sum(prune_count_matrix - new_mat, axis=0),
    )

    # These are counts, so return a matrix of ints.
    return new_mat.astype(int)


def find_predicted_neq_given(
    labels: LabelLike, pred_probs: np.ndarray, *, multi_label: bool = False
) -> np.ndarray:
    """A simple baseline approach that considers ``argmax(pred_probs) != labels`` as the examples with label issues.

    Parameters
    ----------
    labels : np.ndarray or list
      Labels in the same format expected by the `~cleanlab.filter.find_label_issues` function.

    pred_probs : np.ndarray
      Predicted-probabilities in the same format expected by the `~cleanlab.filter.find_label_issues` function.

    multi_label : bool, optional
      Whether each example may have multiple labels or not (see documentation for the `~cleanlab.filter.find_label_issues` function).

    Returns
    -------
    label_issues_mask : np.ndarray
      A boolean mask for the entire dataset where ``True`` represents a
      label issue and ``False`` represents an example that is accurately
      labeled with high confidence.
    """

    assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs, multi_label=multi_label)
    if multi_label:
        if not isinstance(labels, list):
            raise TypeError("`labels` must be list when `multi_label=True`.")
        else:
            return _find_predicted_neq_given_multilabel(labels=labels, pred_probs=pred_probs)
    else:
        return np.argmax(pred_probs, axis=1) != np.asarray(labels)


def _find_predicted_neq_given_multilabel(labels: list, pred_probs: np.ndarray) -> np.ndarray:
    """

    Parameters
     ----------
     labels : list
       List of noisy labels for multi-label classification where each example can belong to multiple classes
       (e.g. ``labels = [[1,2],[1],[0],[],...]`` indicates the first example in dataset belongs to both class 1 and class 2).

     pred_probs : np.ndarray
       Predicted-probabilities in the same format expected by the `~cleanlab.filter.find_label_issues` function.

     Returns
     -------
     label_issues_mask : np.ndarray
       A boolean mask for the entire dataset where ``True`` represents a
       label issue and ``False`` represents an example that is accurately
       labeled with high confidence.

    """
    y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
    pred_neq: np.ndarray = np.zeros(y_one.shape).astype(bool)
    for class_num, (label, pred_prob_for_class) in enumerate(zip(y_one.T, pred_probs.T)):
        pred_probs_binary = stack_complement(pred_prob_for_class)
        pred_neq[:, class_num] = find_predicted_neq_given(
            labels=label, pred_probs=pred_probs_binary
        )
    return pred_neq.sum(axis=1) >= 1


def find_label_issues_using_argmax_confusion_matrix(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    calibrate: bool = True,
    filter_by: str = "prune_by_noise_rate",
) -> np.ndarray:
    """A baseline approach that uses the confusion matrix
    of ``argmax(pred_probs)`` and labels as the confident joint and then uses cleanlab
    (confident learning) to find the label issues using this matrix.

    The only difference between this and `~cleanlab.filter.find_label_issues` is that it uses the confusion matrix
    based on the argmax and given label instead of using the confident joint
    from :py:func:`count.compute_confident_joint
    <cleanlab.count.compute_confident_joint>`.

    Parameters
    ----------
    labels : np.ndarray
        An array of shape ``(N,)`` of noisy labels, i.e. some labels may be erroneous.
        Elements must be in the set 0, 1, ..., K-1, where K is the number of classes.

    pred_probs : np.ndarray
        An array of shape ``(N, K)`` of model-predicted probabilities,
        ``P(label=k|x)``. Each row of this matrix corresponds
        to an example `x` and contains the model-predicted probabilities that
        `x` belongs to each possible class, for each of the K classes. The
        columns must be ordered such that these probabilities correspond to
        class 0, 1, ..., K-1. `pred_probs` should have been computed using 3 (or
        higher) fold cross-validation.

    calibrate : bool, default=True
        Set to ``True`` to calibrate the confusion matrix created by ``pred != given labels``.
        This calibration adjusts the confusion matrix / confident joint so that the
        prior (given noisy labels) is correct based on the original labels.

    filter_by : str, default='prune_by_noise_rate'
        See `filter_by` argument of `~cleanlab.filter.find_label_issues`.

    Returns
    -------
    label_issues_mask : np.ndarray
      A boolean mask for the entire dataset where ``True`` represents a
      label issue and ``False`` represents an example that is accurately
      labeled with high confidence.

    """

    assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs, multi_label=False)
    confident_joint = confusion_matrix(np.argmax(pred_probs, axis=1), labels).T
    if calibrate:
        confident_joint = calibrate_confident_joint(confident_joint, labels)
    return find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        confident_joint=confident_joint,
        filter_by=filter_by,
    )


# Multiprocessing helper functions:

mp_params: Dict[str, Any] = {}  # Globals to be shared across threads in multiprocessing


def _to_np_array(
    mp_arr: bytearray, dtype="int32", shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:  # pragma: no cover
    """multipropecessing Helper function to convert a multiprocessing
    RawArray to a numpy array."""
    arr = np.frombuffer(mp_arr, dtype=dtype)
    if shape is None:
        return arr
    return arr.reshape(shape)


def _init(
    __labels,
    __label_counts,
    __prune_count_matrix,
    __pcm_shape,
    __pred_probs,
    __pred_probs_shape,
    __multi_label,
    __min_examples_per_class,
):  # pragma: no cover
    """Shares memory objects across child processes.
    ASSUMES none of these will be changed by child processes!"""

    mp_params["labels"] = __labels
    mp_params["label_counts"] = __label_counts
    mp_params["prune_count_matrix"] = __prune_count_matrix
    mp_params["pcm_shape"] = __pcm_shape
    mp_params["pred_probs"] = __pred_probs
    mp_params["pred_probs_shape"] = __pred_probs_shape
    mp_params["multi_label"] = __multi_label
    mp_params["min_examples_per_class"] = __min_examples_per_class


def _get_shared_data() -> Any:  # pragma: no cover
    """multiprocessing helper function to extract numpy arrays from
    shared RawArray types used to shared data across process."""

    label_counts = _to_np_array(mp_params["label_counts"])
    prune_count_matrix = _to_np_array(
        mp_arr=mp_params["prune_count_matrix"],
        shape=mp_params["pcm_shape"],
    )
    pred_probs = _to_np_array(
        mp_arr=mp_params["pred_probs"],
        dtype="float32",
        shape=mp_params["pred_probs_shape"],
    )
    min_examples_per_class = mp_params["min_examples_per_class"]
    multi_label = mp_params["multi_label"]
    labels = _to_np_array(mp_params["labels"])  # type: ignore
    return (
        labels,
        label_counts,
        prune_count_matrix,
        pred_probs,
        multi_label,
        min_examples_per_class,
    )


# TODO figure out what the types inside args are.
def _prune_by_class(args: list) -> np.ndarray:
    """multiprocessing Helper function for find_label_issues()
    that assumes globals and produces a mask for class k for each example by
    removing the examples with *smallest probability* of
    belonging to their given class label.

    Parameters
    ----------
    k : int (between 0 and num classes - 1)
      The class of interest."""

    k, min_examples_per_class, arrays = args
    if arrays is None:
        pred_probs = pred_probs_by_class[k]
        prune_count_matrix = prune_count_matrix_cols[k]
    else:
        pred_probs = arrays[0]
        prune_count_matrix = arrays[1]

    label_counts = pred_probs.shape[0]
    label_issues = np.zeros(label_counts, dtype=bool)
    if label_counts > min_examples_per_class:  # No prune if not at least min_examples_per_class
        num_issues = label_counts - prune_count_matrix[k]
        # Get return_indices_ranked_by of the smallest prob of class k for examples with noisy label k
        # rank = np.partition(class_probs, num_issues)[num_issues]
        if num_issues >= 1:
            class_probs = pred_probs[:, k]
            order = np.argsort(class_probs)
            label_issues[order[:num_issues]] = True
        return label_issues

    warnings.warn(
        f"May not flag all label issues in class: {k}, it has too few examples (see argument: `min_examples_per_class`)"
    )
    return label_issues


# TODO figure out what the types inside args are.
def _prune_by_count(args: list) -> np.ndarray:
    """multiprocessing Helper function for find_label_issues() that assumes
    globals and produces a mask for class k for each example by
    removing the example with noisy label k having *largest margin*,
    where
    margin of example := prob of given label - max prob of non-given labels

    Parameters
    ----------
    k : int (between 0 and num classes - 1)
      The true_label class of interest."""

    k, min_examples_per_class, arrays = args
    if arrays is None:
        pred_probs = pred_probs_by_class[k]
        prune_count_matrix = prune_count_matrix_cols[k]
    else:
        pred_probs = arrays[0]
        prune_count_matrix = arrays[1]

    label_counts = pred_probs.shape[0]
    label_issues_mask = np.zeros(label_counts, dtype=bool)
    if label_counts <= min_examples_per_class:
        warnings.warn(
            f"May not flag all label issues in class: {k}, it has too few examples (see `min_examples_per_class` argument)"
        )
        return label_issues_mask

    K = pred_probs.shape[1]
    if K < 1:
        raise ValueError("Must have at least 1 class.")
    for j in range(K):
        num2prune = prune_count_matrix[j]
        # Only prune for noise rates, not diagonal entries
        if k != j and num2prune > 0:
            # num2prune's largest p(true class k) - p(noisy class k)
            # for x with true label j
            margin = pred_probs[:, j] - pred_probs[:, k]
            order = np.argsort(-margin)
            label_issues_mask[order[:num2prune]] = True
    return label_issues_mask


# TODO: decide if we want to keep this based on TODO above. If so move to utils. Add unit test for this.
def _multiclass_crossval_predict(
    labels: list, pred_probs: np.ndarray
) -> np.ndarray:  # pragma: no cover
    """Returns a numpy 2D array of one-hot encoded
    multiclass predictions. Each row in the array
    provides the predictions for a particular example.
    The boundary condition used to threshold predictions
    is computed by maximizing the F1 ROC curve.

    Parameters
    ----------
    labels : list of lists (length N)
      These are multiclass labels. Each list in the list contains all the
      labels for that example.

    pred_probs : np.ndarray (shape (N, K))
        P(label=k|x) is a matrix with K model-predicted probabilities.
        Each row of this matrix corresponds to an example `x` and contains the model-predicted
        probabilities that `x` belongs to each possible class.
        The columns must be ordered such that these probabilities correspond to class 0,1,2,...
        `pred_probs` should have been computed using 3 (or higher) fold cross-validation."""

    from sklearn.metrics import f1_score

    boundaries = np.arange(0.05, 0.9, 0.05)
    K = get_num_classes(
        labels=labels,
        pred_probs=pred_probs,
        multi_label=True,
    )
    labels_one_hot = int2onehot(labels, K)
    f1s = [
        f1_score(
            labels_one_hot,
            (pred_probs > boundary).astype(np.uint8),
            average="micro",
        )
        for boundary in boundaries
    ]
    boundary = boundaries[np.argmax(f1s)]
    pred = (pred_probs > boundary).astype(np.uint8)
    return pred
