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
Methods to identify which examples have label issues.
"""

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import multiprocessing
from multiprocessing.sharedctypes import RawArray
import sys

from cleanlab.rank import order_label_issues
from cleanlab.internal.util import (
    value_counts,
    round_preserving_row_totals,
    onehot2int,
    int2onehot,
)
import numpy as np
import warnings

# tqdm is a module used to print time-to-complete when multiprocessing is used.
# This module is not necessary, and therefore is not a package dependency, but
# when installed it improves user experience for large datasets.
from cleanlab.count import calibrate_confident_joint

try:
    import tqdm

    tqdm_exists = True
except ImportError as e:  # pragma: no cover
    tqdm_exists = False

    w = """To see estimated completion times for methods in cleanlab.filter, "pip install tqdm"."""
    warnings.warn(w)

# Globals to be shared across threads in multiprocessing
mp_params = {}  # parameters passed to multiprocessing helper functions


# Multiprocessing Helper functions


def _to_np_array(mp_arr, dtype="int32", shape=None):  # pragma: no cover
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


def _get_shared_data():  # pragma: no cover
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
    if multi_label:  # Shared data is passed as one-hot encoded matrix
        labels = onehot2int(
            _to_np_array(
                mp_arr=mp_params["labels"],
                shape=(pred_probs.shape[0], pred_probs.shape[1]),
            )
        )
    else:
        labels = _to_np_array(mp_params["labels"])
    return (
        labels,
        label_counts,
        prune_count_matrix,
        pred_probs,
        multi_label,
        min_examples_per_class,
    )


def _prune_by_class(k, args=None):
    """multiprocessing Helper function for find_label_issues()
    that assumes globals and produces a mask for class k for each example by
    removing the examples with *smallest probability* of
    belonging to their given class label.

    Parameters
    ----------
    k : int (between 0 and num classes - 1)
      The class of interest."""

    if args:  # Single processing - params are passed in
        (
            labels,
            label_counts,
            prune_count_matrix,
            pred_probs,
            multi_label,
            min_examples_per_class,
        ) = args
    else:  # Multiprocessing - data is shared across sub-processes
        (
            labels,
            label_counts,
            prune_count_matrix,
            pred_probs,
            multi_label,
            min_examples_per_class,
        ) = _get_shared_data()

    if label_counts[k] > min_examples_per_class:  # No prune if not at least min_examples_per_class
        num_issues = label_counts[k] - prune_count_matrix[k][k]
        # Get return_indices_ranked_by of the smallest prob of class k for examples with noisy label k
        label_filter = np.array([k in lst for lst in labels]) if multi_label else labels == k
        class_probs = pred_probs[:, k]
        rank = np.partition(class_probs[label_filter], num_issues)[num_issues]
        return label_filter & (class_probs < rank)
    else:
        warnings.warn(
            f"May not flag all label issues in class: {k}, it has too few examples (see argument: `min_examples_per_class`)"
        )
        return np.zeros(len(labels), dtype=bool)


def _prune_by_count(k, args=None):
    """multiprocessing Helper function for find_label_issues() that assumes
    globals and produces a mask for class k for each example by
    removing the example with noisy label k having *largest margin*,
    where
    margin of example := prob of given label - max prob of non-given labels

    Parameters
    ----------
    k : int (between 0 and num classes - 1)
      The true_label class of interest."""

    if args:  # Single processing - params are passed in
        (
            labels,
            label_counts,
            prune_count_matrix,
            pred_probs,
            multi_label,
            min_examples_per_class,
        ) = args
    else:  # Multiprocessing - data is shared across sub-processes
        (
            labels,
            label_counts,
            prune_count_matrix,
            pred_probs,
            multi_label,
            min_examples_per_class,
        ) = _get_shared_data()

    label_issues_mask = np.zeros(len(pred_probs), dtype=bool)
    pred_probs_k = pred_probs[:, k]
    K = len(label_counts)
    if label_counts[k] <= min_examples_per_class:  # No prune if not at least min_examples_per_class
        warnings.warn(
            f"May not flag all label issues in class: {k}, it has too few examples (see `min_examples_per_class` argument)"
        )
        return np.zeros(len(labels), dtype=bool)
    for j in range(K):  # j is true label index (k is noisy label index)
        num2prune = prune_count_matrix[j][k]
        # Only prune for noise rates, not diagonal entries
        if k != j and num2prune > 0:
            # num2prune's largest p(true class k) - p(noisy class k)
            # for x with true label j
            margin = pred_probs[:, j] - pred_probs_k
            label_filter = np.array([k in lst for lst in labels]) if multi_label else labels == k
            cut = -np.partition(-margin[label_filter], num2prune - 1)[num2prune - 1]
            label_issues_mask = label_issues_mask | (label_filter & (margin >= cut))
    return label_issues_mask


def _self_confidence(args, _pred_probs):  # pragma: no cover
    """multiprocessing Helper function for find_label_issues() that assumes
    global pred_probs and computes the self-confidence (prob of given label)
    for an example (row in pred_probs) given the example index idx
    and its label l.
    np.mean(pred_probs[]) enables this code to work for multi-class l."""
    (idx, l) = args
    return np.mean(_pred_probs[idx, l])


def _multiclass_crossval_predict(labels, pyx):
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

    pyx : np.array (shape (N, K))
        P(label=k|x) is a matrix with K model-predicted probabilities.
        Each row of this matrix corresponds to an example `x` and contains the model-predicted
        probabilities that `x` belongs to each possible class.
        The columns must be ordered such that these probabilities correspond to class 0,1,2,...
        `pred_probs` should have been computed using 3 (or higher) fold cross-validation."""

    from sklearn.metrics import f1_score

    boundaries = np.arange(0.05, 0.9, 0.05)
    labels_one_hot = MultiLabelBinarizer().fit_transform(labels)
    f1s = [
        f1_score(
            labels_one_hot,
            (pyx > boundary).astype(np.uint8),
            average="micro",
        )
        for boundary in boundaries
    ]
    boundary = boundaries[np.argmax(f1s)]
    pred = (pyx > boundary).astype(np.uint8)
    return pred


def find_label_issues(
    labels,
    pred_probs,
    *,
    confident_joint=None,
    filter_by="prune_by_noise_rate",
    return_indices_ranked_by=None,
    rank_by_kwargs={},
    multi_label=False,
    frac_noise=1.0,
    num_to_remove_per_class=None,
    min_examples_per_class=1,
    n_jobs=None,
    verbose=False,
):
    """
    Identifies potential label issues in the dataset using confident learning.

    Returns a boolean mask for the entire dataset where ``True`` represents
    a label issue and ``False`` represents an example that is confidently/accurately labeled.

    Instead of a mask, you can obtain *indices* of the label issues in your
    dataset by setting `return_indices_ranked_by` to specify the label quality
    score used to order the label issues.

    The number of indices returned is controlled by `frac_noise`: reduce its
    value to identify fewer label issues. If you aren't sure, leave this set to 1.0.

    Tip: if you encounter the error "pred_probs is not defined", try setting
    ``n_jobs=1``.

    Parameters
    ----------
    labels : np.array
      A discrete vector of noisy labels, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, labels must be in 0, 1, ..., K-1.
      All the classes (0, 1, ..., and K-1) MUST be present in ``labels``, such that:
      ``len(set(labels)) == pred_probs.shape[1]`` for standard multi-class classification with single-labeled data (e.g. ``labels =  [1,0,2,1,1,0...]``).
      For multi-label classification where each example can belong to multiple classes(e.g. ``labels = [[1,2],[1],[0],..]``),
      your labels should instead satisfy: ``len(set(k for l in labels for k in l)) == pred_probs.shape[1])``.

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

    confident_joint : np.array, optional
      An array of shape ``(K, K)`` representing the confident joint, the matrix used for identifying label issues, which
      estimates a confident subset of the joint distribution of the noisy and true labels, ``P_{noisy label, true label}``.
      Entry ``(j, k)`` in the matrix is the number of examples confidently counted into the pair of ``(noisy label=j, true label=k)`` classes.
      The `confident_joint` can be computed using :py:func:`count.compute_confident_joint <cleanlab.count.compute_confident_joint>`.
      If not provided, it is computed from the given (noisy) `labels` and `pred_probs`.

    filter_by : {'prune_by_class', 'prune_by_noise_rate', 'both', 'confident_learning', 'predicted_neq_given'}, default='prune_by_noise_rate'

      Method used for filtering/pruning out the label issues:

      - ``'prune_by_noise_rate'``: works by removing examples with *high probability* of being mislabeled for every non-diagonal in the confident joint (see `prune_counts_matrix` in `filter.py`). These are the examples where (with high confidence) the given label is unlikely to match the predicted label for the example.
      - ``'prune_by_class'``: works by removing the examples with *smallest probability* of belonging to their given class label for every class.
      - ``'both'``: Removes only the examples that would be filtered by both ``'prune_by_noise_rate'`` and ``'prune_by_class'``.
      - ``'confident_learning'``: Returns the examples in the off-diagonals of the confident joint. These are the examples that are confidently predicted to be a different label than their given label.
      - ``'predicted_neq_given'``: Find examples where the predicted class (i.e. argmax of the predicted probabilities) does not match the given label.

    return_indices_ranked_by : {None, 'self_confidence', 'normalized_margin', 'confidence_weighted_entropy'}, default=None
      If ``None``, returns a boolean mask (``True`` if example at index is label error).
      If not ``None``, returns an array of the label error indices
      (instead of a boolean mask) where error indices are ordered:

      - ``'normalized_margin'``: ``normalized margin (p(label = k) - max(p(label != k)))``
      - ``'self_confidence'``: ``[pred_probs[i][labels[i]] for i in label_issues_idx]``
      - ``'confidence_weighted_entropy'``: ``entropy(pred_probs) / self_confidence``

    rank_by_kwargs : dict, optional
      Optional keyword arguments to pass into scoring functions for ranking by
      label quality score (see :py:func:`rank.get_label_quality_scores
      <cleanlab.rank.get_label_quality_scores>`).

    multi_label : bool, optional
      If ``True``, labels should be an iterable (e.g. list) of iterables, containing a
      list of labels for each example, instead of just a single label.
      The multi-label setting supports classification tasks where an example has 1 or more labels.
      Example of a multi-labeled `labels` input: ``[[0,1], [1], [0,2], [0,1,2], [0], [1], ...]``.

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

    n_jobs : optional
      Number of processing threads used by multiprocessing. Default ``None``
      sets to the number of cores on your CPU.
      Set this to 1 to *disable* parallel processing (if its causing issues).
      Windows users may see a speed-up with ``n_jobs=1``.

    verbose : optional
      If ``True``, prints when multiprocessing happens.

    Returns
    -------
    label_issues : np.array
      A boolean mask for the entire dataset where ``True`` represents a
      label issue and ``False`` represents an example that is accurately
      labeled with high confidence.

      Note
      ----
      You can also return the *indices* of the label issues in your dataset by setting
      `return_indices_ranked_by`.
    """

    assert filter_by in [
        "prune_by_noise_rate",
        "prune_by_class",
        "both",
        "confident_learning",
        "predicted_neq_given",
    ]  # TODO: change default to confident_learning ?
    assert len(labels) == len(pred_probs)
    if filter_by in ["confident_learning", "predicted_neq_given"] and (
        frac_noise != 1.0 or num_to_remove_per_class is not None
    ):
        warn_str = (
            "WARNING! frac_noise and num_to_remove_per_class parameters are only supported"
            " for filter_by 'prune_by_noise_rate', 'prune_by_class', and 'both'. They "
            "are not supported for methods 'confident_learning' or "
            "'predicted_neq_given'."
        )
        warnings.warn(warn_str)
    if (num_to_remove_per_class is not None) and (
        filter_by in ["confident_learning", "predicted_neq_given"]
    ):
        # TODO - add support for these two filters
        raise ValueError(
            "filter_by 'confident_learning' or 'predicted_neq_given' is not supported (yet) when setting 'num_to_remove_per_class'"
        )

    # Set-up number of multiprocessing threads
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    else:
        assert n_jobs >= 1

    # Number of examples in each class of labels
    if multi_label:
        label_counts = value_counts([i for lst in labels for i in lst])
    else:
        label_counts = value_counts(labels)
    # Number of classes labels
    K = len(pred_probs.T)
    # Boolean set to true if dataset is large
    big_dataset = K * len(labels) > 1e8
    # Ensure labels are of type np.array()
    labels = np.asarray(labels)
    if confident_joint is None or filter_by == "confident_learning":
        from cleanlab.count import compute_confident_joint

        confident_joint, cl_error_indices = compute_confident_joint(
            labels=labels,
            pred_probs=pred_probs,
            multi_label=multi_label,
            return_indices_of_off_diagonals=True,
        )
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
        if n_jobs > 1:
            if multi_label:
                _labels = RawArray("I", int2onehot(labels).flatten())
            else:
                _labels = RawArray("I", labels)
            _label_counts = RawArray("I", label_counts)
            _prune_count_matrix = RawArray("I", prune_count_matrix.flatten())
            _pred_probs = RawArray("f", pred_probs.flatten())
        else:  # Multiprocessing is turned off. Create tuple with all parameters
            args = (
                labels,
                label_counts,
                prune_count_matrix,
                pred_probs,
                multi_label,
                min_examples_per_class,
            )

    # Perform Pruning with threshold probabilities from BFPRT algorithm in O(n)
    # Operations are parallelized across all CPU processes
    if filter_by == "prune_by_class" or filter_by == "both":
        if n_jobs > 1:  # parallelize
            with multiprocessing.Pool(
                n_jobs,
                initializer=_init,
                initargs=(
                    _labels,
                    _label_counts,
                    _prune_count_matrix,
                    prune_count_matrix.shape,
                    _pred_probs,
                    pred_probs.shape,
                    multi_label,
                    min_examples_per_class,
                ),
            ) as p:
                if verbose:  # pragma: no cover
                    print("Parallel processing label issues by class.")
                sys.stdout.flush()
                if big_dataset and tqdm_exists:
                    label_issues_masks_per_class = list(
                        tqdm.tqdm(p.imap(_prune_by_class, range(K)), total=K),
                    )
                else:
                    label_issues_masks_per_class = p.map(_prune_by_class, range(K))
        else:  # n_jobs = 1, so no parallelization
            label_issues_masks_per_class = [_prune_by_class(k, args) for k in range(K)]
        label_issues_mask = np.stack(label_issues_masks_per_class).any(axis=0)

    if filter_by == "both":
        label_issues_mask_by_class = label_issues_mask

    if filter_by == "prune_by_noise_rate" or filter_by == "both":
        if n_jobs > 1:  # parallelize
            with multiprocessing.Pool(
                n_jobs,
                initializer=_init,
                initargs=(
                    _labels,
                    _label_counts,
                    _prune_count_matrix,
                    prune_count_matrix.shape,
                    _pred_probs,
                    pred_probs.shape,
                    multi_label,
                    min_examples_per_class,
                ),
            ) as p:
                if verbose:  # pragma: no cover
                    print("Parallel processing label issues by noise rate.")
                sys.stdout.flush()
                if big_dataset and tqdm_exists:
                    label_issues_masks_per_class = list(
                        tqdm.tqdm(p.imap(_prune_by_count, range(K)), total=K)
                    )
                else:
                    label_issues_masks_per_class = p.map(_prune_by_count, range(K))
        else:  # n_jobs = 1, so no parallelization
            label_issues_masks_per_class = [_prune_by_count(k, args) for k in range(K)]
        label_issues_mask = np.stack(label_issues_masks_per_class).any(axis=0)

    if filter_by == "both":
        label_issues_mask = label_issues_mask & label_issues_mask_by_class

    if filter_by == "confident_learning":
        label_issues_mask = np.zeros(len(labels), dtype=bool)
        for idx in cl_error_indices:
            label_issues_mask[idx] = True

    if filter_by == "predicted_neq_given":
        label_issues_mask = find_predicted_neq_given(labels, pred_probs, multi_label=multi_label)

    # Remove label issues if given label == model prediction
    if multi_label:
        pred = _multiclass_crossval_predict(labels, pred_probs)
        labels = MultiLabelBinarizer().fit_transform(labels)
    else:
        pred = pred_probs.argmax(axis=1)
    for i, pred_label in enumerate(pred):
        if (
            multi_label
            and np.all(pred_label == labels[i])
            or not multi_label
            and pred_label == labels[i]
        ):
            label_issues_mask[i] = False

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


def _keep_at_least_n_per_class(prune_count_matrix, n, *, frac_noise=1.0):
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
    prune_count_matrix : np.array of shape (K, K), K = number of classes
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
    prune_count_matrix : np.array of shape (K, K), K = number of classes
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


def _reduce_prune_counts(prune_count_matrix, frac_noise=1.0):
    """Reduce (multiply) all prune counts (non-diagonal) by frac_noise and
    increase diagonal by the total amount reduced in each column to
    preserve column counts.

    Parameters
    ----------
    prune_count_matrix : np.array of shape (K, K), K = number of classes
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


def find_predicted_neq_given(labels, pred_probs, *, multi_label=False):
    """A simple baseline approach that considers ``argmax(pred_probs) != labels`` as a label error.

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

    multi_label : bool, optional
      If ``True``, labels should be an iterable (e.g. list) of iterables, containing a
      list of labels for each example, instead of just a single label.
      The multi-label setting supports classification tasks where an example has 1 or more labels.
      Example of a multi-labeled `labels` input: ``[[0,1], [1], [0,2], [0,1,2], [0], [1], ...]``.

    Returns
    -------
    label_issues_mask : np.array
      A boolean mask for the entire dataset where ``True`` represents a
      label issue and ``False`` represents an example that is accurately
      labeled with high confidence.
    """

    assert len(pred_probs) == len(labels)
    if multi_label:
        # TODO: This needs to be tested.
        return np.array(
            [all(np.argsort(pred_probs[i])[: len(j)] == sorted(j)) for i, j in enumerate(labels)]
        )
    return np.argmax(pred_probs, axis=1) != np.asarray(labels)


def find_label_issues_using_argmax_confusion_matrix(
    labels,
    pred_probs,
    *,
    calibrate=True,
    filter_by="prune_by_noise_rate",
):
    """This is a baseline approach that uses the confusion matrix
    of ``argmax(pred_probs)`` and labels as the confident joint and then uses cleanlab
    (confident learning) to find the label issues using this matrix.

    The only difference between this and :py:func:`find_label_issues
    <cleanlab.filter.find_label_issues>` is that it uses the confusion matrix
    based on the argmax and given label instead of using the confident joint
    from :py:func:`count.compute_confident_joint
    <cleanlab.count.compute_confident_joint>`.

    Parameters
    ----------
    labels : np.array
        An array of shape ``(N,)`` of noisy labels, i.e. some labels may be erroneous.
        Elements must be in the set 0, 1, ..., K-1, where K is the number of classes.

    pred_probs : np.array (shape (N, K))
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
        See `filter_by` argument of :py:func:`find_label_issues <cleanlab.filter.find_label_issues>`.

    Returns
    -------
    label_issues_mask : np.array
      A boolean mask for the entire dataset where ``True`` represents a
      label issue and ``False`` represents an example that is accurately
      labeled with high confidence.
    """

    assert len(pred_probs) == len(labels)
    confident_joint = confusion_matrix(np.argmax(pred_probs, axis=1), labels).T
    if calibrate:
        confident_joint = calibrate_confident_joint(confident_joint, labels)
    return find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        confident_joint=confident_joint,
        filter_by=filter_by,
    )
