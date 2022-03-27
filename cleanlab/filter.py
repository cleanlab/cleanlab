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


"""Filter (previously in Cleanlab 1.0, this module was called Pruning)

Contains methods for estimating the latent indices of all label issues.
This code uses advanced multiprocessing to speed up computation.
see: https://research.wmz.ninja/articles/2018/03/ (link continued below)
on-sharing-large-arrays-when-using-pythons-multiprocessing.html
"""


from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import multiprocessing
from multiprocessing.sharedctypes import RawArray
import sys

from cleanlab.rank import order_label_issues
from cleanlab.utils.util import (
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

# Leave at least 1 example in each class after filtering/pruning, even if noise estimates are larger
MIN_NUM_PER_CLASS = 1

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
    return labels, label_counts, prune_count_matrix, pred_probs, multi_label


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
        labels, label_counts, prune_count_matrix, pred_probs, multi_label = args
    else:  # Multiprocessing - data is shared across sub-processes
        labels, label_counts, prune_count_matrix, pred_probs, multi_label = _get_shared_data()

    if label_counts[k] > MIN_NUM_PER_CLASS:  # No prune if not MIN_NUM_PER_CLASS
        num_issues = label_counts[k] - prune_count_matrix[k][k]
        # Get return_indices_ranked_by of the smallest prob of class k for examples with noisy label k
        label_filter = np.array([k in lst for lst in labels]) if multi_label else labels == k
        class_probs = pred_probs[:, k]
        rank = np.partition(class_probs[label_filter], num_issues)[num_issues]
        return label_filter & (class_probs < rank)
    else:
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
        labels, label_counts, prune_count_matrix, pred_probs, multi_label = args
    else:  # Multiprocessing - data is shared across sub-processes
        labels, label_counts, prune_count_matrix, pred_probs, multi_label = _get_shared_data()

    label_issues_mask = np.zeros(len(pred_probs), dtype=bool)
    pred_probs_k = pred_probs[:, k]
    K = len(label_counts)
    if label_counts[k] <= MIN_NUM_PER_CLASS:  # No prune if not MIN_NUM_PER_CLASS
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


def multiclass_crossval_predict(labels, pyx):
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
    n_jobs=None,
    verbose=0,
):
    """By default, this method returns a boolean mask for the entire dataset where True represents
    a label issue and False represents an example that is confidently/accurately labeled.

    You can return ONLY the indices of the label issues in your dataset, by setting
    return_indices_ranked_by = {`self_confidence`, `normalized_margin`}.

    number of indices returned is specified by frac_noise. When
    frac_noise = 1.0, all "confident" estimated noise indices are returned.
    * If you encounter the error 'pred_probs is not defined', try setting n_jobs = 1.

    WARNING! is a matrix with K model-predicted probabilities and num_to_remove_per_class parameters
    are only supported when filter_by is either 'prune_by_noise_rate', 'prune_by_class', or 'both'.
    They are not supported for methods 'confident_learning' or 'predicted_neq_given'. TODO.

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

    confident_joint : np.array (shape (K, K), type int) (default: None)
      A K,K integer matrix of count(label=k, true_label=k). Estimates a a confident
      subset of the joint distribution of the noisy and true labels P_{labels,y}.
      Each entry in the matrix contains the number of examples confidently
      counted into every pair (label=j, true_label=k) classes. The `confident joint` can be computed using
    `count.compute_confident_joint`

    filter_by : str (default: 'prune_by_noise_rate')  TODO: change default to cl_off_diag?
      Possible Values: {'prune_by_class', 'prune_by_noise_rate', 'both',
                        'confident_learning', 'predicted_neq_given'}
      Method used for filtering/pruning out the label issues.
      1. 'prune_by_noise_rate': works by removing examples with
      *high probability* of being mislabeled for every non-diagonal in the confident joint
      (see `prune_counts_matrix` in `filter.py`). These are the examples where (with high
      confidence) the given label is unlikely to match the predicted label for the example.
      2. 'prune_by_class': works by removing the examples with *smallest
      probability* of belonging to their given class label for every class.
      3. 'both': Removes only the examples that would be filtered by both (1) AND (2).
      4. 'confident_learning': Returns the examples in the off-diagonals of the confident joint.
      These are the examples that are confidently predicted to be a different label than their
      given label.
      that's different from their given label while computing the confident joint.
      5. 'predicted_neq_given': Find examples where the predicted class
      (i.e. argmax of the predicted probabilities) does not match the given label.

    return_indices_ranked_by : {:obj:`None`, :obj:`self_confidence`, :obj:`normalized_margin`, :obj:`confidence_weighted_entropy`}
      If None, returns a boolean mask (true if example at index is label error)
      If not None, returns an array of the label error indices
      (instead of a bool mask) where error indices are ordered by the either:
      ``'normalized_margin' := normalized margin (p(label = k) - max(p(label != k)))``
      ``'self_confidence' := [pred_probs[i][labels[i]] for i in label_issues_idx]``
      ``'confidence_weighted_entropy' := entropy(pred_probs) / self_confidence``

    rank_by_kwargs : dict
      Optional keyword arguments to pass into scoring functions for ranking by label quality score
      (see: `get_label_quality_scores()` in cleanlab/rank.py).
      Accepted args include:
      adjust_pred_probs : bool, default = False

    multi_label : bool
      If true, labels should be an iterable (e.g. list) of iterables, containing a
      list of labels for each example, instead of just a single label.
      The multi-label setting supports classification tasks where an example has 1 or more labels.
      Example of a multi-labeled `labels` input: [[0,1], [1], [0,2], [0,1,2], [0], [1], ...]

    frac_noise : float
      When frac_noise = 1.0, return all "confident" estimated noise indices.
      Value in range (0, 1] that determines the fraction of noisy example
      indices to return based on the following formula for example class k.
      frac_noise * number_of_mislabeled_examples_in_class_k, or equivalently
      frac_noise * inverse_noise_rate_class_k * num_examples_with_s_equal_k

    num_to_remove_per_class : list of int of length K (# of classes)
      e.g. if K = 3, num_to_remove_per_class = [5, 0, 1] would return
      the indices of the 5 most likely mislabeled examples in class labels = 0,
      and the most likely mislabeled example in class labels = 1.

      Note
      ----
      Only set this parameter if ``filter_by == 'prune_by_class'``
      You may use with ``filter_by == 'prune_by_noise_rate'``, but
      if ``num_to_remove_per_class == k``, then either k-1, k, or k+1
      examples may be removed for any class. This is because noise rates
      are floats, and rounding may cause a one-off. If you need exactly
      'k' examples removed from every class, you should use ``'prune_by_class'``

    n_jobs : int (Windows users may see a speed-up with n_jobs = 1)
      Number of processing threads used by multiprocessing. Default None
      sets to the number of processing threads on your CPU.
      Set this to 1 to REMOVE parallel processing (if its causing issues).

    verbose : int
      If 0, no print statements. If 1, prints when multiprocessing happens.

    Returns
    -------
    label_issues_mask : np.array<bool>
      This method returns a boolean mask for the entire dataset where True represents
      a label issue and False represents an example that is confidently/accurately labeled.

      Note
      ----
      You can also return ONLY the indices of the label issues in your dataset, by setting
      return_indices_ranked_by = {`self_confidence`, `normalized_margin`}."""

    assert filter_by in [
        "prune_by_noise_rate",
        "prune_by_class",
        "both",
        "confident_learning",
        "predicted_neq_given",
    ]
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
        # leave at least MIN_NUM_PER_CLASS examples per class.
        # `prune_count_matrix` is transposed relative to the confident_joint.
        prune_count_matrix = keep_at_least_n_per_class(
            prune_count_matrix=confident_joint.T,
            n=MIN_NUM_PER_CLASS,
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
            args = (labels, label_counts, prune_count_matrix, pred_probs, multi_label)

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
                ),
            ) as p:
                if verbose:
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
                ),
            ) as p:
                if verbose:
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
        pred = multiclass_crossval_predict(labels, pred_probs)
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
    confident_joint
    return label_issues_mask


def keep_at_least_n_per_class(prune_count_matrix, n, *, frac_noise=1.0):
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

    frac_noise : float
        When frac_noise = 1.0, return all estimated noise indices.
        Value in range (0, 1] that determines the fraction of noisy example
        indices to return based on the following formula for example class k.
        frac_noise * number_of_mislabeled_examples_in_class_k, or
        frac_noise * inverse_noise_rate_class_k * num_examples_s_equal_k

    Returns
    -------

    prune_count_matrix : np.array of shape (K, K), K = number of classes
        Number of examples to remove from each class, for every other class."""

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
    new_mat = reduce_prune_counts(new_mat, frac_noise)

    # These are counts, so return a matrix of ints.
    return round_preserving_row_totals(new_mat).astype(int)


def reduce_prune_counts(prune_count_matrix, frac_noise=1.0):
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
        When frac_noise = 1.0, return all estimated noise indices.
        Value in range (0, 1] that determines the fraction of noisy example
        indices to return based on the following formula for example class k.
        frac_noise * number_of_mislabeled_examples_in_class_k, or
        frac_noise * inverse_noise_rate_class_k * num_examples_s_equal_k."""

    new_mat = prune_count_matrix * frac_noise
    np.fill_diagonal(new_mat, prune_count_matrix.diagonal())
    np.fill_diagonal(
        new_mat, prune_count_matrix.diagonal() + np.sum(prune_count_matrix - new_mat, axis=0)
    )

    # These are counts, so return a matrix of ints.
    return new_mat.astype(int)


def find_predicted_neq_given(labels, pred_probs, *, multi_label=False):
    """This is the simplest baseline approach. Just consider
    anywhere argmax != labels as a label error.

    Parameters
    ----------
    labels : np.array
        A discrete vector of noisy labels, i.e. some labels may be erroneous.
        *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    pred_probs : np.array (shape (N, K))
        P(label=k|x) is a matrix with K model-predicted probabilities.
        Each row of this matrix corresponds to an example `x` and contains the model-predicted
        probabilities that `x` belongs to each possible class.
        The columns must be ordered such that these probabilities correspond to class 0,1,2,...
        `pred_probs` should have been computed using 3 (or higher) fold cross-validation.

    multi_label : bool
        Set to True if labels is multi-label (list of lists, or np.array of np.array)
        The multi-label setting supports classification tasks where an example has 1 or more labels.
        Example of a multi-labeled `labels` input: [[0,1], [1], [0,2], [0,1,2], [0], [1], ...]

    Returns
    -------
        A boolean mask that is true if the example belong
        to that index is label error."""

    assert len(pred_probs) == len(labels)
    if multi_label:
        # Todo: This needs to be tested.
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
    """This is a baseline approach. That uses the confusion matrix
    of argmax(pred_probs) and labels as the confident joint and then uses cleanlab
    (confident learning) to find the label issues using this matrix.

    The only difference between this and find_label_issues is that it uses the confusion matrix
    based on the argmax and given label instead of using the confident joint from
    `count.compute_confident_joint`.

    This method does not support multi-label labels.

    Parameters
    ----------

    labels : np.array
        A discrete vector of noisy labels, i.e. some labels may be erroneous.
        *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    pred_probs : np.array (shape (N, K))
        P(label=k|x) is a matrix with K model-predicted probabilities.
        Each row of this matrix corresponds to an example `x` and contains the model-predicted
        probabilities that `x` belongs to each possible class.
        The columns must be ordered such that these probabilities correspond to class 0,1,2,...
        `pred_probs` should have been computed using 3 (or higher) fold cross-validation.

    calibrate : bool
        Set to True to calibrate the confusion matrix created by pred != given labels.
        This calibration adjusts the confusion matrix / confident joint so that the
        prior(given noisy labels) is correct based on the original labels.

    filter_by : str (default: 'prune_by_noise_rate')
        Possible Values: 'prune_by_class', 'prune_by_noise_rate', or 'both'.
        Method used for pruning/filtering out the label issues:
        1. 'prune_by_noise_rate': works by removing examples with
        *high probability* of being mislabeled for every non-diagonal in the confident joint
        (see `prune_counts_matrix` in `filter.py`). These are the examples where (with high
        confidence) the given label is unlikely to match the predicted label.
        2. 'prune_by_class': works by removing the examples with *smallest
        probability* of belonging to their given class label for every class.
        3. 'both': Finds the examples satisfying (1) AND (2) and
        removes their set conjunction.
        4. 'confident_learning': Find examples that are confidently labeled as a class
        that's different from their given label while computing the confident joint.

    Returns
    -------
        A boolean mask: true if the example at that index is label issue."""

    assert len(pred_probs) == len(labels)
    confident_joint = confusion_matrix(np.argmax(pred_probs, axis=1), labels).T
    if calibrate:
        confident_joint = calibrate_confident_joint(confident_joint, labels)
    return find_label_issues(
        labels=labels, pred_probs=pred_probs, confident_joint=confident_joint, filter_by=filter_by
    )
