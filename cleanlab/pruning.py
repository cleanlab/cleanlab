# coding: utf-8

# Copyright (c) 2017-2050 Curtis G. Northcutt

# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License

# This agreement applies to this version and all previous versions of cleanlab.


# ## Pruning
# 
# #### Contains methods for estimating the latent indices of all label errors.
# This code uses advanced multiprocessing to speed up computation.
# see: https://research.wmz.ninja/articles/2018/03/ (link continued below)
# on-sharing-large-arrays-when-using-pythons-multiprocessing.html
# This approach supports posix and nt OS's: i.e. Windows, Mac, and Linux


from __future__ import (
    print_function, absolute_import, division, unicode_literals, with_statement)
from sklearn.preprocessing import MultiLabelBinarizer
import multiprocessing
from multiprocessing.sharedctypes import RawArray
import sys
import os
import time
from cleanlab.util import (value_counts, round_preserving_row_totals,
                           onehot2int, int2onehot, )
import numpy as np

# tqdm is a module used to print time-to-complete when multiprocessing is used.
# This module is not necessary, and therefore is not a package dependency, but 
# when installed it improves user experience for large datasets.
try:
    import tqdm

    tqdm_exists = True
except ImportError as e:
    tqdm_exists = False
    import warnings

    w = '''If you want to see estimated completion times
    while running methods in cleanlab.pruning, install tqdm
    via "pip install tqdm".'''
    warnings.warn(w)

# Leave at least this many examples in each class after
# pruning, regardless if noise estimates are larger.
MIN_NUM_PER_CLASS = 5

# For python 2/3 compatibility, define pool context manager
# to support the 'with' statement in Python 2
if sys.version_info[0] == 2:
    from contextlib import contextmanager


    @contextmanager
    def multiprocessing_context(*args, **kwargs):
        pool = multiprocessing.Pool(*args, **kwargs)
        yield pool
        pool.terminate()
else:
    multiprocessing_context = multiprocessing.Pool

# Globals to be shared across threads in multiprocessing
mp_params = {}  # parameters passed to multiprocessing helper functions


# Multiprocessing Helper functions


def _to_np_array(mp_arr, dtype="int32", shape=None):  # pragma: no cover
    """multipropcessing Helper function to convert a multiprocessing
    RawArray to a numpy array."""
    arr = np.frombuffer(mp_arr, dtype=dtype)
    if shape is None:
        return arr
    return arr.reshape(shape)


def _init(
        __s,
        __s_counts,
        __prune_count_matrix,
        __pcm_shape,
        __psx,
        __psx_shape,
        __multi_label,
):  # pragma: no cover
    """Shares memory objects across child processes.
    ASSUMES none of these will be changed by child processes!"""

    mp_params['s'] = __s
    mp_params['s_counts'] = __s_counts
    mp_params['prune_count_matrix'] = __prune_count_matrix
    mp_params['pcm_shape'] = __pcm_shape
    mp_params['psx'] = __psx
    mp_params['psx_shape'] = __psx_shape
    mp_params['multi_label'] = __multi_label


def _get_shared_data():  # pragma: no cover
    """multiprocessing helper function to extract numpy arrays from
    shared RawArray types used to shared data across process."""

    s_counts = _to_np_array(mp_params['s_counts'])
    prune_count_matrix = _to_np_array(
        mp_arr=mp_params['prune_count_matrix'],
        shape=mp_params['pcm_shape'],
    )
    psx = _to_np_array(
        mp_arr=mp_params['psx'],
        dtype='float32',
        shape=mp_params['psx_shape'],
    )
    multi_label = mp_params['multi_label']
    if multi_label:  # Shared data is passed as one-hot encoded matrix
        print('before', mp_params['s'])
        s = onehot2int(_to_np_array(
            mp_arr=mp_params['s'],
            shape=(psx.shape[0], psx.shape[1]),
        ))
        print('after', s)
    else:
        s = _to_np_array(mp_params['s'])
    return s, s_counts, prune_count_matrix, psx, multi_label


def _prune_by_class(k, args=None):
    """multiprocessing Helper function for get_noise_indices()
    that assumes globals and produces a mask for class k for each example by
    removing the examples with *smallest probability* of
    belonging to their given class label.

    Parameters
    ----------
    k : int (between 0 and num classes - 1)
      The class of interest."""

    if args:  # Single processing - params are passed in
        s, s_counts, prune_count_matrix, psx, multi_label = args
    else:  # Multiprocessing - data is shared across sub-processes
        s, s_counts, prune_count_matrix, psx, multi_label = _get_shared_data()

    if s_counts[k] > MIN_NUM_PER_CLASS:  # No prune if not MIN_NUM_PER_CLASS
        num_errors = s_counts[k] - prune_count_matrix[k][k]
        # Get rank of smallest prob of class k for examples with noisy label k
        s_filter = np.array(
            [k in lst for lst in s]) if multi_label else s == k
        class_probs = psx[:, k]
        rank = np.partition(class_probs[s_filter], num_errors)[num_errors]
        return s_filter & (class_probs < rank)
    else:
        return np.zeros(len(s), dtype=bool)


def _prune_by_count(k, args=None):
    """multiprocessing Helper function for get_noise_indices() that assumes
    globals and produces a mask for class k for each example by
    removing the example with noisy label k having *largest margin*,
    where
    margin of example := prob of given label - max prob of non-given labels

    Parameters
    ----------
    k : int (between 0 and num classes - 1)
      The true, hidden label class of interest."""

    if args:  # Single processing - params are passed in
        s, s_counts, prune_count_matrix, psx, multi_label = args
    else:  # Multiprocessing - data is shared across sub-processes
        s, s_counts, prune_count_matrix, psx, multi_label = _get_shared_data()

    noise_mask = np.zeros(len(psx), dtype=bool)
    psx_k = psx[:, k]
    K = len(s_counts)
    if s_counts[k] <= MIN_NUM_PER_CLASS:  # No prune if not MIN_NUM_PER_CLASS
        return np.zeros(len(s), dtype=bool)
    for j in range(K):  # j is true label index (k is noisy label index)
        num2prune = prune_count_matrix[j][k]
        # Only prune for noise rates, not diagonal entries
        if k != j and num2prune > 0:
            # num2prune'th largest p(true class k) - p(noisy class k)
            # for x with true label j
            margin = psx[:, j] - psx_k
            s_filter = np.array(
                [k in lst for lst in s]
            ) if multi_label else s == k
            cut = -np.partition(-margin[s_filter], num2prune - 1)[num2prune - 1]
            noise_mask = noise_mask | (s_filter & (margin >= cut))
    return noise_mask


def _self_confidence(args, _psx):  # pragma: no cover
    """multiprocessing Helper function for get_noise_indices() that assumes
    global psx and computes the self confidence (prob of given label)
    for an example (row in psx) given the example index idx
    and its label l.
    np.mean(psx[]) enables this code to work for multi-class l."""
    (idx, l) = args
    return np.mean(_psx[idx, l])


def multiclass_crossval_predict(pyx, labels):
    """Returns an numpy 2D array of one-hot encoded
    multiclass predictions. Each row in the array
    provides the predictions for a particular example.
    The boundary condition used to threshold predictions
    is computed by maximizing the F1 ROC curve.

    Parameters
    ----------
    pyx : np.array (shape (N, K))
      P(label=k|x) is a NxK matrix with K probs for each of N examples.
      This is the probability distribution over all K classes, for each
      pyx should have been computed out of sample (holdout or crossval).

    labels : list of lists (length N)
      These are multiclass labels. Each list in the list contains all the
      labels for that example."""

    from sklearn.metrics import f1_score
    boundaries = np.arange(0.05, 0.9, .05)
    labels_one_hot = MultiLabelBinarizer().fit_transform(labels)
    f1s = [f1_score(
        labels_one_hot, (pyx > boundary).astype(np.uint8), average='micro',
    ) for boundary in boundaries]
    boundary = boundaries[np.argmax(f1s)]
    pred = (pyx > boundary).astype(np.uint8)
    return pred


def get_noise_indices(
        s,
        psx,
        inverse_noise_matrix=None,
        confident_joint=None,
        frac_noise=1.0,
        num_to_remove_per_class=None,
        prune_method='prune_by_noise_rate',
        sorted_index_method=None,
        multi_label=False,
        n_jobs=None,
        verbose=0,
):
    """Returns the indices of most likely (confident) label errors in s. The
    number of indices returned is specified by frac_of_noise. When
    frac_of_noise = 1.0, all "confident" estimated noise indices are returned.
    * If you encounter the error 'psx is not defined', try setting n_jobs = 1.

    Parameters
    ----------

    s : np.array
      A binary vector of labels, s, which may contain mislabeling. "s" denotes
      the noisy label instead of \tilde(y), for ASCII encoding reasons.

    psx : np.array (shape (N, K))
      P(s=k|x) is a matrix with K (noisy) probabilities for each of the N
      examples x.
      This is the probability distribution over all K classes, for each
      example, regarding whether the example has label s==k P(s=k|x).
      psx should have been computed using 3+ fold cross-validation.

    inverse_noise_matrix : np.array of shape (K, K), K = number of classes
      A conditional probability matrix of the form P(y=k_y|s=k_s) representing
      the estimated fraction observed examples in each class k_s, that are
      mislabeled examples from every other class k_y. If None, the
      inverse_noise_matrix will be computed from psx and s.
      Assumes columns of inverse_noise_matrix sum to 1.

    confident_joint : np.array (shape (K, K), type int) (default: None)
      A K,K integer matrix of count(s=k, y=k). Estimates a a confident
      subset of the joint distribution of the noisy and true labels P_{s,y}.
      Each entry in the matrix contains the number of examples confidently
      counted into every pair (s=j, y=k) classes.

    frac_noise : float
      When frac_of_noise = 1.0, return all "confident" estimated noise indices.
      Value in range (0, 1] that determines the fraction of noisy example
      indices to return based on the following formula for example class k.
      frac_of_noise * number_of_mislabeled_examples_in_class_k, or equivalently
      frac_of_noise * inverse_noise_rate_class_k * num_examples_with_s_equal_k

    num_to_remove_per_class : list of int of length K (# of classes)
      e.g. if K = 3, num_to_remove_per_class = [5, 0, 1] would return
      the indices of the 5 most likely mislabeled examples in class s = 0,
      and the most likely mislabeled example in class s = 1.
      ***Only set this parameter if prune_method == 'prune_by_class'
      You may use with prune_method == 'prune_by_noise_rate', but
      if num_to_remove_per_class == k, then either k-1, k, or k+1
      examples may be removed for any class. This is because noise rates
      are floats, and rounding may cause a one-off. If you need exactly
      'k' examples removed from every class, you should use 'prune_by_class'.

    prune_method : str (default: 'prune_by_noise_rate')
      Possible Values: 'prune_by_class', 'prune_by_noise_rate', or 'both'.
      Method used for pruning.
      1. 'prune_by_noise_rate': works by removing examples with
      *high probability* of being mislabeled for every non-diagonal
      in the prune_counts_matrix (see pruning.py).
      2. 'prune_by_class': works by removing the examples with *smallest
      probability* of belonging to their given class label for every class.
      3. 'both': Finds the examples satisfying (1) AND (2) and
      removes their set conjunction.

    sorted_index_method : str [None, 'prob_given_label', 'normalized_margin']
      If None, returns a boolean mask (true if example at index is label error)
      If not None, returns an array of the label error indices
      (instead of a bool mask) where error indices are ordered by the either:
        'normalized_margin' := normalized margin (p(s = k) - max(p(s != k)))
        'prob_given_label' := [psx[i][labels[i]] for i in label_errors_idx]

    multi_label : bool
      If true, s should be an iterable (e.g. list) of iterables, containing a
      list of labels for each example, instead of just a single label.

    n_jobs : int (Windows users may see a speed-up with n_jobs = 1)
      Number of processing threads used by multiprocessing. Default None
      sets to the number of processing threads on your CPU.
      Set this to 1 to REMOVE parallel processing (if its causing issues).

    verbose : int
      If 0, no print statements. If 1, prints when multiprocessing happens."""

    # Set-up number of multiprocessing threads
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    else:
        assert (n_jobs >= 1)

    # Number of examples in each class of s
    if multi_label:
        s_counts = value_counts([i for lst in s for i in lst])
    else:
        s_counts = value_counts(s)
    # Number of classes s
    K = len(psx.T)
    # Boolean set to true if dataset is large
    big_dataset = K * len(s) > 1e8
    # Ensure labels are of type np.array()
    s = np.asarray(s)

    if confident_joint is None:
        from cleanlab.latent_estimation import compute_confident_joint
        confident_joint = compute_confident_joint(
            s=s,
            psx=psx,
            multi_label=multi_label,
        )

    # Leave at least MIN_NUM_PER_CLASS examples per class.
    # NOTE prune_count_matrix is transposed (relative to confident_joint)
    prune_count_matrix = keep_at_least_n_per_class(
        prune_count_matrix=confident_joint.T,
        n=MIN_NUM_PER_CLASS,
        frac_noise=frac_noise,
    )

    if num_to_remove_per_class is not None:
        # Estimate joint probability distribution over label errors
        psy = prune_count_matrix / np.sum(prune_count_matrix, axis=1)
        noise_per_s = psy.sum(axis=1) - psy.diagonal()
        # Calibrate s.t. noise rates sum to num_to_remove_per_class
        tmp = (psy.T * num_to_remove_per_class / noise_per_s).T
        np.fill_diagonal(tmp, s_counts - num_to_remove_per_class)
        prune_count_matrix = round_preserving_row_totals(tmp)

    if n_jobs > 1:  # Prepare multiprocessing shared data
        if multi_label:
            _s = RawArray('I', int2onehot(s).flatten())
        else:
            _s = RawArray('I', s)
        _s_counts = RawArray('I', s_counts)
        _prune_count_matrix = RawArray(
            'I', prune_count_matrix.flatten())
        _psx = RawArray(
            'f', psx.flatten())
    else:  # Multiprocessing is turned off. Create tuple with all parameters
        args = (s, s_counts, prune_count_matrix, psx, multi_label)

    # Perform Pruning with threshold probabilities from BFPRT algorithm in O(n)
    # Operations are parallelized across all CPU processes
    if prune_method == 'prune_by_class' or prune_method == 'both':
        if n_jobs > 1:  # parallelize
            with multiprocessing_context(
                    n_jobs,
                    initializer=_init,
                    initargs=(_s, _s_counts, _prune_count_matrix,
                              prune_count_matrix.shape, _psx, psx.shape,
                              multi_label),
            ) as p:
                if verbose:
                    print('Parallel processing label errors by class.')
                sys.stdout.flush()
                if big_dataset and tqdm_exists:
                    noise_masks_per_class = list(
                        tqdm.tqdm(p.imap(_prune_by_class, range(K)), total=K),
                    )
                else:
                    noise_masks_per_class = p.map(_prune_by_class, range(K))
        else:  # n_jobs = 1, so no parallelization
            noise_masks_per_class = [_prune_by_class(k, args) for k in range(K)]
        label_errors_mask = np.stack(noise_masks_per_class).any(axis=0)

    if prune_method == 'both':
        label_errors_mask_by_class = label_errors_mask

    if prune_method == 'prune_by_noise_rate' or prune_method == 'both':
        if n_jobs > 1:  # parallelize
            with multiprocessing_context(
                    n_jobs,
                    initializer=_init,
                    initargs=(_s, _s_counts, _prune_count_matrix,
                              prune_count_matrix.shape, _psx, psx.shape,
                              multi_label),
            ) as p:
                if verbose:
                    print('Parallel processing label errors by noise rate.')
                sys.stdout.flush()
                if big_dataset and tqdm_exists:
                    noise_masks_per_class = list(
                        tqdm.tqdm(p.imap(_prune_by_count, range(K)), total=K)
                    )
                else:
                    noise_masks_per_class = p.map(_prune_by_count, range(K))
        else:  # n_jobs = 1, so no parallelization
            noise_masks_per_class = [_prune_by_count(k, args) for k in range(K)]
        label_errors_mask = np.stack(noise_masks_per_class).any(axis=0)

    if prune_method == 'both':
        label_errors_mask = label_errors_mask & label_errors_mask_by_class

    # Remove label errors if given label == model prediction
    if multi_label:
        pred = multiclass_crossval_predict(psx, s)
        s = MultiLabelBinarizer().fit_transform(s)
    else:
        pred = psx.argmax(axis=1)
    for i, pred_label in enumerate(pred):
        if multi_label and np.all(pred_label == s[i]) or \
                not multi_label and pred_label == s[i]:
            label_errors_mask[i] = False

    if sorted_index_method is not None:
        er = order_label_errors(label_errors_mask, psx, s, sorted_index_method)
        return er

    return label_errors_mask


def keep_at_least_n_per_class(prune_count_matrix, n, frac_noise=1.0):
    """Make sure every class has at least n examples after removing noise.
    Functionally, increase each column, increases the diagonal term #(y=k,s=k)
    of prune_count_matrix until it is at least n, distributing the amount
    increased by subtracting uniformly from the rest of the terms in the
    column. When frac_of_noise = 1.0, return all "confidently" estimated
    noise indices, otherwise this returns frac_of_noise fraction of all
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
        When frac_of_noise = 1.0, return all estimated noise indices.
        Value in range (0, 1] that determines the fraction of noisy example
        indices to return based on the following formula for example class k.
        frac_of_noise * number_of_mislabeled_examples_in_class_k, or
        frac_of_noise * inverse_noise_rate_class_k * num_examples_s_equal_k

    Output
    ------

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
        np.count_nonzero(prune_count_matrix, axis=0) - 1.,
        1.,
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
        When frac_of_noise = 1.0, return all estimated noise indices.
        Value in range (0, 1] that determines the fraction of noisy example
        indices to return based on the following formula for example class k.
        frac_of_noise * number_of_mislabeled_examples_in_class_k, or
        frac_of_noise * inverse_noise_rate_class_k * num_examples_s_equal_k."""

    new_mat = prune_count_matrix * frac_noise
    np.fill_diagonal(new_mat, prune_count_matrix.diagonal())
    np.fill_diagonal(new_mat, prune_count_matrix.diagonal() +
                     np.sum(prune_count_matrix - new_mat, axis=0))

    # These are counts, so return a matrix of ints.
    return new_mat.astype(int)


def order_label_errors(
        label_errors_bool,
        psx,
        labels,
        sorted_index_method='normalized_margin',
):
    """Sorts label errors by normalized margin.
    See https://arxiv.org/pdf/1810.05369.pdf (eqn 2.2)
    eg. normalized_margin = prob_label - max_prob_not_label

    Parameters
    ----------
    label_errors_bool : np.array (bool)
      Contains True if the index of labels is an error, o.w. false

    psx : np.array (shape (N, K))
      P(s=k|x) is a matrix with K probabilities for all N examples x.
      This is the probability distribution over all K classes, for each
      example, regarding whether the example has label s==k P(s=k|x). psx
      should computed using 3 (or higher) fold cross-validation.

    labels : np.array
      A binary vector of labels, which may contain label errors.

    sorted_index_method : str ['normalized_margin', 'prob_given_label']
      Method to order label error indices (instead of a bool mask), either:
        'normalized_margin' := normalized margin (p(s = k) - max(p(s != k)))
        'prob_given_label' := [psx[i][labels[i]] for i in label_errors_idx]

    Returns
    -------
      label_errors_idx : np.array (int)
        Return the index integers of the label errors, ordered by
        the normalized margin."""

    # Convert bool mask to index mask
    label_errors_idx = np.arange(len(labels))[label_errors_bool]
    # self confidence is the holdout probability that an example
    # belongs to its given class label
    self_confidence = np.array(
        [np.mean(psx[i][labels[i]]) for i in label_errors_idx]
    )
    if sorted_index_method == 'prob_given_label':
        return label_errors_idx[np.argsort(self_confidence)]
    else:  # sorted_index_method == 'normalized_margin'
        margin = self_confidence - psx[label_errors_bool].max(axis=1)
        return label_errors_idx[np.argsort(margin)]
