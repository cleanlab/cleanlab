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


# ## Latent Estimation
# 
# Contains methods for estimating latent structures used for confident learning.
# * The latent prior of the unobserved, error-less labels $y$:
#     denoted $p(y)$ (latex) & '```py```' (code).
# * The latent noisy channel (noise matrix) characterizing the flipping rates:
#     denoted $P_{labels \vert y }$ (latex) & '```nm```' (code).
# * The latent inverse noise matrix characterizing flipping process:
#     denoted $P_{y \vert labels}$ (latex) & '```inv```' (code).
# * The latent ```confident_joint```, an un-normalized counts matrix of
#     counting a confident subset of the joint counts of label errors.


from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import numpy as np
import copy

from cleanlab.utils.util import (
    value_counts, clip_values, clip_noise_rates, round_preserving_row_totals,
    assert_inputs_are_valid,
)
from cleanlab.utils.latent_algebra import (
    compute_inv_noise_matrix, compute_py, compute_noise_matrix_from_inverse
)


def num_label_issues(
        labels,
        pred_probs,
        confident_joint=None,
):
    """Estimates the number of label issues in `labels`.

    This method is **more accurate** than `sum(find_label_issues())` because its computed using only
    the Trace(joint), ignoring all off-diagonals (used by `find_label_issues` and harder to
    estimate). Here we sum over only diagonal elements in the joint (which have more data
    are more constrained, and therefore easier to compute).

    tl;dr - Use this method to get the most accurate estimate of number of label issues when you
    don't need the indices of the label issues. This is the canonical way to find errors
    simply by combining a ranking/scoring function from `rank.py` with `num_label_issues()`.

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

    confident_joint : np.array (shape (K, K), type int)
        A K,K integer matrix of count(label=k, true_label=k). Estimates a confident subset of
        the joint distribution of the noisy and true labels P_{labels,y}.
        Each entry in the matrix contains the number of examples confidently
        counted into every pair (label=j, true_label=k) classes.

    Returns
    -------
        An integer estimating the number of label issues."""

    if confident_joint is None:
        confident_joint = compute_confident_joint(labels=labels, pred_probs=pred_probs)
    # Normalize confident joint so that it estimates the joint, p(labels,y)
    joint = confident_joint / float(np.sum(confident_joint))
    frac_issues = 1. - joint.trace()
    num_issues = int(frac_issues * len(labels))

    return num_issues


def calibrate_confident_joint(confident_joint, labels, multi_label=False):
    """Calibrates any confident joint estimate P(label=i, true_label=j) such that
    np.sum(cj) == len(labels) and np.sum(cj, axis = 1) == np.bincount(labels).

    In other words, this function forces the confident joint to have the
    true noisy prior p(labels) (summed over columns for each row) and also
    forces the confident joint to add up to the total number of examples.

    This method makes the confident joint a valid counts estimate
    of the actual joint of noisy and true labels.

    Parameters
    ----------

    confident_joint : np.array (shape (K, K))
        A K,K integer matrix of count(label=k, true_label=k). Estimates a confident subset of
        the joint distribution of the noisy and true labels P_{labels,y}.
        Each entry in the matrix contains the number of examples confidently
        counted into every pair (label=j, true_label=k) classes.

    labels : np.array
        A discrete vector of noisy labels, i.e. some labels may be erroneous.
        *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    multi_label : bool
        If true, labels should be an iterable (e.g. list) of iterables, containing a
        list of labels for each example, instead of just a single label.
        The MAJOR DIFFERENCE in how this is calibrated versus single_label, is the total number of
        errors considered is based on the number of labels, not the number of examples. So, the
        calibrated confident_joint will sum to the number of total labels.
        The multi-label setting supports classification tasks where an example has 1 or more labels.
        Example of a multi-labeled `labels` input: [[0,1], [1], [0,2], [0,1,2], [0], [1], ...]


    Returns
    -------
        An np.array of shape (K, K) of type float representing a valid
        estimate of the joint COUNTS of noisy and true labels.
    """

    if multi_label:
        label_counts = value_counts([x for lst in labels for x in lst])
    else:
        label_counts = value_counts(labels)
    # Calibrate confident joint to have correct p(labels) prior on noisy labels.
    calibrated_cj = (
            confident_joint.T / confident_joint.sum(axis=1) * label_counts
    ).T
    # Calibrate confident joint to sum to:
    # The number of examples (for single labeled datasets)
    # The number of total labels (for multi-labeled datasets)
    calibrated_cj = calibrated_cj / np.sum(calibrated_cj) * sum(label_counts)
    return round_preserving_row_totals(calibrated_cj)


def estimate_joint(labels, pred_probs=None, confident_joint=None, multi_label=False):
    """Estimates the joint distribution of label noise P(label=i, true_label=j) guaranteed to
      * sum to 1
      * np.sum(joint_estimate, axis = 1) == p(labels)

    Parameters
    ----------
    See cleanlab.count.calibrate_confident_joint docstring.

    Returns
    -------
        An np.array of shape (K, K) of type float representing a valid
        estimate of the true joint of noisy and true labels.
    """

    if confident_joint is None:
        calibrated_cj = compute_confident_joint(
            labels,
            pred_probs,
            calibrate=True,
            multi_label=multi_label,
        )
    else:
        calibrated_cj = calibrate_confident_joint(confident_joint, labels, multi_label)

    return calibrated_cj / float(np.sum(calibrated_cj))


def _compute_confident_joint_multi_label(
        labels,
        pred_probs,
        thresholds=None,
        calibrate=True,
        return_indices_of_off_diagonals=False,
):
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
        These are multiclass labels. Each list in the list contains
        all the labels for that example. This method will fail if labels
        is not a list of lists (or a list of np.arrays or iterable).

    pred_probs : np.array (shape (N, K))
        P(label=k|x) is a matrix with K model-predicted probabilities.
        Each row of this matrix corresponds to an example `x` and contains the model-predicted
        probabilities that `x` belongs to each possible class.
        The columns must be ordered such that these probabilities correspond to class 0,1,2,...
        `pred_probs` should have been computed using 3 (or higher) fold cross-validation.

    thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
        P(label^=k|label=k). If an example has a predicted probability "greater" than
        this threshold, it is counted as having hidden true_label = k. This is
        not used for filtering/pruning, only for estimating the noise rates using
        confident counts. This value should be between 0 and 1. Default is None.

    calibrate : bool (default: True)
        Calibrates confident joint estimate P(label=i, true_label=j) such that
        np.sum(cj) == len(labels) and np.sum(cj, axis = 1) == np.bincount(labels).

    return_indices_of_off_diagonals: bool
        If true returns indices of examples that were counted in off-diagonals
        of confident joint as a baseline proxy for the label issues. This
        sometimes works as well as filter.find_label_issues(confident_joint)."""

    # Compute unique number of classes K by flattening labels (list of lists)
    K = len(np.unique([i for lst in labels for i in lst]))
    # Compute thresholds = p(label=k | k in set of given labels)
    k_in_l = np.array([[k in lst for lst in labels] for k in range(K)])
    if thresholds is None:
        # the avg probability of class given that the label is represented.
        thresholds = [np.mean(pred_probs[:, k][k_in_l[k]]) for k in range(K)]
    # Create mask for every example if for each class, prob >= threshold
    pred_probs_bool = pred_probs >= thresholds

    # Compute confident joint
    # (no need to avoid collisions for multi-label, double counting is okay!)
    confident_joint = np.array([pred_probs_bool[k_in_l[k]].sum(axis=0) for k in range(K)])
    if calibrate:
        confident_joint = calibrate_confident_joint(confident_joint, labels, multi_label=True)
    if return_indices_of_off_diagonals:
        # Todo add more tests and consider refactoring for efficiency (combine with confident_joint)
        # Convert boolean mask of k_in_l to indices of examples that include class k in the labels
        indices_k_in_l = [np.arange(len(labels))[k_in_l[k]] for k in range(K)]
        # Find boolean mask where the thresholds exceed for non-given class, for each class k
        issues_mask = [np.any(np.delete(pred_probs_bool, k, axis=1)[k_in_l[k]], axis=1) for k in range(K)]
        # Map boolean mask to the indices of the examples, for each class k
        indices_of_issues = [indices_k_in_l[k][issues_mask[k]] for k in range(K)]
        # Combine error indices for each class k into a single set of all the indices of errors
        indices_of_issues = np.asarray(list(set([z for lst in indices_of_issues for z in lst])))
        return confident_joint, sorted(indices_of_issues)

    return confident_joint


def compute_confident_joint(
        labels,
        pred_probs,
        thresholds=None,
        calibrate=True,
        multi_label=False,
        return_indices_of_off_diagonals=False,
):
    """Estimates P(labels,y), the confident counts of the latent
    joint distribution of true and noisy labels
    using observed labels and predicted probabilities pred_probs.

    This estimate is called the confident joint.

    When calibrate = True, this method returns an estimate of
    the latent true joint counts of noisy and true labels.

    Important! This function assumes that pred_probs are out-of-sample
    holdout probabilities. This can be done with cross validation. If
    the probabilities are not computed out-of-sample, overfitting may occur.

    This function estimates the joint of shape (K, K). This is the
    confident counts of examples in every class, labeled as every other class.

    Under certain conditions, estimates are exact, and in most
    conditions, the estimate is within 1 percent of the truth.

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

    K : int (default: None)
        Number of unique classes. Calculated as len(np.unique(labels)) when K == None

    thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
        P(label^=k|label=k). If an example has a predicted probability "greater" than
        this threshold, it is counted as having hidden true_label = k. This is
        not used for filtering/pruning, only for estimating the noise rates using
        confident counts. This value should be between 0 and 1. Default is None.

    calibrate : bool (default: True)
        Calibrates confident joint estimate P(label=i, true_label=j) such that
        np.sum(cj) == len(labels) and np.sum(cj, axis = 1) == np.bincount(labels).

    multi_label : bool
        If true, labels should be an iterable (e.g. list) of iterables, containing a
        list of labels for each example, instead of just a single label.
        The multi-label setting supports classification tasks where an example has 1 or more labels.
        Example of a multi-labeled `labels` input: [[0,1], [1], [0,2], [0,1,2], [0], [1], ...]

    return_indices_of_off_diagonals: bool
        If true returns indices of examples that were counted in off-diagonals
        of confident joint as a baseline proxy for the label issues. This
        sometimes works as well as filter.find_label_issues(confident_joint).


    Examples
    --------

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
        return _compute_confident_joint_multi_label(
            labels=labels,
            pred_probs=pred_probs,
            thresholds=thresholds,
            calibrate=calibrate,
            return_indices_of_off_diagonals=return_indices_of_off_diagonals,
        )

    # labels needs to be a numpy array
    labels = np.asarray(labels)

    # Find the number of unique classes
    num_classes = len(np.unique(labels))

    # Estimate the probability thresholds for confident counting
    if thresholds is None:
        # P(we predict the given noisy label is k | given noisy label is k)
        thresholds = [np.mean(pred_probs[:, k][labels == k]) for k in range(num_classes)]
    thresholds = np.asarray(thresholds)

    # Compute confident joint (vectorized for speed).

    # pred_probs_bool is a bool matrix where each row represents a training example as a boolean vector of
    # size num_classes, with True if the example confidently belongs to that class and False if not.
    pred_probs_bool = (pred_probs >= thresholds - 1e-6)
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
    confident_joint = confusion_matrix(true_labels_confident, labels_confident).T
    if calibrate:
        confident_joint = calibrate_confident_joint(confident_joint, labels)

    if return_indices_of_off_diagonals:
        true_labels_neq_given_labels = true_labels_confident != labels_confident
        indices = np.arange(len(labels))[at_least_one_confident][true_labels_neq_given_labels]

        return confident_joint, indices

    return confident_joint


def estimate_latent(
        confident_joint,
        labels,
        py_method='cnt',
        converge_latent_estimates=False,
):
    """Computes the latent prior p(y), the noise matrix P(labels|y) and the
    inverse noise matrix P(y|labels) from the `confident_joint` count(labels, y). The
    `confident_joint` estimated by `compute_confident_joint`
    by counting confident examples.

    Parameters
    ----------

    labels : np.array
        A discrete vector of noisy labels, i.e. some labels may be erroneous.
        *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    confident_joint : np.array (shape (K, K), type int)
        A K,K integer matrix of count(label=k, true_label=k). Estimates a confident subset
        of the joint distribution of the noisy and true labels P_{labels,y}.
        Each entry in the matrix contains the number of examples confidently
        counted into every pair (label=j, true_label=k) classes.

    py_method : str (Options: ["cnt", "eqn", "marginal", "marginal_ps"])
        `py` is shorthand for the `class proportions (a.k.a prior) of the true labels`
        This method defines how to compute the latent prior p(true_label=k). Default is "cnt".
        "cnt" works well even when the noise matrices are estimated poorly by using
        the matrix diagonals instead of all the probabilities.

    converge_latent_estimates : bool
      If true, forces numerical consistency of estimates. Each is estimated
      independently, but they are related mathematically with closed form
      equivalences. This will iteratively make them mathematically consistent.

    Returns
    ------
        A tuple containing (py, noise_matrix, inv_noise_matrix)."""

    # 'ps' is p(labels=k)
    ps = value_counts(labels) / float(len(labels))
    # Number of training examples confidently counted from each noisy class
    labels_class_counts = confident_joint.sum(axis=1).astype(float)
    # Number of training examples confidently counted into each true class
    true_labels_class_counts = confident_joint.sum(axis=0).astype(float)
    # Confident Counts Estimator: p(label=k_s|true_label=k_y) ~ |label=k_s and true_label=k_y| / |true_label=k_y|
    noise_matrix = confident_joint / true_labels_class_counts
    # Confident Counts Estimator: p(true_label=k_y|label=k_s) ~ |true_label=k_y and label=k_s| / |label=k_s|
    inv_noise_matrix = confident_joint.T / labels_class_counts
    # Compute the prior p(y), the latent (uncorrupted) class distribution.
    py = compute_py(ps, noise_matrix, inv_noise_matrix, py_method, true_labels_class_counts)
    # Clip noise rates to be valid probabilities.
    noise_matrix = clip_noise_rates(noise_matrix)
    inv_noise_matrix = clip_noise_rates(inv_noise_matrix)
    # Make latent estimates mathematically agree in their algebraic relations.
    if converge_latent_estimates:
        py, noise_matrix, inv_noise_matrix = converge_estimates(
            ps, py, noise_matrix, inv_noise_matrix)
        # Again clip py and noise rates into proper range [0,1)
        py = clip_values(py, low=1e-5, high=1.0, new_sum=1.0)
        noise_matrix = clip_noise_rates(noise_matrix)
        inv_noise_matrix = clip_noise_rates(inv_noise_matrix)

    return py, noise_matrix, inv_noise_matrix


def estimate_py_and_noise_matrices_from_probabilities(
        labels,
        pred_probs,
        thresholds=None,
        converge_latent_estimates=True,
        py_method='cnt',
        calibrate=True,
):
    """Computes the confident counts
    estimate of latent variables py and the noise rates
    using observed labels and predicted probabilities, pred_probs.

    Important! This function assumes that pred_probs are out-of-sample
    holdout probabilities. This can be done with cross validation. If
    the probabilities are not computed out-of-sample, overfitting may occur.

    This function estimates the noise_matrix of shape (K, K). This is the
    fraction of examples in every class, labeled as every other class. The
    noise_matrix is a conditional probability matrix for P(label=k_s|true_label=k_y).

    Under certain conditions, estimates are exact, and in most
    conditions, estimates are within one percent of the actual noise rates.

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

    thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
        P(label^=k|label=k). If an example has a predicted probability "greater" than
        this threshold, it is counted as having hidden true_label = k. This is
        not used for filtering/pruning, only for estimating the noise rates using
        confident counts. This value should be between 0 and 1. Default is None.

    converge_latent_estimates : bool
        If true, forces numerical consistency of estimates. Each is estimated
        independently, but they are related mathematically with closed form
        equivalences. This will iteratively make them mathematically consistent.

    py_method : str (Options: ["cnt", "eqn", "marginal", "marginal_ps"])
        How to compute the latent prior p(true_label=k). Default is "cnt" as it often
        works well even when the noise matrices are estimated poorly by using
        the matrix diagonals instead of all the probabilities.

    calibrate : bool (default: True)
        Calibrates confident joint estimate P(label=i, true_label=j) such that
        np.sum(cj) == len(labels) and np.sum(cj, axis = 1) == np.bincount(labels).

    Returns
    ------
        py, noise_matrix, inverse_noise_matrix"""

    confident_joint = compute_confident_joint(
        labels=labels,
        pred_probs=pred_probs,
        thresholds=thresholds,
        calibrate=calibrate,
    )
    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        labels=labels,
        py_method=py_method,
        converge_latent_estimates=converge_latent_estimates,
    )

    return py, noise_matrix, inv_noise_matrix, confident_joint


def estimate_confident_joint_and_cv_pred_proba(
        X,
        labels,
        clf=LogReg(multi_class='auto', solver='lbfgs'),
        cv_n_folds=5,
        thresholds=None,
        seed=None,
        calibrate=True,
):
    """Estimates P(labels,y), the confident counts of the latent
    joint distribution of true and noisy labels
    using observed labels and predicted probabilities pred_probs.

    The output of this function is a numpy array of shape (K, K).

    Under certain conditions, estimates are exact, and in many
    conditions, estimates are within one percent of actual.

    Notes: There are two ways to compute the confident joint with pros/cons.
    1. For each holdout set, we compute the confident joint, then sum them up.
    2. Compute pred_proba for each fold, combine, compute the confident joint.
    (1) is more accurate because it correctly computes thresholds for each fold
    (2) is more accurate when you have only a little data because it computes
    the confident joint using all the probabilities. For example if you had 100
    examples, with 5-fold cross validation + uniform p(y) you would only have 20
    examples to compute each confident joint for (1). Such small amounts of data
    is bound to result in estimation errors. For this reason, we implement (2),
    but we implement (1) as a commented out function at the end of this file.

    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array

    labels : np.array
          A discrete vector of noisy labels, i.e. some labels may be erroneous.
          *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    clf : sklearn.classifier or equivalent
      Default classifier used is logistic regression. Assumes clf
      has predict_proba() and fit() defined.

    cv_n_folds : int
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.

    thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
      P(label^=k|label=k). If an example has a predicted probability "greater" than
      this threshold, it is counted as having hidden true_label = k. This is
      not used for filtering/pruning, only for estimating the noise rates using
      confident counts. This value should be between 0 and 1. Default is None.

    seed : int (default = None)
        Set the default state of the random number generator used to split
        the cross-validated folds. If None, uses np.random current random state.

    calibrate : bool (default: True)
        Calibrates confident joint estimate P(label=i, true_label=j) such that
        np.sum(cj) == len(labels) and np.sum(cj, axis = 1) == np.bincount(labels).

    Returns
    ------
      Returns a tuple of two numpy array matrices in the form:
      (joint counts matrix, predicted probability matrix)"""

    assert_inputs_are_valid(X, labels)
    # Number of classes
    K = len(np.unique(labels))

    # Ensure labels are of type np.array()
    labels = np.asarray(labels)

    # Create cross-validation object for out-of-sample predicted probabilities.
    # CV folds preserve the fraction of noisy positive and
    # noisy negative examples in each class.
    kf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=seed)

    # Initialize pred_probs array
    pred_probs = np.zeros((len(labels), K))

    # Split X and labels into "cv_n_folds" stratified folds.
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(X, labels)):
        clf_copy = copy.deepcopy(clf)

        # Select the training and holdout cross-validated sets.
        X_train_cv, X_holdout_cv = X[cv_train_idx], X[cv_holdout_idx]
        s_train_cv, s_holdout_cv = labels[cv_train_idx], labels[cv_holdout_idx]

        # Fit the clf classifier to the training set and
        # predict on the holdout set and update pred_probs.
        clf_copy.fit(X_train_cv, s_train_cv)
        pred_probs_cv = clf_copy.predict_proba(X_holdout_cv)  # P(labels = k|x) # [:,1]
        pred_probs[cv_holdout_idx] = pred_probs_cv

    # Compute the confident counts, a K x K matrix for all pairs of labels.
    confident_joint = compute_confident_joint(
        labels=labels,
        pred_probs=pred_probs,  # P(labels = k|x)
        thresholds=thresholds,
        calibrate=calibrate,
    )

    return confident_joint, pred_probs


def estimate_py_noise_matrices_and_cv_pred_proba(
        X,
        labels,
        clf=LogReg(multi_class='auto', solver='lbfgs'),
        cv_n_folds=5,
        thresholds=None,
        converge_latent_estimates=False,
        py_method='cnt',
        seed=None,
):
    """This function computes the out-of-sample predicted
    probability P(label=k|x) for every example x in X using cross
    validation while also computing the confident counts noise
    rates within each cross-validated subset and returning
    the average noise rate across all examples.

    This function estimates the noise_matrix of shape (K, K). This is the
    fraction of examples in every class, labeled as every other class. The
    noise_matrix is a conditional probability matrix for P(label=k_s|true_label=k_y).

    Under certain conditions, estimates are exact, and in most
    conditions, estimates are within one percent of the actual noise rates.

    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array

    labels : np.array
        A discrete vector of noisy labels, i.e. some labels may be erroneous.
        *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    clf : sklearn.classifier or equivalent
      Default classifier used is logistic regression. Assumes clf
      has predict_proba() and fit() defined.

    cv_n_folds : int
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.

    thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
      P(label^=k|label=k). If an example has a predicted probability "greater" than
      this threshold, it is counted as having hidden true_label = k. This is
      not used for filtering/pruning, only for estimating the noise rates using
      confident counts. This value should be between 0 and 1. Default is None.

    converge_latent_estimates : bool
      If true, forces numerical consistency of estimates. Each is estimated
      independently, but they are related mathematically with closed form
      equivalences. This will iteratively make them mathematically consistent.

    py_method : str (Options: ["cnt", "eqn", "marginal", "marginal_ps"])
        How to compute the latent prior p(true_label=k). Default is "cnt" as it often
        works well even when the noise matrices are estimated poorly by using
        the matrix diagonals instead of all the probabilities.

    seed : int (default = None)
        Set the default state of the random number generator used to split
        the cross-validated folds. If None, uses np.random current random state.

    Returns
    ------
      Returns a tuple of five numpy array matrices in the form:
      (py, noise_matrix, inverse_noise_matrix,
      joint count matrix i.e. confident joint, predicted probability matrix)"""

    confident_joint, pred_probs = estimate_confident_joint_and_cv_pred_proba(
        X=X,
        labels=labels,
        clf=clf,
        cv_n_folds=cv_n_folds,
        thresholds=thresholds,
        seed=seed,
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
        labels,  # class labels can be noisy (labels) or not noisy (y).
        clf=LogReg(multi_class='auto', solver='lbfgs'),
        cv_n_folds=5,
        seed=None,
):
    """This function computes the out-of-sample predicted
    probability [P(label=k|x)] for every example in X using cross
    validation. Output is a np.array of shape (N, K) where N is
    the number of training examples and K is the number of classes.

    Parameters
    ----------

    X : np.array
      Input feature matrix (N, D), 2D numpy array

    labels : np.array or list of ints from [0,1,..,K-1]
      A discrete vector of class labels which may or may not contain mislabeling.
      *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    clf : sklearn.classifier or equivalent
      Default classifier used is logistic regression. Assumes clf
      has predict_proba() and fit() defined.

    cv_n_folds : int
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.

    seed : int (default = None)
        Set the default state of the random number generator used to split
        the cross-validated folds. If None, uses np.random current random state.

    Returns
    --------
    pred_probs : np.array (shape (N, K))
        P(label=k|x) is a matrix with K model-predicted probabilities.
        Each row of this matrix corresponds to an example `x` and contains the model-predicted
        probabilities that `x` belongs to each possible class.
        The columns must be ordered such that these probabilities correspond to class 0,1,2,...
        `pred_probs` should have been computed using 3 (or higher) fold cross-validation."""

    return estimate_py_noise_matrices_and_cv_pred_proba(
        X=X,
        labels=labels,
        clf=clf,
        cv_n_folds=cv_n_folds,
        seed=seed,
    )[-1]


def estimate_noise_matrices(
        X,
        labels,
        clf=LogReg(multi_class='auto', solver='lbfgs'),
        cv_n_folds=5,
        thresholds=None,
        converge_latent_estimates=True,
        seed=None,
):
    """Estimates the noise_matrix of shape (K, K). This is the
    fraction of examples in every class, labeled as every other class. The
    noise_matrix is a conditional probability matrix for P(label=k_s|true_label=k_y).

    Under certain conditions, estimates are exact, and in most
    conditions, estimates are within one percent of the actual noise rates.

    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array

    labels : np.array
        A discrete vector of noisy labels, i.e. some labels may be erroneous.
        *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    clf : sklearn.classifier or equivalent
      Default classifier used is logistic regression. Assumes clf
      has predict_proba() and fit() defined.

    cv_n_folds : int
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.

    thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
      P(label^=k|label). If an example has a predicted probability "greater" than
      this threshold, it is counted as having hidden true_label = k. This is
      not used for filtering/pruning, only for estimating the noise rates using
      confident counts. This value should be between 0 and 1. Default is None.

    converge_latent_estimates : bool
      If true, forces numerical consistency of estimates. Each is estimated
      independently, but they are related mathematically with closed form
      equivalences. This will iteratively make them mathematically consistent.

    seed : int (default = None)
        Set the default state of the random number generator used to split
        the cross-validated folds. If None, uses np.random current random state.

    Returns
    ------
        A two-item tuple containing (noise_matrix, inv_noise_matrix)."""

    return estimate_py_noise_matrices_and_cv_pred_proba(
        X=X,
        labels=labels,
        clf=clf,
        cv_n_folds=cv_n_folds,
        thresholds=thresholds,
        converge_latent_estimates=converge_latent_estimates,
        seed=seed,
    )[1:-2]


def converge_estimates(
        ps,
        py,
        noise_matrix,
        inverse_noise_matrix,
        inv_noise_matrix_iterations=5,
        noise_matrix_iterations=3,
):
    """Computes py := P(true_label=k) and both noise_matrix and inverse_noise_matrix,
    by numerically converging ps := P(labels=k), py, and the noise matrices.

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

    ps : np.array (shape (K, ) or (1, K))
        The fraction (prior probability) of each observed, NOISY class P(labels = k).

    py : np.array (shape (K, ) or (1, K))
        The estimated fraction (prior probability) of each TRUE class P(true_label = k).

    noise_matrix : np.array of shape (K, K), K = number of classes
        A conditional probability matrix of the form P(label=k_s|true_label=k_y) containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1.

    inverse_noise_matrix : np.array of shape (K, K), K = number of classes
        A conditional probability matrix of the form P(true_label=k_y|labels=k_s) representing
        the estimated fraction observed examples in each class k_s, that are
        mislabeled examples from every other class k_y. If None, the
        inverse_noise_matrix will be computed from pred_probs and labels.
        Assumes columns of inverse_noise_matrix sum to 1.

    inv_noise_matrix_iterations : int (Default: 5)
        Number of times to converge inverse noise matrix with py and noise mat.

    noise_matrix_iterations : int (Default: 3)
        Number of times to converge noise matrix with py and inverse noise mat.

    Returns
    ------
        Three np.arrays of the form (py, noise_matrix, inverse_noise_matrix) all
        having numerical agreement in terms of their mathematical relations."""

    for j in range(noise_matrix_iterations):
        for i in range(inv_noise_matrix_iterations):
            inverse_noise_matrix = compute_inv_noise_matrix(
                py, noise_matrix, ps)
            py = compute_py(ps, noise_matrix, inverse_noise_matrix)
        noise_matrix = compute_noise_matrix_from_inverse(
            ps, inverse_noise_matrix, py)

    return py, noise_matrix, inverse_noise_matrix
