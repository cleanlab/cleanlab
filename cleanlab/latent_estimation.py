
# coding: utf-8

# ## Latent Estimation
# 
# #### Contains methods for estimating four latent structures used for confident learning.
# * The latent prior of the unobserved, errorless labels $y$: denoted $p(y)$ (latex) & '```py```' (code).
# * The latent noisy channel (noise matrix) characterizing the flipping rates: denoted $P_{s \vert y }$ (latex) & '```nm```' (code).
# * The latent inverse noise matrix characterizing flipping process: denoted $P_{y \vert s}$ (latex) & '```inv```' (code).
# * The latent ```confident_joint```, an unnormalized counts matrix of counting a confident subset of the joint counts of label errors.


from __future__ import (
    print_function, absolute_import, division, unicode_literals, with_statement
)
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
import numpy as np
import copy
import warnings

from cleanlab.util import (
    value_counts, clip_values, clip_noise_rates, round_preserving_row_totals
)
from cleanlab.latent_algebra import (
    compute_inv_noise_matrix, compute_py, compute_noise_matrix_from_inverse
)


def num_label_errors(
    labels,
    psx,
    confident_joint=None,
):
    '''Estimates the number of label errors in `labels`.

    Parameters
    ----------

    labels : np.array
        A discrete vector of noisy labels, i.e. some labels may be erroneous.

    psx : np.array (shape (N, K))
        P(label=k|x) is a matrix with K (noisy) probabilities for each of the N examples x.
        This is the probability distribution over all K classes, for each
        example, regarding whether the example has label s==k P(s=k|x). psx should
        have been computed using 3 (or higher) fold cross-validation.

    confident_joint : np.array (shape (K, K), type int)
        A K,K integer matrix of count(s=k, y=k). Estimates a confident subset of
        the joint disribution of the noisy and true labels P_{s,y}.
        Each entry in the matrix contains the number of examples confidently
        counted into every pair (s=j, y=k) classes.

    Returns
    -------
        An integer estimating the number of label errors.'''

    if confident_joint is None:
        confident_joint = compute_confident_joint(s=labels, psx=psx)
    # Normalize confident joint so that it estimates the joint, p(s,y)
    joint = confident_joint / float(np.sum(confident_joint))
    frac_errors = 1. - joint.trace()
    num_errors = int(frac_errors * len(labels))

    return num_errors


def calibrate_confident_joint(confident_joint, s, multi_label=False):
    '''Calibrates any confident joint estimate P(s=i, y=j) such that
    np.sum(cj) == len(s) and np.sum(cj, axis = 1) == np.bincount(s).

    In other words, this function forces the confident joint to have the
    true noisy prior p(s) (summed over columns for each row) and also
    forces the confident joint to add up to the total number of examples.

    This method makes the confident joint a valid counts estimate
    of the actual joint of noisy and true labels.

    Parameters
    ----------

    confident_joint : np.array (shape (K, K))
        A K,K integer matrix of count(s=k, y=k). Estimates a confident subset of
        the joint disribution of the noisy and true labels P_{s,y}.
        Each entry in the matrix contains the number of examples confidently
        counted into every pair (s=j, y=k) classes.

    s : np.array
        A discrete vector of labels, s, which may contain mislabeling. "s" denotes
        the noisy label instead of \tilde(y), for ASCII encoding reasons.

    multi_label : bool
        If true, s should be an iterable (e.g. list) of iterables, containing a
        list of labels for each example, instead of just a single label.
        The MAJOR DIFFERENCE in how this is calibrated versus single_label,
        is the total number of errors considered is based on the number
        of labels, not the number of examples. So, the calibrated
        confident_joint will sum to the number of total labels.


    Returns
    -------
        An np.array of shape (K, K) of type float representing a valid
        estimate of the joint COUNTS of noisy and true labels.
    '''

    if multi_label:
        s_counts = value_counts([x for l in s for x in l])
    else:
        s_counts = value_counts(s)
    # Calibrate confident joint to have correct p(s) prior on noisy labels.
    calibrated_cj = (confident_joint.T / confident_joint.sum(axis=1) * s_counts).T
    # Calibrate confident joint to sum to:
    # The number of examples (for single labeled datasets)
    # The number of total labels (for multi-labeled datasets)
    calibrated_cj = calibrated_cj / np.sum(calibrated_cj) * sum(s_counts)
    return round_preserving_row_totals(calibrated_cj)


def estimate_joint(s, psx=None, confident_joint=None, multi_label=False):
    '''Estimates the joint distribution of label noise P(s=i, y=j) guranteed to
      * sum to 1
      * np.sum(joint_estimate, axis = 1) == p(s)

    Parameters
    ----------
    See cleanlab.latent_estimation.calibrate_confident_joint docstring.

    Returns
    -------
        An np.array of shape (K, K) of type float representing a valid
        estimate of the true joint of noisy and true labels.
    '''
    
    if confident_joint is None:
        calibrated_cj = compute_confident_joint(
            s,
            psx,
            calibrate=True,
            multi_label=multi_label,
        )
    else:
        calibrated_cj = calibrate_confident_joint(confident_joint, s)

    return calibrated_cj / float(np.sum(calibrated_cj))


def _compute_confident_joint_multi_label(
    labels,
    psx,
    thresholds=None,
    calibrate=True,
):
    '''Computes the confident joint for multi_labeled data. Thus,
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
        all the labels for that example.

    psx : np.array (shape (N, K))
        P(s=k|x) is a matrix with K (noisy) probabilities for each of the N examples x.
        This is the probability distribution over all K classes, for each
        example, regarding whether the example has label s==k P(s=k|x). psx should
        have been computed using 3 (or higher) fold cross-validation.

    thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
        P(s^=k|s=k). If an example has a predicted probability "greater" than
        this threshold, it is counted as having hidden label y = k. This is
        not used for pruning, only for estimating the noise rates using
        confident counts. This value should be between 0 and 1. Default is None.

    calibrate : bool (default: True)
        Calibrates confident joint estimate P(s=i, y=j) such that
        np.sum(cj) == len(s) and np.sum(cj, axis = 1) == np.bincount(s).'''

    K = len(np.unique(labels))
    # Compute thresholds = p(s=k | k in set of given labels)
    # This is the average probability of class given that the label is represented.
    k_in_l = np.array([[k in l for l in labels] for k in range(K)])
    thresholds = [np.mean(psx[:,k][k_in_l[k]]) for k in range(K)]
    # Create mask for every example if for each class, prob >= threshold
    psx_bool = psx >= thresholds
    # Compute confident joint
    # (no need to avoid collisions for multi-label, double counting is okay!)
    confident_joint = np.array([psx_bool[k_in_l[k]].sum(axis = 0) for k in range(K)])
    if calibrate:
        return calibrate_confident_joint(confident_joint, labels, multi_label=True)

    return confident_joint


def compute_confident_joint(
    s,
    psx,
    K=None,
    thresholds=None,
    calibrate=True,
    multi_label=False,
    return_indices_of_off_diagonals=False,
):
    '''Estimates P(s,y), the confident counts of the latent
    joint distribution of true and noisy labels
    using observed s and predicted probabilities psx.

    This estimate is called the confident joint.

    When calibrate = True, this method returns an estimate of
    the latent true joint counts of noisy and true labels.

    Important! This function assumes that psx are out-of-sample
    holdout probabilities. This can be done with cross validation. If
    the probabilities are not computed out-of-sample, overfitting may occur.

    This function estimates the joint of shape (K, K). This is the
    confident counts of examples in every class, labeled as every other class.

    Under certain conditions, estimates are exact, and in most
    conditions, the estimate is within 1 percent of the truth.

    We provide a for-loop based simplification of the confident joint
    below. This implementation is not efficient, not used in practice, and
    not complete, but covers the jist of how the confident joint is computed:

    # Confident examples are those that we are confident have label y = k
    # Estimate the (K, K) matrix of confident examples with s = k_s and y = k_y
    cj_ish = np.zeros((K, K))
    for k_s in range(K): # k_s is the class value k of noisy label s
        for k_y in range(K): # k_y is the (guessed) class value k of true label y
            cj_ish[k_s][k_y] = sum((psx[:,k_y] >= (thresholds[k_y] - 1e-8)) & (s == k_s))

    Parameters
    ----------

    s : np.array
        A discrete vector of labels, s, which may contain mislabeling. "s" denotes
        the noisy label instead of \tilde(y), for ASCII encoding reasons.

    psx : np.array (shape (N, K))
        P(s=k|x) is a matrix with K (noisy) probabilities for each of the N examples x.
        This is the probability distribution over all K classes, for each
        example, regarding whether the example has label s==k P(s=k|x). psx should
        have been computed using 3 (or higher) fold cross-validation.

    K : int (default: None)
        Number of unique classes. Calculated as len(np.unique(s)) when K == None.

    thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
        P(s^=k|s=k). If an example has a predicted probability "greater" than
        this threshold, it is counted as having hidden label y = k. This is
        not used for pruning, only for estimating the noise rates using
        confident counts. This value should be between 0 and 1. Default is None.
        
    calibrate : bool (default: True)
        Calibrates confident joint estimate P(s=i, y=j) such that
        np.sum(cj) == len(s) and np.sum(cj, axis = 1) == np.bincount(s).

    multi_label : bool
        If true, s should be an iterable (e.g. list) of iterables, containing a
        list of labels for each example, instead of just a single label.

    return_indices_of_off_digaonals: bool
        If true returns indices of examples that were counted in off-diagonals of
        the confident joint. These are a baseline proxy for the label errors. This
        is NOT as effective as using pruning.get_noise_indices(confident_joint)
        because (1) this method is based on probabilities which may be erroneous,
        and (2) this simple approach does not normalize or take into account priors.
        '''

    if multi_label:
        return _compute_confident_joint_multi_label(
            labels=s,
            psx=psx,
            thresholds=thresholds,
            calibrate=calibrate,
        )

    # s needs to be a numpy array
    s = np.asarray(s)

    # Find the number of unique classes if K is not given
    if K is None:
        K = len(np.unique(s))

    # Estimate the probability thresholds for confident counting
    if thresholds is None:
        thresholds = [np.mean(psx[:,k][s == k]) for k in range(K)] # P(s^=k|s=k)
    thresholds = np.asarray(thresholds)

    # The following code computes the confident joint.
    # The code is optimized with vectorized functions.
    # For ease of understanding, here is (a slow) implementation with for loops.
    #     confident_joint = np.zeros((K, K), dtype = int)
    #     for i, row in enumerate(psx):
    #         s_label = s[i]
    #         confident_bins = row >= thresholds - 1e-6
    #         num_confident_bins = sum(confident_bins)
    #         if num_confident_bins == 1:
    #             confident_joint[s_label][np.argmax(confident_bins)] += 1
    #         elif num_confident_bins > 1:
    #             confident_joint[s_label][np.argmax(row)] += 1

    # Compute confident joint (vectorized for speed).

    # psx_bool is a bool matrix where each row represents a training example as
    # a boolean vector of size K, with True if the example confidentally belongs
    # to that class and False if not.
    psx_bool = (psx >= thresholds - 1e-6)
    num_confident_bins = psx_bool.sum(axis = 1)
    at_least_one_confident = num_confident_bins > 0
    more_than_one_confident = num_confident_bins > 1
    psx_argmax = psx.argmax(axis=1)
    # Note that confident_argmax is meaningless for rows of all False
    confident_argmax = psx_bool.argmax(axis=1)
    # For each example, choose the confident class (greater than threshold)
    # When there is more than one confident class, choose the class with largest prob.
    true_label_guess = np.where(more_than_one_confident, psx_argmax, confident_argmax)
    y_confident = true_label_guess[at_least_one_confident] # Omits meaningless all-False rows
    s_confident = s[at_least_one_confident]
    confident_joint = confusion_matrix(y_confident, s_confident).T
    
    if calibrate:
        confident_joint = calibrate_confident_joint(confident_joint, s)

    if return_indices_of_off_diagonals:
        indices = np.arange(len(s))[at_least_one_confident][y_confident != s_confident]

        return confident_joint, indices

    return confident_joint


def estimate_latent(
    confident_joint,
    s,
    py_method='cnt',
    converge_latent_estimates=False,
):
    '''Computes the latent prior p(y), the noise matrix P(s|y) and the
    inverse noise matrix P(y|s) from the `confident_joint` count(s, y). The
    `confident_joint` estimated by `compute_confident_joint`
    by counting confident examples.

    Parameters
    ----------

    s : np.array
        A discrete vector of labels, s, which may contain mislabeling. "s" denotes
        the noisy label instead of \tilde(y), for ASCII encoding reasons.

    confident_joint : np.array (shape (K, K), type int)
        A K,K integer matrix of count(s=k, y=k). Estimatesa a confident subset of
        the joint disribution of the noisy and true labels P_{s,y}.
        Each entry in the matrix contains the number of examples confidently
        counted into every pair (s=j, y=k) classes.

    py_method : str (Options: ["cnt", "eqn", "marginal", "marginal_ps"])
        How to compute the latent prior p(y=k). Default is "cnt" as it often
        works well even when the noise matrices are estimated poorly by using
        the matrix diagonals instead of all the probabilities.

    converge_latent_estimates : bool
      If true, forces numerical consistency of estimates. Each is estimated
      independently, but they are related mathematically with closed form
      equivalences. This will iteratively make them mathematically consistent.

    Returns
    ------
        A tuple containing (py, noise_matrix, inv_noise_matrix).'''

    # Number of classes
    K = len(np.unique(s))
    # 'ps' is p(s=k)
    ps = value_counts(s) / float(len(s))
    # Ensure labels are of type np.array()
    s = np.asarray(s)
    # Number of training examples confidently counted from each noisy class
    s_count = confident_joint.sum(axis=1).astype(float)
    # Number of training examples confidently counted into each true class
    y_count = confident_joint.sum(axis=0).astype(float)
    # Confident Counts Estimator for p(s=k_s|y=k_y) ~ |s=k_s and y=k_y| / |y=k_y|
    noise_matrix = confident_joint / y_count
    # Confident Counts Estimator for p(y=k_y|s=k_s) ~ |y=k_y and s=k_s| / |s=k_s|
    inv_noise_matrix = confident_joint.T / s_count
    # Compute the prior p(y), the latent (uncorrupted) class distribution.
    py = compute_py(ps, noise_matrix, inv_noise_matrix, py_method, y_count)
    # Clip noise rates to be valid probabilities.
    noise_matrix = clip_noise_rates(noise_matrix)
    inv_noise_matrix = clip_noise_rates(inv_noise_matrix)
    # Make latent estimates mathematically agree in their algebraic relations.
    if converge_latent_estimates:
        py, noise_matrix, inv_noise_matrix = converge_estimates(ps, py, noise_matrix, inv_noise_matrix)
        # Again clip py and noise rates into proper range [0,1)
        py = clip_values(py, low=1e-5, high=1.0, new_sum=1.0)
        noise_matrix = clip_noise_rates(noise_matrix)
        inv_noise_matrix = clip_noise_rates(inv_noise_matrix)

    return py, noise_matrix, inv_noise_matrix


def estimate_py_and_noise_matrices_from_probabilities(
    s,
    psx,
    thresholds=None,
    converge_latent_estimates=True,
    py_method='cnt',
    calibrate=True,
):
    '''Computes the confident counts
    estimate of latent variables py and the noise rates
    using observed s and predicted probabilities psx.

    Important! This function assumes that psx are out-of-sample
    holdout probabilities. This can be done with cross validation. If
    the probabilities are not computed out-of-sample, overfitting may occur.

    This function estimates the noise_matrix of shape (K, K). This is the
    fraction of examples in every class, labeled as every other class. The
    noise_matrix is a conditional probability matrix for P(s=k_s|y=k_y).

    Under certain conditions, estimates are exact, and in most
    conditions, estimates are within one percent of the actual noise rates.

    Parameters
    ----------

    s : np.array
      A discrete vector of labels, s, which may contain mislabeling. "s" denotes
      the noisy label instead of \tilde(y), for ASCII encoding reasons.

    psx : np.array (shape (N, K))
      P(s=k|x) is a matrix with K (noisy) probabilities for each of the N examples x.
      This is the probability distribution over all K classes, for each
      example, regarding whether the example has label s==k P(s=k|x). psx should
      have been computed using 3 (or higher) fold cross-validation.

    thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
      P(s^=k|s=k). If an example has a predicted probability "greater" than
      this threshold, it is counted as having hidden label y = k. This is
      not used for pruning, only for estimating the noise rates using
      confident counts. This value should be between 0 and 1. Default is None.

    converge_latent_estimates : bool
      If true, forces numerical consistency of estimates. Each is estimated
      independently, but they are related mathematically with closed form
      equivalences. This will iteratively make them mathematically consistent.

    py_method : str
        How to compute the latent prior p(y=k). Default is "cnt" as it tends to
        work best, but you may also set this hyperparameter to "eqn" or "marginal".

    calibrate : bool (default: True)
        Calibrates confident joint estimate P(s=i, y=j) such that
        np.sum(cj) == len(s) and np.sum(cj, axis = 1) == np.bincount(s).

    Returns
    ------
        py, noise_matrix, inverse_noise_matrix'''

    confident_joint = compute_confident_joint(
        s=s,
        psx=psx,
        thresholds=thresholds,
        calibrate=calibrate,
    )
    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        s=s,
        py_method=py_method,
        converge_latent_estimates=converge_latent_estimates,
    )

    return py, noise_matrix, inv_noise_matrix, confident_joint


def estimate_confident_joint_and_cv_pred_proba(
    X,
    s,
    clf=logreg(multi_class='auto', solver='lbfgs'),
    cv_n_folds=5,
    thresholds=None,
    seed=None,
    calibrate=True,
):
    '''Estimates P(s,y), the confident counts of the latent
    joint distribution of true and noisy labels
    using observed s and predicted probabilities psx.

    The output of this function is a numpy array of shape (K, K).

    Under certain conditions, estimates are exact, and in many
    conditions, estimates are within one percent of actual.

    Notes: There are two ways to compute the confident joint with pros/cons.
    1. For each holdout set, we compute the confident joint, then sum them up.
    2. We get all the pred_proba, combine them, compute the confident joint on all.
    (1) is more accurate because it computes the appropriate thresholds for each fold
    (2) is more accurate when you have only a little data because it computes
    the confident joint using all the probabilities. For example if you had only 100
    examples, with 5-fold cross validation and uniform p(y) you would only have 20
    examples to compute each confident joint for (1). Such small amounts of data
    is bound to result in estimation errors. For this reason, we implement (2),
    but we implement (1) as a commented out function at the end of this file.

    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array

    s : np.array
      A discrete vector of labels, s, which may contain mislabeling. "s" denotes
      the noisy label instead of \tilde(y), for ASCII encoding reasons.

    clf : sklearn.classifier or equivalent
      Default classifier used is logistic regression. Assumes clf
      has predict_proba() and fit() defined.

    cv_n_folds : int
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.

    thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
      P(s^=k|s=k). If an example has a predicted probability "greater" than
      this threshold, it is counted as having hidden label y = k. This is
      not used for pruning, only for estimating the noise rates using
      confident counts. This value should be between 0 and 1. Default is None.

    seed : int (default = None)
        Number to set the default state of the random number generator used to split
        the cross-validated folds. If None, uses np.random current random state.
        
    calibrate : bool (default: True)
        Calibrates confident joint estimate P(s=i, y=j) such that
        np.sum(cj) == len(s) and np.sum(cj, axis = 1) == np.bincount(s).

    Returns
    ------
      Returns a tuple of two numpy array matrices in the form:
      (joint counts matrix, predicted probability matrix)'''

    # Number of classes
    K = len(np.unique(s))
    # 'ps' is p(s=k)
    ps = value_counts(s) / float(len(s))

    # Ensure labels are of type np.array()
    s = np.asarray(s)

    # Create cross-validation object for out-of-sample predicted probabilities.
    # CV folds preserve the fraction of noisy positive and
    # noisy negative examples in each class.
    kf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=seed)

    # Intialize psx array
    psx = np.zeros((len(s), K))

    # Split X and s into "cv_n_folds" stratified folds.
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(X, s)):

        clf_copy = copy.deepcopy(clf)

        # Select the training and holdout cross-validated sets.
        X_train_cv, X_holdout_cv = X[cv_train_idx], X[cv_holdout_idx]
        s_train_cv, s_holdout_cv = s[cv_train_idx], s[cv_holdout_idx]

        # Fit the clf classifier to the training set and
        # predict on the holdout set and update psx.
        clf_copy.fit(X_train_cv, s_train_cv)
        psx_cv = clf_copy.predict_proba(X_holdout_cv) # P(s = k|x) # [:,1]
        psx[cv_holdout_idx] = psx_cv

    # Compute the confident counts of all pairwise label-flipping mislabeling rates.
    confident_joint = compute_confident_joint(
        s=s,
        psx=psx, # P(s = k|x)
        thresholds=thresholds,
        calibrate=calibrate,
    )

    return confident_joint, psx


def estimate_py_noise_matrices_and_cv_pred_proba(
    X,
    s,
    clf=logreg(multi_class='auto', solver='lbfgs'),
    cv_n_folds=5,
    thresholds=None,
    converge_latent_estimates=False,
    py_method='cnt',
    seed=None,
):
    '''This function computes the out-of-sample predicted
    probability P(s=k|x) for every example x in X using cross
    validation while also computing the confident counts noise
    rates within each cross-validated subset and returning
    the average noise rate across all examples.

    This function estimates the noise_matrix of shape (K, K). This is the
    fraction of examples in every class, labeled as every other class. The
    noise_matrix is a conditional probability matrix for P(s=k_s|y=k_y).

    Under certain conditions, estimates are exact, and in most
    conditions, estimates are within one percent of the actual noise rates.

    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array

    s : np.array
      A discrete vector of labels, s, which may contain mislabeling. "s" denotes
      the noisy label instead of \tilde(y), for ASCII encoding reasons.

    clf : sklearn.classifier or equivalent
      Default classifier used is logistic regression. Assumes clf
      has predict_proba() and fit() defined.

    cv_n_folds : int
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.

    thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
      P(s^=k|s=k). If an example has a predicted probability "greater" than
      this threshold, it is counted as having hidden label y = k. This is
      not used for pruning, only for estimating the noise rates using
      confident counts. This value should be between 0 and 1. Default is None.

    converge_latent_estimates : bool
      If true, forces numerical consistency of estimates. Each is estimated
      independently, but they are related mathematically with closed form
      equivalences. This will iteratively make them mathematically consistent.

    py_method : str
        How to compute the latent prior p(y=k). Default is "cnt" as it tends to
        work best, but you may also set this hyperparameter to "eqn" or "marginal".

    seed : int (default = None)
        Number to set the default state of the random number generator used to split
        the cross-validated folds. If None, uses np.random current random state.

    Returns
    ------
      Returns a tuple of five numpy array matrices in the form:
      (py, noise_matrix, inverse_noise_matrix,
      joint count matrix i.e. confident joint, predicted probability matrix)'''

    confident_joint, psx = estimate_confident_joint_and_cv_pred_proba(
        X=X,
        s=s,
        clf=clf,
        cv_n_folds=cv_n_folds,
        thresholds=thresholds,
        seed=seed,
    )

    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        s=s,
        py_method=py_method,
        converge_latent_estimates=converge_latent_estimates,
    )

    return py, noise_matrix, inv_noise_matrix, confident_joint, psx


def estimate_cv_predicted_probabilities(
    X,
    labels, # class labels can be noisy (s) or not noisy (y).
    clf=logreg(multi_class='auto', solver='lbfgs'),
    cv_n_folds=5,
    seed=None,
):
    '''This function computes the out-of-sample predicted
    probability [P(s=k|x)] for every example in X using cross
    validation. Output is a np.array of shape (N, K) where N is
    the number of training examples and K is the number of classes.

    Parameters
    ----------

    X : np.array
      Input feature matrix (N, D), 2D numpy array

    labels : np.array or list of ints from [0,1,..,K-1]
      A discrete vector of class labels which may or may not contain mislabeling

    clf : sklearn.classifier or equivalent
      Default classifier used is logistic regression. Assumes clf
      has predict_proba() and fit() defined.

    cv_n_folds : int
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.

    seed : int (default = None)
        Number to set the default state of the random number generator used to split
        the cross-validated folds. If None, uses np.random current random state.

    Returns
    --------
    psx : np.array (shape (N, K))
        P(s=k|x) is a matrix with K (noisy) probabilities for each of the N examples x.
        This is the probability distribution over all K classes, for each
        example, regarding whether the example has label s==k P(s=k|x). psx should
        have been computed using 3 (or higher) fold cross-validation.'''

    return estimate_py_noise_matrices_and_cv_pred_proba(
        X=X,
        s=labels,
        clf=clf,
        cv_n_folds=cv_n_folds,
        seed=seed,
    )[-1]


def estimate_noise_matrices(
    X,
    s,
    clf=logreg(multi_class='auto', solver='lbfgs'),
    cv_n_folds=5,
    thresholds=None,
    converge_latent_estimates=True,
    seed=None,
):
    '''Estimates the noise_matrix of shape (K, K). This is the
    fraction of examples in every class, labeled as every other class. The
    noise_matrix is a conditional probability matrix for P(s=k_s|y=k_y).

    Under certain conditions, estimates are exact, and in most
    conditions, estimates are within one percent of the actual noise rates.

    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array

    s : np.array
      A discrete vector of labels, s, which may contain mislabeling

    clf : sklearn.classifier or equivalent
      Default classifier used is logistic regression. Assumes clf
      has predict_proba() and fit() defined.

    cv_n_folds : int
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.

    thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
      P(s^=k|s=k). If an example has a predicted probability "greater" than
      this threshold, it is counted as having hidden label y = k. This is
      not used for pruning, only for estimating the noise rates using
      confident counts. This value should be between 0 and 1. Default is None.

    converge_latent_estimates : bool
      If true, forces numerical consistency of estimates. Each is estimated
      independently, but they are related mathematically with closed form
      equivalences. This will iteratively make them mathematically consistent.

    seed : int (default = None)
        Number to set the default state of the random number generator used to split
        the cross-validated folds. If None, uses np.random current random state.

    Returns
    ------
        A two-item tuple containing (noise_matrix, inv_noise_matrix).'''

    return estimate_py_noise_matrices_and_cv_pred_proba(
        X=X,
        s=s,
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
    '''Computes py := P(y=k) and both noise_matrix and inverse_noise_matrix,
    by numerically converging ps := P(s=k), py, and the noise matrices.

    Forces numerical consistency of estimates. Each is estimated
    independently, but they are related mathematically with closed form
    equivalences. This will iteratively make them mathematically consistent.

    py := P(y=k) and the inverse noise matrix P(y=k_y|s=k_s) specify one another,
    meaning one can be computed from the other and vice versa. When numerical
    discrepancy exists due to poor estimation, they can be made to agree by repeatedly
    computing one from the other, for some a certain number of iterations (3-10 works fine.)

    Do not set iterations too high or performance will decrease as small deviations
    will get perturbated over and over and potentially magnified.

    Note that we have to first converge the inverse_noise_matrix and py,
    then we can update the noise_matrix, then repeat. This is becauase the
    inverse noise matrix depends on py (which is unknown/latent), but the
    noise matrix depends on ps (which is known), so there will be no change
    in the noise matrix if we recompute it when py and inverse_noise_matrix change.


    Parameters
    ----------

    ps : np.array (shape (K, ) or (1, K))
        The fraction (prior probability) of each observed, noisy class label, P(y = k).

    noise_matrix : np.array of shape (K, K), K = number of classes
        A conditional probablity matrix of the form P(s=k_s|y=k_y) containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1.

    inverse_noise_matrix : np.array of shape (K, K), K = number of classes
        A conditional probablity matrix of the form P(y=k_y|s=k_s) representing
        the estimated fraction observed examples in each class k_s, that are
        mislabeled examples from every other class k_y. If None, the
        inverse_noise_matrix will be computed from psx and s.
        Assumes columns of inverse_noise_matrix sum to 1.

    Returns
    ------
        Three np.arrays of the form (py, noise_matrix, inverse_noise_matrix) with py
        and inverse_noise_matrix and noise_matrix having numerical agreement.'''

    for j in range(noise_matrix_iterations):
        for i in range(inv_noise_matrix_iterations):
            inverse_noise_matrix = compute_inv_noise_matrix(py, noise_matrix, ps)
            py = compute_py(ps, noise_matrix, inverse_noise_matrix)
        noise_matrix = compute_noise_matrix_from_inverse(ps, inverse_noise_matrix, py)

    return py, noise_matrix, inverse_noise_matrix


# Deprecated methods

# pragma: no cover
def estimate_confident_joint_from_probabilities(
    s,
    psx,
    thresholds=None,
    force_ps=False,
    return_list_of_converging_cj_matrices=False,
):
    '''DEPRECATED AS OF VERSION 0.0.8.
    REMOVED AS OF VERSION 0.0.10.
    
    Estimates P(s,y), the confident counts of the latent
    joint distribution of true and noisy labels
    using observed s and predicted probabilities psx.

    UNLIKE compute_confident_joint, this function calibrates
    the confident joint estimate P(s=i, y=j) such that
    np.sum(cj) == len(s) and np.sum(cj, axis = 1) == np.bincount(s).

    Parameters
    ----------

    s : np.array
        A discrete vector of labels, s, which may contain mislabeling. "s" denotes
        the noisy label instead of \tilde(y), for ASCII encoding reasons.

    psx : np.array (shape (N, K))
        P(s=k|x) is a matrix with K (noisy) probabilities for each of the N examples x.
        This is the probability distribution over all K classes, for each
        example, regarding whether the example has label s==k P(s=k|x). psx should
        have been computed using 3 (or higher) fold cross-validation.

    thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
        P(s^=k|s=k). If an example has a predicted probability "greater" than
        this threshold, it is counted as having hidden label y = k. This is
        not used for pruning, only for estimating the noise rates using
        confident counts. This value should be between 0 and 1. Default is None.

    force_ps : bool or int
        If true, forces the output confident_joint matrix to have p(s) closer to the true
        p(s). The method used is SGD with a learning rate of eta = 0.5.
        If force_ps is an integer, it represents the number of epochs.
        Setting this to True is not always good. To make p(s) match, fewer confident
        examples are used to estimate the confident_joint, resulting in poorer estimation of
        the overall matrix even if p(s) is more accurate.

    return_list_of_converging_cj_matrices : bool (default = False)
        When force_ps is true, it converges the joint count matrix that is returned.
        Setting this to true will return the list of the converged matrices. The first
        item in the list is the original and the last item is the final result.

    Output
    ------
        confident_joint matrix count(s, y) : np.array (shape (K, K))
        where np.sum(confident_joint) ~ len(s) and rows sum to np.bincount(s)'''
    
    w = '''WARNING! THIS METHOD IS DEPRICATED.
    USE compute_confident_joint INSTEAD.
    THIS METHOD WILL BE ~REMOVED~ in cleanlab version 0.0.10.'''
    warnings.warn(w)

    # Number of classes
    K = len(np.unique(s))
    # 'ps' is p(s=k)
    ps = value_counts(s) / float(len(s))
    # Estimate the probability thresholds for confident counting
    s = np.asarray(s)
    if thresholds is None:
        thresholds = [np.mean(psx[:,k][s == k]) for k in range(K)] # P(s^=k|s=k)
    thresholds = np.asarray(thresholds)
    # joint counts
    cjs = []
    sgd_epochs = 5 if force_ps is True else 1 # Default 5 epochs if force_ps
    if type(force_ps) == int:
        sgd_epochs = force_ps
    for sgd_iteration in range(sgd_epochs): #  ONLY 1 iteration by default.
        # Compute the confident joint.
        confident_joint = compute_confident_joint(s, psx, K, thresholds)
        cjs.append(confident_joint)

        if force_ps:
            joint_ps = confident_joint.sum(axis=1) / float(np.sum(confident_joint))
            # Update thresholds (SGD) to converge p(s) of joint with actual p(s)
            eta = 0.5 # learning rate
            thresholds += eta * (joint_ps - ps)
        else: # Do not converge p(s) of joint with actual p(s)
            break

    return cjs if return_list_of_converging_cj_matrices else confident_joint