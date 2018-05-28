
# coding: utf-8

# In[ ]:

# Notes
# -----

# s - used to denote the noisy label 
# Typically,\tilde(y) is used in the literature

# Class labels (K classes) must be formatted as natural numbers: 0, 1, 2, ..., K-1
# Do not skip a natural number, i.e. 0, 1, 3, 4, .. is ***NOT*** okay!


# In[ ]:

from __future__ import print_function

from sklearn.linear_model import LogisticRegression as logreg
from sklearn.model_selection import StratifiedKFold
import numpy as np
import math


# In[ ]:

# Leave at least this many examples in each class after
# pruning, regardless if noise estimates are larger.
MIN_NUM_PER_CLASS = 5


# In[1]:

class RankPruning(object):
    '''Rank Pruning is a state-of-the-art algorithm (2017) for 
      multiclass classification with (potentially extreme) mislabeling 
      across any or all pairs of class labels. It works with ANY classifier,
      including deep neural networks. See clf parameter.
    This subfield of machine learning is referred to as Confident Learning.
    Rank Pruning also achieves state-of-the-art performance for binary
      classification with noisy labels and positive-unlabeled
      learning (PU learning) where a subset of positive examples is given and
      all other examples are unlabeled and assumed to be negative examples.
    Rank Pruning works by "learning from confident examples." Confident examples are
      identified as examples with high predicted probability for their training label.
    Given any classifier having the predict_proba() method, an input feature matrix, X, 
      and a discrete vector of labels, s, which may contain mislabeling, Rank Pruning 
      estimates the classifications that would be obtained if the hidden, true labels, y,
      had instead been provided to the classifier during training.
    "s" denotes the noisy label instead of \tilde(y), for ASCII encoding reasons.

    Parameters 
    ----------
    clf : sklearn.classifier or equivalent class
      The clf object must have the following three functions defined:
        1. clf.predict_proba(X) # Predicted probabilities
        2. clf.predict(X) # Predict labels
        3. clf.fit(X,y) # Train classifier
      Stores the classifier used in Rank Pruning.
      Default classifier used is logistic regression.

    noise_matrix : np.array of shape (K, K), K = number of classes 
      A conditional probablity matrix of the form P(s=k_s|y=k_y) containing
      the fraction of examples in every class, (mis)labeled as every other class.
      Only provide this matrix if you know the fractions of mislabeling for all
      pairs of classes already. Noise rates may be referred to as rho in literature.
      Assumes columns of noise_matrix sum to 1.'''  
  
  
    def __init__(self, clf = None):
        self.clf = logreg() if clf is None else clf
  
  
    def fit(
        self, 
        X,
        s,
        cv_n_folds = 5,
        pulearning = None,
        psx = None,
        thresholds = None,
        noise_matrix = None,
        inverse_noise_matrix = None,
        method = 'prune_by_noise_rate',
        converge_latent_estimates = False,
        verbose = False,
    ):
        '''This method implements the Rank Pruning mantra 'learning with confident examples.'
        This function fits the classifer (self.clf) to (X, s) accounting for the noise in
        both the positive and negative sets.

        Parameters
        ----------
        X : np.array
          Input feature matrix (N, D), 2D numpy array

        s : np.array
          A binary vector of labels, s, which may contain mislabeling. "s" denotes
          the noisy label instead of \tilde(y), for ASCII encoding reasons.

        cv_n_folds : int
          The number of cross-validation folds used to compute
          out-of-sample probabilities for each example in X.

        pulearning : int
          Set to the integer of the class that is perfectly labeled, if such
          a class exists. Otherwise, or if you are unsure, 
          leave pulearning = None (default).

        psx : np.array (shape (N, K))
          P(s=k|x) is a matrix with K (noisy) probabilities for each of the N examples x.
          This is the probability distribution over all K classes, for each
          example, regarding whether the example has label s==k P(s=k|x). psx should
          have been computed using 3 (or higher) fold cross-validation.
          If you are not sure, leave psx = None (default) and
          it will be computed for you using cross-validation.

        thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
          P(s^=k|s=k). If an example has a predicted probability "greater" than 
          this threshold, it is counted as having hidden label y = k. This is 
          not used for pruning, only for estimating the noise rates using 
          confident counts. This value should be between 0 and 1. Default is None.

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

        method : str
          'prune_by_class', 'prune_by_noise_rate', or 'both'. Method used for pruning.

        converge_latent_estimates : bool
          If true, forces numerical consistency of estimates. Each is estimated
          independently, but they are related mathematically with closed form 
          equivalences. This will iteratively enforce mathematically consistency. 

        verbose : bool
          Set to true if you wish to print additional information while running.

        Output
        ------
          Returns (noise_mask, sample_weight)'''
    
        # Check inputs
        assert_inputs_are_valid(X, s, psx)
        if noise_matrix is not None and np.trace(noise_matrix) <= 1:
            raise Exception("Trace(noise_matrix) must exceed 1.")
        if inverse_noise_matrix is not None and np.trace(inverse_noise_matrix) <= 1:
            raise Exception("Trace(inverse_noise_matrix) must exceed 1.")

        # Number of classes
        self.K = len(np.unique(s))

        # 'ps' is p(s=k)
        self.ps = value_counts(s) / float(len(s))

        # If needed, compute noise rates (fraction of mislabeling) for all classes. 
        # Also, if needed, compute P(s=k|x), denoted psx.
        
        if noise_matrix is not None:
            self.noise_matrix = noise_matrix
            if inverse_noise_matrix is None:
                self.py, self.inverse_noise_matrix = compute_py_inv_noise_matrix(self.ps, self.noise_matrix)
        if inverse_noise_matrix is not None:
            self.inverse_noise_matrix = inverse_noise_matrix
            if noise_matrix is None:
                self.noise_matrix = compute_noise_matrix_from_inverse(self.ps, self.inverse_noise_matrix)
        if noise_matrix is None and inverse_noise_matrix is None:
            if psx is None:
                self.py, self.noise_matrix, self.inverse_noise_matrix, psx =                 compute_py_noise_matrices_and_cv_pred_proba(
                    X = X, 
                    s = s, 
                    clf = self.clf,
                    cv_n_folds = cv_n_folds,
                    thresholds = thresholds, 
                    converge_latent_estimates = converge_latent_estimates,
                )
            else: # psx is provided by user (assumed holdout probabilities)
                self.py, self.noise_matrix, self.inverse_noise_matrix, _ =                 estimate_py_and_noise_matrices_from_probabilities(
                    s = s, 
                    psx = psx,
                    thresholds = thresholds, 
                    converge_latent_estimates = converge_latent_estimates,
                )

        if psx is None: 
            psx = compute_cv_predicted_probabilities(
                X = X, 
                labels = s, 
                clf = self.clf,
                cv_n_folds = cv_n_folds,
                verbose = verbose,
            ) 

        # Zero out noise matrix entries if pulearning = the integer specifying the class without noise.
        if pulearning is not None:
            self.noise_matrix = remove_noise_from_class(self.noise_matrix, class_without_noise=pulearning)
            # TODO: self.inverse_noise_matrix = remove_noise_from_class(self.inverse_noise_matrix, class_without_noise=pulearning)

        # This is the actual work of this function.

        # Get the indices of the examples we wish to prune
        self.noise_mask = get_noise_indices(s, psx, self.inverse_noise_matrix, method=method)

        X_mask = ~self.noise_mask
        X_pruned = X[X_mask]
        s_pruned = s[X_mask]

        # Re-weight examples in the loss function for the final fitting
        # s.t. the "apparent" original number of examples in each class
        # is preserved, even though the pruned sets may differ.
        self.sample_weight = np.ones(np.shape(s_pruned))
        for k in range(self.K): 
            self.sample_weight[s_pruned == k] = 1.0 / self.noise_matrix[k][k]

        self.clf.fit(X_pruned, s_pruned, sample_weight=self.sample_weight)

        return self.clf
    
    def predict(self, X):
        '''Returns a binary vector of predictions.'''

        return self.clf.predict(X)
  
  
    def predict_proba(self, X):
        '''Returns a vector of probabilties P(y=k)
        for each example in X.'''

        return self.clf.predict_proba(X)[:,1]


# In[ ]:

# Rank Pruning specific helper functions exposed to rankpruning module. 

def assert_inputs_are_valid(X, s, psx = None):
    '''Checks that X, s, and psx
    are correctly formatted'''

    if psx is not None:
        if not isinstance(psx, (np.ndarray, np.generic)):
            raise TypeError("psx should be a numpy array.")
    if len(psx) != len(s):
        raise ValueError("psx and s must have same length.")
    # Check for valid probabilities.
    if (psx < 0).any() or (psx > 1).any():
        raise ValueError("Values in psx must be between 0 and 1.")

    if not isinstance(s, (np.ndarray, np.generic)):
        raise TypeError("s should be a numpy array.")
    if not isinstance(X, (np.ndarray, np.generic)):
        raise TypeError("X should be a numpy array.")
    
    
def remove_noise_from_class(noise_matrix, class_without_noise):
    '''A helper function in the setting of PU learning.
    Sets all P(s=class_without_noise|y=any_other_class) = 0
    in noise_matrix for pulearning setting, where we have 
    generalized the positive class in PU learning to be any
    class of choosing, denoted by class_without_noise.

    Parameters
    ----------

    noise_matrix : np.array of shape (K, K), K = number of classes 
        A conditional probablity matrix of the form P(s=k_s|y=k_y) containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1.

    class_without_noise : int
        Integer value of the class that has no noise. Traditionally,
        this is 1 (positive) for PU learning.'''
  
    # Number of classes
    K = len(noise_matrix)

    cwn = class_without_noise
    x = np.copy(noise_matrix)

    # Set P( s = cwn | y != cwn) = 0 (no noise)
    x[cwn, [i for i in range(K) if i!=cwn]] = 0.0

    # Normalize columns by increasing diagnol terms
    # Ensures noise_matrix is a valid probability matrix
    for i in range(K):
        x[i][i] = 1 - float(np.sum(x[:,i]) - x[i][i])

    return x
    

def estimate_joint_counts_from_probabilities(
    s, 
    psx, 
    thresholds = None, 
    force_ps = False,
    return_list_of_converging_jc_matrices = False,
):
    '''Estimates P(s,y), the confident counts of the latent 
    joint distribution of true and noisy labels 
    using observed s and predicted probabilities psx.

    Important! This function assumes that psx are out-of-sample 
    holdout probabilities. This can be done with cross validation. If
    the probabilities are not computed out-of-sample, overfitting may occur.

    This function estimates the joint of shape (K, K). This is the
    confident counts of examples in every class, labeled as every other class.

    Under certain conditions, estimates are exact, and in most
    conditions, the estimate is within 1 percent of the truth.

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
        If true, forces the output joint_count matrix to have p(s) closer to the true
        p(s). The method used is SGD with a learning rate of eta = 0.5.
        If force_ps is an integer, it represents the number of epochs.
        Setting this to True is not always good. To make p(s) match, fewer confident
        examples are used to estimate the joint_count, resulting in poorer estimation of
        the overall matrix even if p(s) is more accurate. 
        
    return_list_of_converging_jc_matrices : bool (default = False)
        When force_ps is true, it converges the joint count matrix that is returned.
        Setting this to true will return the list of the converged matrices. The first
        item in the list is the original and the last item is the final result.

    Output
    ------
        joint_count matrix count(s, y) : np.array (shape (K, K))'''
    
    # Number of classes
    K = len(np.unique(s))  
    # 'ps' is p(s=k)
    ps = value_counts(s) / float(len(s))
    # joint counts
    jcs = []
    
    # Ensure labels are of type np.array()
    s = np.asarray(s)
    
    sgd_epochs = force_ps if type(force_ps) == int else 1   
    for sgd_iteration in range(sgd_epochs): 
        # Estimate the probability thresholds for confident counting 
        if thresholds is None:
            thresholds = [np.mean(psx[:,k][s == k]) for k in range(K)] # P(s^=k|s=k)

        # Confident examples are those that we are confident have label y = k
        # Estimate the (K, K) matrix of confident examples having s = k_s and y = k_y
        joint_count = np.zeros((K,K))
        for k_s in range(K): # k_s is the class value k of noisy label s
            for k_y in range(K): # k_y is the (guessed) class value k of true label y
                joint_count[k_s][k_y] = sum((psx[:,k_y] >= thresholds[k_y]) & (s == k_s))
        jcs.append(joint_count)
        
        if force_ps:
            joint_ps = joint_count.sum(axis=1) / float(np.sum(joint_count))
            # Update thresholds (SGD) to converge p(s) of joint with actual p(s)    
            eta = 0.5 # learning rate
            thresholds += eta * (joint_ps - ps)
        else: # Do not converge p(s) of joint with actual p(s)
            break
            
    return jcs if return_list_of_converging_jc_matrices else joint_count
 
    
def estimate_latent(
    joint_count, 
    s, 
    py_method = 'cnt', 
    converge_latent_estimates = False,
):
    '''Computes the latent prior p(y), the noise matrix P(s|y) and the
    inverse noise matrix P(y|s) from the joint_count count(s, y). The
    joint_count estimated by estimate_joint_counts_from_probabilities()
    by counting confident examples.

    Parameters
    ----------

    s : np.array
        A discrete vector of labels, s, which may contain mislabeling. "s" denotes
        the noisy label instead of \tilde(y), for ASCII encoding reasons.
        
    joint_count : np.array (shape (K, K), type int)
        A K,K integer matrix of count(s=k, y=k). Captures the joint disribution of
        the noisy and true labels P_{s,y}. Each entry in the matrix contains
        the number of examples confidently counted into both classes.
        
    py_method : str
        How to compute the latent prior p(y=k). Default is "cnt" as it tends to
        work best, but you may also set this hyperparameter to "eqn" or "marginal".

    converge_latent_estimates : bool
      If true, forces numerical consistency of estimates. Each is estimated
      independently, but they are related mathematically with closed form 
      equivalences. This will iteratively make them mathematically consistent. '''
    
    # Number of classes
    K = len(np.unique(s))  
    # 'ps' is p(s=k)
    ps = value_counts(s) / float(len(s))
    
    # Ensure labels are of type np.array()
    s = np.asarray(s)
    
    # Number of training examples confidently counted from each noisy class
    s_count = joint_count.sum(axis=1).astype(float)
    
    # Number of training examples confidently counted into each true class
    y_count = joint_count.sum(axis=0).astype(float)
    
    # Confident Counts Estimator for p(s=k_s|y=k_y) ~ |s=k_s and y=k_y| / |y=k_y|
    noise_matrix = joint_count / y_count

    # Confident Counts Estimator for p(y=k_y|s=k_s) ~ |y=k_y and s=k_s| / |s=k_s|
    inv_noise_matrix = joint_count.T / s_count
    
    if py_method == 'cnt': 
        py = (y_count / s_count) * ps
        # Equivalently,
        # py = inv_noise_matrix.diagonal() / noise_matrix.diagonal() * ps
    elif py_method == 'eqn':
        py = np.linalg.inv(noise_matrix).dot(ps)
    elif py_method == 'marginal':
        py = y_count / float(sum(y_count))
    else:
        raise ValueError('py_method parameter should be cnt, eqn, or marginal')
    
    # Clip py and noise rates into proper range [0,1)
    # For py, no class should have probability 0 so we use 1e-5
    py = clip_values(py, low=1e-5, high=1.0, new_sum = 1.0)
    noise_matrix = clip_noise_rates(noise_matrix) 
    inv_noise_matrix = clip_noise_rates(inv_noise_matrix)

    if converge_latent_estimates:
        py, noise_matrix, inv_noise_matrix = converge_estimates(ps, py, noise_matrix, inv_noise_matrix)
        # Again clip py and noise rates into proper range [0,1)
        py = clip_values(py, low=1e-5, high=1.0, new_sum = 1.0) 
        noise_matrix = clip_noise_rates(noise_matrix) 
        inv_noise_matrix = clip_noise_rates(inv_noise_matrix)

    return py, noise_matrix, inv_noise_matrix                  
    
    
def estimate_py_and_noise_matrices_from_probabilities(
    s, 
    psx, 
    thresholds = None,
    converge_latent_estimates = True,
    force_ps = False,
    py_method = 'cnt', 
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
        
    force_ps : bool or int
        If true, forces the output joint_count matrix to have p(s) closer to the true
        p(s). The method used is SGD with a learning rate of eta = 0.5.
        If force_ps is an integer, it represents the number of epochs.
        Setting this to True is not always good. To make p(s) match, fewer confident
        examples are used to estimate the joint_count, resulting in poorer estimation of
        the overall matrix even if p(s) is more accurate. 
        
    py_method : str
        How to compute the latent prior p(y=k). Default is "cnt" as it tends to
        work best, but you may also set this hyperparameter to "eqn" or "marginal".

    Output
    ------
        py, noise_matrix, inverse_noise_matrix'''
  
    joint_count = estimate_joint_counts_from_probabilities(s, psx, thresholds, force_ps)
    py, noise_matrix, inv_noise_matrix = estimate_latent(        
        joint_count=joint_count, 
        s=s, 
        py_method=py_method, 
        converge_latent_estimates=converge_latent_estimates,
    )
    
    return py, noise_matrix, inv_noise_matrix, joint_count


def estimate_joint_counts_and_cv_pred_proba(
    X, 
    s, 
    clf = logreg(),
    cv_n_folds = 5,
    thresholds = None,
    force_ps = False,
    return_list_of_converging_jc_matrices = False,
    seed = None,
):
    '''Estimates P(s,y), the confident counts of the latent 
    joint distribution of true and noisy labels 
    using observed s and predicted probabilities psx. 

    The output of this function is a numpy array of shape (K, K). 

    Under certain conditions, estimates are exact, and in many
    conditions, estimates are within one percent of actual.

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
        
    force_ps : bool or int
        If true, forces the output joint_count matrix to have p(s) closer to the true
        p(s). The method used is SGD with a learning rate of eta = 0.5.
        If force_ps is an integer, it represents the number of epochs.
        Setting this to True is not always good. To make p(s) match, fewer confident
        examples are used to estimate the joint_count, resulting in poorer estimation of
        the overall matrix even if p(s) is more accurate. 
        
    return_list_of_converging_jc_matrices : bool (default = False)
        When force_ps is true, it converges the joint count matrix that is returned.
        Setting this to true will return the list of the converged matrices. The first
        item in the list is the original and the last item is the final result.
        
    seed : int (default = None)
        Number to set the default state of the random number generator used to split 
        the cross-validated folds. If None, uses np.random current random state.

    Output
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
    kf = StratifiedKFold(n_splits = cv_n_folds, shuffle = True, random_state = seed)

    # Intialize result storage and final psx array
#     py_per_cv_fold = []
#     noise_matrix_per_cv_fold = []
#     inv_noise_matrix_per_cv_fold = []
    joint_count_per_cv_fold = []
    psx = np.zeros((len(s), K))

    # Split X and s into "cv_n_folds" stratified folds.
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(X, s)):

        # Select the training and holdout cross-validated sets.
        X_train_cv, X_holdout_cv = X[cv_train_idx], X[cv_holdout_idx]
        s_train_cv, s_holdout_cv = s[cv_train_idx], s[cv_holdout_idx]

        # Fit the clf classifier to the training set and 
        # predict on the holdout set and update psx. 
        clf.fit(X_train_cv, s_train_cv)
        psx_cv = clf.predict_proba(X_holdout_cv) # P(s = k|x) # [:,1]
        psx[cv_holdout_idx] = psx_cv

        # Compute and append the confident counts noise estimators
        # to estimate the positive and negative mislabeling rates.
        joint_count_cv = estimate_joint_counts_from_probabilities(
            s = s_holdout_cv, 
            psx = psx_cv, # P(s = k|x)
            thresholds = thresholds, 
            force_ps = force_ps,
            return_list_of_converging_jc_matrices = return_list_of_converging_jc_matrices,
        )
#         py_cv, noise_matrix_cv, inv_noise_matrix_cv, joint_count_cv = \
#         estimate_py_and_noise_matrices_from_probabilities(
#             s = s_holdout_cv, 
#             psx = psx_cv, # P(s = k|x) 
#             thresholds = thresholds,
#             converge_latent_estimates = False, # Converge at end, if we converge.
#         )

#         py_per_cv_fold.append(py_cv)
#         noise_matrix_per_cv_fold.append(noise_matrix_cv)
#         inv_noise_matrix_per_cv_fold.append(inv_noise_matrix_cv)
        joint_count_per_cv_fold.append(joint_count_cv)
    
    if return_list_of_converging_jc_matrices:
        jcs = [np.sum(np.stack(jcs), axis=0) for jcs in zip(*joint_count_per_cv_fold)] 
        return jcs, psx

    
# CONSIDER REMOVING THE FOLLOWING CODE WHICH IS NOW OBSELETE

    # Compute mean py, noise_matrix, inverse noise marix (disregarding nan
    #   or inf values) and psx
    # Mean is computed by stacking each cv fold's noise_matrix (forming 
    #  a 3D tensor), then average along the newly formed depth axis to
    #  to yield the mean 2D tensor.
#     py = np.apply_along_axis(func1d=_mean_without_nan_inf, axis=0, arr=np.vstack(py_per_cv_fold))
#     noise_matrix = np.apply_along_axis(func1d=_mean_without_nan_inf, axis=2, arr=np.dstack(noise_matrix_per_cv_fold))
#     inv_noise_matrix = np.apply_along_axis(func1d=_mean_without_nan_inf, axis=2, arr=np.dstack(inv_noise_matrix_per_cv_fold))
#     joint_count = np.apply_along_axis(func1d=_mean_without_nan_inf, axis=2, arr=np.dstack(joint_count_per_cv_fold)) 

#     if converge_latent_estimates: # Force numerical consistency of estimates.
#         py, noise_matrix, inv_noise_matrix = converge_estimates(ps, py, noise_matrix, inv_noise_matrix)
  
#     return py, noise_matrix, inv_noise_matrix, psx

    joint_count = np.sum(np.stack(joint_count_per_cv_fold), axis=0)
    return joint_count, psx


def compute_py_noise_matrices_and_cv_pred_proba(
    X, 
    s, 
    clf = logreg(),
    cv_n_folds = 5,
    thresholds = None,
    converge_latent_estimates = False,
    force_ps = False,
    return_list_of_converging_jc_matrices = False,
    py_method = 'cnt',
    seed = None,
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
        
    force_ps : bool or int
        If true, forces the output joint_count matrix to have p(s) closer to the true
        p(s). The method used is SGD with a learning rate of eta = 0.5.
        If force_ps is an integer, it represents the number of epochs.
        Setting this to True is not always good. To make p(s) match, fewer confident
        examples are used to estimate the joint_count, resulting in poorer estimation of
        the overall matrix even if p(s) is more accurate. 
        
    return_list_of_converging_jc_matrices : bool (default = False)
        When force_ps is true, it converges the joint count matrix that is returned.
        Setting this to true will return the list of the converged matrices. The first
        item in the list is the original and the last item is the final result.
        
    py_method : str
        How to compute the latent prior p(y=k). Default is "cnt" as it tends to
        work best, but you may also set this hyperparameter to "eqn" or "marginal".
        
    seed : int (default = None)
        Number to set the default state of the random number generator used to split 
        the cross-validated folds. If None, uses np.random current random state.

    Output
    ------
      Returns a tuple of three numpy array matrices in the form:
      (noise_matrix, inverse_noise_matrix, predicted probability matrix)'''
    
    joint_count, psx = estimate_joint_counts_and_cv_pred_proba(
        X = X, 
        s = s, 
        clf = clf,
        cv_n_folds = cv_n_folds,
        thresholds = thresholds,
        force_ps = force_ps,
        return_list_of_converging_jc_matrices = return_list_of_converging_jc_matrices,
        seed = seed,
    )
    
    py, noise_matrix, inv_noise_matrix = estimate_latent(
        joint_count = joint_count, 
        s = s, 
        py_method = py_method, 
        converge_latent_estimates = converge_latent_estimates,
    )
    
    return py, noise_matrix, inv_noise_matrix, psx 


def compute_cv_predicted_probabilities(
    X, 
    labels, # class labels can be noisy (s) or not noisy (y).
    clf = logreg(),
    cv_n_folds = 5,
    seed = None,
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
    '''

    return compute_py_noise_matrices_and_cv_pred_proba(
        X = X, 
        s = labels, 
        clf = clf,
        cv_n_folds = cv_n_folds,
        seed = seed,
    )[-1]


def compute_noise_matrices(
    X, 
    s, 
    clf = logreg(),
    cv_n_folds = 5,
    thresholds = None,
    converge_latent_estimates = True,
    seed = None,
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
        the cross-validated folds. If None, uses np.random current random state.'''

    return compute_py_noise_matrices_and_cv_pred_proba(
        X = X, 
        s = s, 
        clf = clf,
        cv_n_folds = cv_n_folds,
        thresholds = thresholds,
        converge_latent_estimates = converge_latent_estimates,
        seed = seed,
    )[1:-1]


def compute_ps_py_inv_noise_matrix(s, noise_matrix):
    '''Compute ps := P(s=k), py := P(y=k), and the inverse noise matrix.

    Parameters
    ----------

    s : np.array
        A discrete vector of labels, s, which may contain mislabeling. "s" denotes
        the noisy label instead of \tilde(y), for ASCII encoding reasons.

    noise_matrix : np.array of shape (K, K), K = number of classes 
        A conditional probablity matrix of the form P(s=k_s|y=k_y) containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1.'''
  
    # 'ps' is p(s=k)
    ps = value_counts(s) / float(len(s))

    py, inverse_noise_matrix = compute_py_inv_noise_matrix(ps, noise_matrix)
    return ps, py, inverse_noise_matrix


def compute_py_inv_noise_matrix(ps, noise_matrix):
    '''Compute py := P(y=k), and the inverse noise matrix.

    Parameters
    ----------

    ps : np.array (shape (K, 1))
        The fraction (prior probability) of each observed, noisy class label, P(s = k)

    noise_matrix : np.array of shape (K, K), K = number of classes 
        A conditional probablity matrix of the form P(s=k_s|y=k_y) containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1.'''
  
    # Number of classes
    K = len(ps)

    # 'py' is p(y=k) = noise_matrix^(-1) * p(s=k)
    # because in *vector computation*: P(s=k|y=k) * p(y=k) = P(s=k)
    # The pseudoinverse is used when noise_matrix is not invertible.
    py = np.linalg.inv(noise_matrix).dot(ps)

    # No class should have probability 0 so we use .001
    # Make sure valid probabilites that sum to 1.0
    py = clip_values(py, low=0.001, high=1.0, new_sum = 1.0)

    # All the work is done in this function (below)
    return py, compute_inv_noise_matrix(py, noise_matrix, ps)


def compute_inv_noise_matrix(py, noise_matrix, ps = None):
  '''Compute the inverse noise matrix if py := P(y=k) is given.

  Parameters
  ----------

  py : np.array (shape (K, 1))
    The fraction (prior probability) of each true, hidden class label, P(y = k)
    
  noise_matrix : np.array of shape (K, K), K = number of classes 
    A conditional probablity matrix of the form P(s=k_s|y=k_y) containing
    the fraction of examples in every class, labeled as every other class.
    Assumes columns of noise_matrix sum to 1.

  ps : np.array (shape (K, 1))
    The fraction (prior probability) of each observed, noisy class label, P(s = k).
    ps is easily computable from py and should only be provided if it has
    already been precomputed, to increase code efficiency.'''
  
  # Number of classes
  K = len(py)
  
  # 'ps' is p(s=k) = noise_matrix * p(s=k)
  # because in *vector computation*: P(s=k|y=k) * p(y=k) = P(s=k)
  if ps is None:
    ps = noise_matrix.dot(py)
  
  # Estimate the (K, K) inverse noise matrix P(y = k_y | s = k_s)
  inverse_noise_matrix = np.empty(shape=(K,K))
  # k_s is the class value k of noisy label s
  for k_s in range(K):
      # k_y is the (guessed) class value k of true label y
      for k_y in range(K):
        # P(y|s) = P(s|y) * P(y) / P(s)
        inverse_noise_matrix[k_y][k_s] = noise_matrix[k_s][k_y] * py[k_y] / ps[k_s]
  
  # Clip inverse noise rates P(y=k_s|y=k_y) into proper range [0,1)
  return clip_noise_rates(inverse_noise_matrix)


def compute_noise_matrix_from_inverse(ps, inverse_noise_matrix, py = None):
  '''Compute the noise matrix P(s=k_s|y=k_y).

  Parameters
  ----------

  py : np.array (shape (K, 1))
    The fraction (prior probability) of each true, hidden class label, P(y = k)
    
  noise_matrix : np.array of shape (K, K), K = number of classes 
    A conditional probablity matrix of the form P(s=k_s|y=k_y) containing
    the fraction of examples in every class, labeled as every other class.
    Assumes columns of noise_matrix sum to 1.

  ps : np.array (shape (K, 1))
    The fraction (prior probability) of each observed, noisy class label, P(s = k).
    ps is easily computable from py and should only be provided if it has
    already been precomputed, to increase code efficiency.
    
  Output
  ------
  
  noise_matrix : np.array of shape (K, K), K = number of classes 
    A conditional probablity matrix of the form P(s=k_s|y=k_y) containing
    the fraction of examples in every class, labeled as every other class.
    Columns of noise_matrix sum to 1.'''
  
  # Number of classes s
  K = len(ps)
  
  # 'py' is p(y=k) = inverse_noise_matrix * p(y=k)
  # because in *vector computation*: P(y=k|s=k) * p(s=k) = P(y=k)
  if py is None:
    py = inverse_noise_matrix.dot(ps)
  
  # Estimate the (K, K) noise matrix P(s = k_s | y = k_y)
  noise_matrix = np.empty(shape=(K,K))
  # k_s is the class value k of noisy label s
  for k_s in range(K):
      # k_y is the (guessed) class value k of true label y
      for k_y in range(K):
        # P(s|y) = P(y|s) * P(s) / P(y)
        noise_matrix[k_s][k_y] = inverse_noise_matrix[k_y][k_s] * ps[k_s] / py[k_y]
  
  # Clip inverse noise rates P(y=k_y|y=k_s) into proper range [0,1)
  return clip_noise_rates(noise_matrix)

  
def compute_py(ps, noise_matrix, inverse_noise_matrix):
  '''Compute py := P(y=k) from ps := P(s=k), noise_matrix, and inverse noise matrix.
  
  This method is ** ROBUST ** - meaning it works well even when the
  noise matrices are estimated poorly by only using the diagonals of the matrices
  instead of all the probabilities in the entire matrix.

  Parameters
  ----------

  ps : np.array (shape (K, ) or (1, K)) 
    The fraction (prior probability) of each observed, noisy class label, P(s = k).
    
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
    
  Output
  ------
  
  py : np.array (shape (K, ) or (1, K))
    The fraction (prior probability) of each observed, noisy class label, P(y = k).'''
  
  if len(np.shape(ps)) > 2 or (len(np.shape(ps)) == 2 and np.shape(ps)[0] != 1):
    import warnings
    warnings.warn("Input parameter np.array 'ps' has shape " + str(np.shape(ps)) +                   ", but shape should be (K, ) or (1, K)")
  
  # Computing py this way avoids dividing by zero noise rates! Also more robust.
  # More robust because error est_p(y|s) / est_p(s|y) ~ p(y|s) / p(s|y) 
  py = ps * inverse_noise_matrix.diagonal() / noise_matrix.diagonal()
  # Make sure valid probabilites that sum to 1.0
  return clip_values(py, low=0.0, high=1.0, new_sum = 1.0)
  
  

def compute_pyx(psx, noise_matrix, inverse_noise_matrix):
  '''Compute pyx := P(y=k|x) from psx := P(s=k|x), and the noise_matrix and inverse
  noise matrix. 
  
  This method is ROBUST - meaning it works well even when the
  noise matrices are estimated poorly by only using the diagonals of the matrices
  which tend to be easy to estimate correctly.

  Parameters
  ----------
  
  psx : np.array (shape (N, K))
    P(s=k|x) is a matrix with K (noisy) probabilities for each of the N examples x.
    
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
    
  Output
  ------
  
  pyx : np.array (shape (N, K))
    P(y=k|x) is a matrix with K probabilities for each of the N examples x.'''
  
  if len(np.shape(psx)) != 2:
    raise ValueError("Input parameter np.array 'psx' has shape " + str(np.shape(psx)) +                   ", but shape should be (N, K)")
  
  pyx = psx * inverse_noise_matrix.diagonal() / noise_matrix.diagonal()
  # Make sure valid probabilites that sum to 1.0
  return np.apply_along_axis(
    func1d=clip_values, 
    axis=1, 
    arr=pyx,
    **{"low":0.0, "high":1.0, "new_sum":1.0}
  )
  

def converge_estimates(
    ps,
    py,
    noise_matrix, 
    inverse_noise_matrix, 
    inv_noise_matrix_iterations = 5,
    noise_matrix_iterations = 3,
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

    Output
    ------  
        Three np.arrays of the form (py, noise_matrix, inverse_noise_matrix) with py 
        and inverse_noise_matrix and noise_matrix having numerical agreement.'''  
  
    for j in range(noise_matrix_iterations):
        for i in range(inv_noise_matrix_iterations):
            inverse_noise_matrix = compute_inv_noise_matrix(py, noise_matrix, ps)
            py = compute_py(ps, noise_matrix, inverse_noise_matrix)
        noise_matrix = compute_noise_matrix_from_inverse(ps, inverse_noise_matrix, py)
    
    return py, noise_matrix, inverse_noise_matrix
  
  
def get_noise_indices(
    s, 
    psx, 
    inverse_noise_matrix = None,
    frac_noise = 1.0,
    num_to_remove_per_class = None,
    method = 'prune_by_noise_rate',
    joint_count = None,
):
    '''Returns the indices of most likely (confident) label errors in s. The
    number of indices returned is specified by frac_of_noise. When 
    frac_of_noise = 1.0, all "confidently" estimated noise indices are returned.

    Parameters
    ----------

    s : np.array
      A binary vector of labels, s, which may contain mislabeling. "s" denotes
      the noisy label instead of \tilde(y), for ASCII encoding reasons.
    
    psx : np.array (shape (N, K))
      P(s=k|x) is a matrix with K (noisy) probabilities for each of the N examples x.
      This is the probability distribution over all K classes, for each
      example, regarding whether the example has label s==k P(s=k|x). psx should
      have been computed using 3 (or higher) fold cross-validation.
      
    inverse_noise_matrix : np.array of shape (K, K), K = number of classes 
      A conditional probablity matrix of the form P(y=k_y|s=k_s) representing
      the estimated fraction observed examples in each class k_s, that are
      mislabeled examples from every other class k_y. If None, the 
      inverse_noise_matrix will be computed from psx and s.
      Assumes columns of inverse_noise_matrix sum to 1.
  
    frac_noise : float
      When frac_of_noise = 1.0, return all "confidently" estimated noise indices.
      Value in range (0, 1] that determines the fraction of noisy example 
      indices to return based on the following formula for example class k.
      frac_of_noise * number_of_mislabeled_examples_in_class_k, or equivalently    
      frac_of_noise * inverse_noise_rate_class_k * num_examples_with_s_equal_k
      
    num_to_remove_per_class : list of int of length K (# of classes)
      e.g. if K = 3, num_to_remove_per_class = [5, 0, 1] would return 
      the indices of the 5 most likely mislabeled examples in class s = 0,
      and the most likely mislabeled example in class s = 1.
      ***Only set this parameter if method == 'prune_by_class'
      
    method : str (default: 'prune_by_noise_rate')
      'prune_by_class', 'prune_by_noise_rate', or 'both'. Method used for pruning.
      'both' creates a mask based on removing the least likely in
      each class (prune_by_class) and a mask based on the most likely to be labeled
      another class (prune_by_noise_rate) and then 'AND's the two masks disjunctively
      such that an example must be pruned under both masks to be pruned.
        
    joint_count : np.array (shape (K, K), type int) (default: None)
        NOTE: If this is provided, pruning counts will be determined entirely by
        this input and the inverse_noise_matrix input WILL BE IGNORED!
        A K,K integer matrix of count(s=k, y=k). Captures the joint disribution of
        the noisy and true labels P_{s,y}. Each entry in the matrix contains
        the number of examples confidently counted into both classes.'''
  
    # Number of examples in each class of s
    s_counts = value_counts(s)
    # 'ps' is p(s=k)
    ps = s_counts / float(len(s))
    # Number of classes s
    K = len(psx.T)

    # Ensure labels are of type np.array()
    s = np.asarray(s)

    if inverse_noise_matrix is None:
        py, noise_matrix, inverse_noise_matrix, _ =         estimate_py_and_noise_matrices_from_probabilities(s, psx, converge_latent_estimates=converge_latent_estimates)
  
    # Estimate the number of examples to confidently prune per class.
    if joint_count is None:
        prune_count_matrix = inverse_noise_matrix * s_counts # Matrix of counts(y=k and s=l)
    else:
        prune_count_matrix = joint_count.T / float(joint_count.sum()) * len(s) # calibrate
    # Leave at least MIN_NUM_PER_CLASS examples per class.
    prune_count_matrix = keep_at_least_n_per_class(
        prune_count_matrix=prune_count_matrix, 
        n=MIN_NUM_PER_CLASS, 
        frac_noise=frac_noise,
    )
  
    if num_to_remove_per_class is not None:
        np.fill_diagonal(prune_count_matrix, s_counts - num_to_remove_per_class)

    # Initialize the boolean mask of noise indices.
    noise_mask = np.zeros(len(psx), dtype=bool)

    # Peform Pruning with threshold probabilities from BFPRT algorithm in O(n)

    if method == 'prune_by_class' or method == 'both':
        for k in range(K):
            if s_counts[k] > MIN_NUM_PER_CLASS: # Don't prune if not MIN_NUM_PER_CLASS
                num2prune = s_counts[k] - prune_count_matrix[k][k]
                # num2keep'th smallest probability of class k for examples with noisy label k
                threshold = np.partition(psx[:,k][s == k], num2prune)[num2prune]
                noise_mask = noise_mask | ((psx[:,k] < threshold) & (s == k))
  
    if method == 'both':
        noise_mask_by_class = noise_mask

    if method == 'prune_by_noise_rate' or method == 'both':
        noise_mask = np.zeros(len(psx), dtype=bool)
        for k in range(K):
            if s_counts[k] > MIN_NUM_PER_CLASS: # Don't prune if not MIN_NUM_PER_CLASS
                for j in range(K):
                    if k!=j: # Only prune for noise rates
                        num2prune = prune_count_matrix[k][j]
                        # num2prune'th largest probability of class k for examples with noisy label j
                        threshold = -np.partition(-psx[:,k][s == j], num2prune)[num2prune]
                        noise_mask = noise_mask | ((psx[:,k] > threshold) & (s == j))
            
    return noise_mask & noise_mask_by_class if method == 'both' else noise_mask


# In[ ]:

# Generic helper function`s exposed to rankpruning module.   

def keep_at_least_n_per_class(prune_count_matrix, n, frac_noise=1.0):
  '''Make sure every class has at least n examples after removing noise.
  Functionally, increase each column, increases the diagonal term #(y=k,s=k) of 
  prune_count_matrix until it is at least n, distributing the amount increased
  by subtracting uniformly from the rest of the terms in the column. When 
  frac_of_noise = 1.0, return all "confidently" estimated noise indices, otherwise
  this returns frac_of_noise fraction of all the noise counts, with diagonal terms
  adjusted to ensure column totals are preserved.

  Parameters
  ----------
    
  prune_count_matrix : np.array of shape (K, K), K = number of classes 
    A counts of mislabeled examples in every class. For this function, it
    does not matter what the rows or columns are, but the diagonal terms
    reflect the number of correctly labeled examples.
    
  n : int
    Number of examples to make sure are left in each class.

  frac_noise : float
    When frac_of_noise = 1.0, return all "confidently" estimated noise indices.
    Value in range (0, 1] that determines the fraction of noisy example 
    indices to return based on the following formula for example class k.
    frac_of_noise * number_of_mislabeled_examples_in_class_k, or equivalently    
    frac_of_noise * inverse_noise_rate_class_k * num_examples_with_s_equal_k
    
  Output
  ------
  
  prune_count_matrix : np.array of shape (K, K), K = number of classes 
    Number of examples to remove from each class, for every other class.'''
  
  K = len(prune_count_matrix)
  prune_count_matrix_diagonal = np.diagonal(prune_count_matrix)
  
  # Set diagonal terms less than n, to n. 
  new_diagonal = np.maximum(prune_count_matrix_diagonal, n)

  # Find how much diagonal terms were increased.
  diff_per_col = new_diagonal - prune_count_matrix_diagonal

  # Count non-zero, non-diagonal items per column
  # np.maximum(*, 1) makes this never 0 (we divide by this next)
  num_noise_rates_per_col = np.maximum(np.count_nonzero(prune_count_matrix, axis=0) - 1., 1.) 

  # Uniformly decrease non-zero noise rates by amount diagonal items were increased
  new_mat = prune_count_matrix - diff_per_col / num_noise_rates_per_col
  
  # Originally zero noise rates will now be negative, fix them back to zero
  new_mat[new_mat < 0] = 0

  # Round diagonal terms (correctly labeled examples)
  np.fill_diagonal(new_mat, new_diagonal.round())
  
  # Reduce (multiply) all noise rates (non-diagonal) by frac_noise and
  # increase diagonal by the total amount reduced in each column to preserve column counts.
  new_mat = reduce_prune_counts(new_mat, frac_noise)
  
  # These are counts, so return a matrix of ints.
  return new_mat.astype(int)
  

def reduce_prune_counts(prune_count_matrix, frac_noise=1.0):
  '''Reduce (multiply) all prune counts (non-diagonal) by frac_noise and
  increase diagonal by the total amount reduced in each column to 
  preserve column counts.

  Parameters
  ----------
    
  prune_count_matrix : np.array of shape (K, K), K = number of classes 
    A counts of mislabeled examples in every class. For this function, it
    does not matter what the rows or columns are, but the diagonal terms
    reflect the number of correctly labeled examples.

  frac_noise : float
    When frac_of_noise = 1.0, return all "confidently" estimated noise indices.
    Value in range (0, 1] that determines the fraction of noisy example 
    indices to return based on the following formula for example class k.
    frac_of_noise * number_of_mislabeled_examples_in_class_k, or equivalently    
    frac_of_noise * inverse_noise_rate_class_k * num_examples_with_s_equal_k'''
  
  new_mat = prune_count_matrix * frac_noise
  np.fill_diagonal(new_mat, prune_count_matrix.diagonal())
  np.fill_diagonal(new_mat, prune_count_matrix.diagonal() + np.sum(prune_count_matrix - new_mat, axis=0))
  
  # These are counts, so return a matrix of ints.
  return new_mat.astype(int)
  

def clip_noise_rates(noise_matrix):
  '''Clip all noise rates to proper range [0,1), but
  do not modify the diagonal terms because they are not
  noise rates.
  
  ASSUMES noise_matrix columns sum to 1.

  Parameters
  ----------
    
  noise_matrix : np.array of shape (K, K), K = number of classes 
    A conditional probablity matrix containing the fraction of
    examples in every class, labeled as every other class.
    Diagonal terms are not noise rates, but are consistency P(s=k|y=k)
    Assumes columns of noise_matrix sum to 1'''
  
  def clip_noise_rate_range(noise_rate):
    '''Clip noise rate P(s=k'|y=k) or P(y=k|s=k')
    into proper range [0,1)'''
    return min(max(noise_rate, 0.0), 0.9999)
  
  # Vectorize clip_noise_rate_range for efficiency with np.arrays.  
  vectorized_clip = np.vectorize(clip_noise_rate_range)
  
  # Preserve because diagonal entries are not noise rates.
  diagonal = np.diagonal(noise_matrix)
  
  # Clip all noise rates (efficiently).
  noise_matrix = vectorized_clip(noise_matrix)
  
  # Put unmodified diagonal back.
  np.fill_diagonal(noise_matrix, diagonal)
  
  # Re-normalized noise_matrix so that columns sum to one.
  noise_matrix = noise_matrix / noise_matrix.sum(axis=0)
  
  return noise_matrix


def clip_values(x, low = 0.0, high = 1.0, new_sum = None):
  '''Clip all values in p to range [low,high].
  Preserves sum of x.

  Parameters
  ----------
    
  x : np.array 
    An array / list of values to be clipped.
    
  low : float
    values in x greater than 'low' are clipped to this value
    
  high : float
    values in x greater than 'high' are clipped to this value
    
  new_sum : float
    normalizes x after clipping to sum to new_sum
    
  Returns
  -------
  
  x : np.array
    A list of clipped values, summing to the same sum as x.'''
  
  def clip_range(a, low = low, high = high):
    '''Clip a into range [low,high]'''
    return min(max(a, low), high)
  
  # Vectorize clip_range for efficiency with np.arrays.  
  vectorized_clip = np.vectorize(clip_range)
  
  # Store previous sum
  prev_sum = sum(x) if new_sum is None else new_sum
  
  # Clip all values (efficiently).
  x = vectorized_clip(x)
  
  # Re-normalized values to sum to previous sum.
  x = x * prev_sum / float(sum(x))
  
  return x


def value_counts(x):
  '''Returns an np.array of shape (K, 1), with the
  value counts for every unique item in the labels list/array, 
  where K is the number of unique entries in labels.
  
  Why this matters? Here is an example:
    x = [np.random.randint(0,100) for i in range(100000)]

    %timeit np.bincount(x)
    --Result: 100 loops, best of 3: 3.9 ms per loop

    %timeit np.unique(x, return_counts=True)[1]
    --Result: 100 loops, best of 3: 7.47 ms per loop

  Parameters
  ----------
  
  x : list or np.array (one dimensional)
    A list of discrete objects, like lists or strings, for
    example, class labels 'y' when training a classifier.
    e.g. ["dog","dog","cat"] or [1,2,0,1,1,0,2]
'''
  if type(x[0]) is int and (np.array(x) >= 0).all():
    return np.bincount(x)
  else:
    return np.unique(x, return_counts=True)[1]  


# In[ ]:

# # Private generic helper functions.

# def _mean_without_nan_inf(arr, replacement = None):
#   '''Private helper method for computing the mean
#   of a 1D numpy array or iterable by replacing NaN and inf
#   values with a replacement value or ignore those values
#   if replacement = None.

#   Parameters 
#   ----------
#   arr : iterable (list or 1D np.array)
#     Any 1-dimensional iterable that may contain NaN or inf values.

#   replacement : float
#     Replace NaN and inf values in arr with this value.
#   '''
#   if replacement is not None:
#     return np.mean(
#       [replacement if math.isnan(x) or math.isinf(x) else x for x in arr]
#     )
  
#   x_real = [x for x in arr if not math.isnan(x) and not math.isinf(x)]
  
#   if len(x_real) == 0:
#       raise ValueError("All values are np.NaN or np.inf. If you are" \
#         "using this function in the context of Rank Pruning:\n\n" \
#         "\tFor Rank Pruning: All noise_rate estimates are np.NaN or np.inf" \
#         "for one of the noise rates in the noise_matrix. Check that" \
#         "threshold values are not too extreme (near 1 or 0), " \
#         "resulting in division by zero.")
#   else:
#     return np.mean(x_real)


# In[ ]:

# Useful functions for multiclass learning with noisy labels

def generate_noisy_labels(y, noise_matrix, verbose=False):  
  '''Generates noisy labels s (shape (N, 1)) from perfect labels y,
  'exactly' yielding the provided noise_matrix between s and y.
  
  Parameters
  ----------

  y : np.array (shape (N, 1))
    Perfect labels, without any noise. Contains K distinct natural number
    classes, e.g. 0, 1,..., K-1
    
  noise_matrix : np.array of shape (K, K), K = number of classes 
    A conditional probablity matrix of the form P(s=k_s|y=k_y) containing
    the fraction of examples in every class, labeled as every other class.
    Assumes columns of noise_matrix sum to 1.'''
  
  # Number of classes
  K = len(noise_matrix)
  
  # Compute p(y=k)
  py = value_counts(y) / float(len(y))
  
  # Generate s
  count_joint = (noise_matrix * py * len(y)).round().astype(int) # count(s and y)
  s = np.array(y)
  for k_s in range(K):
    for k_y in range(K):
      if k_s != k_y:
        s[np.random.choice(np.where((s==k_y)&(y==k_y))[0], count_joint[k_s][k_y], replace=False)] = k_s

  # Compute the actual noise matrix induced by s
  from sklearn.metrics import confusion_matrix
  counts = confusion_matrix(s, y).astype(float)
  new_noise_matrix = counts / counts.sum(axis=0)

  # Validate that s indeed produces the correct noise_matrix (or close to it)
  if np.linalg.norm(noise_matrix - new_noise_matrix) > 1:
    raise ValueError("s does not yield the same noise_matrix. " +                     "The difference in norms is " + str(np.linalg.norm(noise_matrix - new_noise_matrix)))

  return s  


def estimate_pu_f1(s, prob_s_eq_1):
  '''    Computes Claesen's estimate of f1 in the pulearning setting.
    
    Parameters
    ----------
    s : iterable (list or np.array)
      Binary label (whether each element is labeled or not) in pu learning.
      
    prob_s_eq_1 : iterable (list or np.array)
      The probability, for each example, whether it is s==1 P(s==1|x)
      
    Output (float)
    ------
    Claesen's estimate for f1 in the pulearning setting.'''
  
  pred = prob_s_eq_1 >= 0.5
  true_positives = sum((np.array(s) == 1) & (np.array(pred) == 1))
  all_positives = sum(s)
  recall = true_positives / float(all_positives)
  frac_positive_predictions = sum(pred) / float(len(s))
  return recall ** 2 / frac_positive_predictions if frac_positive_predictions != 0 else np.nan


def compute_confusion_noise_rate_matrix(y, s):
  '''Implements a confusion matrix assuming y as true classes
  and s as noisy (or sometimes predicted) classes.
  
  Results are identical (and similar computation time) to: 
    "from sklearn.metrics import confusion_matrix"
    
  However, this function avoids the dependency on sklearn.'''
  
  K = len(np.unique(y)) # Number of classes 
  result = np.zeros((K, K))
  
  for i in range(len(y)):
    result[y[i]][s[i]] += 1
    
  return result.astype(float) / result.sum(axis=0)  

