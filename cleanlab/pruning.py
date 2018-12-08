
# coding: utf-8

# ## Pruning
# 
# #### Contains methods for estimating the latent indices of all label errors.

# In[ ]:


from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
import numpy as np

from cleanlab.util import value_counts


# In[ ]:


# Leave at least this many examples in each class after
# pruning, regardless if noise estimates are larger.
MIN_NUM_PER_CLASS = 5


# In[ ]:


def get_noise_indices(
    s, 
    psx, 
    inverse_noise_matrix = None,
    confident_joint = None,
    frac_noise = 1.0,
    num_to_remove_per_class = None,
    prune_method = 'prune_by_noise_rate',
    prune_count_method = 'inverse_nm_dot_s',
    converge_latent_estimates = False,
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
        
    confident_joint : np.array (shape (K, K), type int) (default: None)
      A K,K integer matrix of count(s=k, y=k). Estimatesa a confident subset of
      the joint disribution of the noisy and true labels P_{s,y}.
      Each entry in the matrix contains the number of examples confidently
      counted into every pair (s=j, y=k) classes.
  
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
      ***Only set this parameter if prune_method == 'prune_by_class'

    prune_method : str (default: 'prune_by_noise_rate')
      'prune_by_class', 'prune_by_noise_rate', or 'both'. Method used for pruning.
      1. 'prune_by_noise_rate': works by removing examples with *high probability* of 
      being mislabeled for every non-diagonal in the prune_counts_matrix (see pruning.py).
      2. 'prune_by_class': works by removing the examples with *smallest probability* of
      belonging to their given class label for every class.
      3. 'both': Finds the examples satisfying (1) AND (2) and removes their set conjunction. 

    prune_count_method : str (default 'inverse_nm_dot_s')
      Options are 'inverse_nm_dot_s' or 'calibrate_confident_joint'. 
        !DO NOT USE! 'calibrate_confident_joint' if you already know the noise matrix
      and will call .fit(noise_matrix = known_noise_matrix) or
      .fit(inverse_noise_matrix = known_inverse_noise_matrix) because
      'calibrate_confident_joint' will estimate the noise without using this information.
        !IN ALL OTHER CASES! We recommend always using 'calibrate_confident_joint'
      because it is faster and more robust when no noise matrix info is given.
        Determines the method used to estimate the counts of the joint P(s, y) that will 
      be used to determine how many examples to prune
      for every class that are flipped to every other class, as follows:
        if prune_count_method == 'inverse_nm_dot_s':
          prune_count_matrix = inverse_noise_matrix * s_counts # Matrix of counts(y=k and s=l)
        elif prune_count_method == 'calibrate_confident_joint':# calibrate
          prune_count_matrix = confident_joint.T / float(confident_joint.sum()) * len(s)

    converge_latent_estimates : bool (Default: False)
      If true, forces numerical consistency of estimates. Each is estimated
      independently, but they are related mathematically with closed form 
      equivalences. This will iteratively enforce mathematically consistency.'''
  
    # Number of examples in each class of s
    s_counts = value_counts(s)
    # 'ps' is p(s=k)
    ps = s_counts / float(len(s))
    # Number of classes s
    K = len(psx.T)

    # Ensure labels are of type np.array()
    s = np.asarray(s)
    
    # Estimate the number of examples to confidently prune for each (s=j, y=k) pair.
    if (inverse_noise_matrix is None and prune_count_method == 'inverse_nm_dot_s') or        (confident_joint is None and prune_count_method == 'calibrate_confident_joint'):
            from cleanlab.latent_estimation import estimate_py_and_noise_matrices_from_probabilities
            _, _, inverse_noise_matrix, confident_joint = estimate_py_and_noise_matrices_from_probabilities(
                s, 
                psx, 
                converge_latent_estimates=converge_latent_estimates,
            )
    if prune_count_method == 'inverse_nm_dot_s':
        prune_count_matrix = inverse_noise_matrix * s_counts # Matrix of counts(y=k and s=l)
    elif prune_count_method == 'calibrate_confident_joint':
        prune_count_matrix = confident_joint.T / float(confident_joint.sum()) * len(s) # calibrate
    else:
        raise ValueError("prune_count_method should be 'inverse_nm_dot_s' or " + 
            "'calibrate_confident_joint', but '" + prune_count_method + "' was given.")
        
    # Leave at least MIN_NUM_PER_CLASS examples per class.
    prune_count_matrix = keep_at_least_n_per_class(
        prune_count_matrix=prune_count_matrix, 
        n=MIN_NUM_PER_CLASS, 
        frac_noise=frac_noise,
    )
  
    if num_to_remove_per_class is not None:
        # Estimate joint probability distribution over label errors
        psy = prune_count_matrix / np.sum(prune_count_matrix, axis = 1)
        noise_per_s = psy.sum(axis = 1) - psy.diagonal()
        # Calibrate s.t. noise rates sum to num_to_remove_per_class
        tmp = (psy.T * num_to_remove_per_class / noise_per_s).T
        np.fill_diagonal(tmp, s_counts - num_to_remove_per_class)
        prune_count_matrix = np.round(tmp).astype(int)

    # Initialize the boolean mask of noise indices.
    noise_mask = np.zeros(len(psx), dtype=bool)

    # Peform Pruning with threshold probabilities from BFPRT algorithm in O(n)

    if prune_method == 'prune_by_class' or prune_method == 'both':
        for k in range(K):
            if s_counts[k] > MIN_NUM_PER_CLASS: # Don't prune if not MIN_NUM_PER_CLASS
                num2prune = s_counts[k] - prune_count_matrix[k][k]
                # num2keep'th smallest probability of class k for examples with noisy label k
                threshold = np.partition(psx[:,k][s == k], num2prune)[num2prune]
                noise_mask = noise_mask | ((psx[:,k] < threshold) & (s == k))
  
    if prune_method == 'both':
        noise_mask_by_class = noise_mask

    if prune_method == 'prune_by_noise_rate' or prune_method == 'both':
        noise_mask = np.zeros(len(psx), dtype=bool)
        for k in range(K):
            if s_counts[k] > MIN_NUM_PER_CLASS: # Don't prune if not MIN_NUM_PER_CLASS
                for j in range(K):
                    if k!=j: # Only prune for noise rates
                        num2prune = prune_count_matrix[k][j]
                        # num2prune'th largest probability of class k for examples with noisy label j
                        threshold = -np.partition(-psx[:,k][s == j], num2prune)[num2prune]
                        noise_mask = noise_mask | ((psx[:,k] > threshold) & (s == j))
            
    return noise_mask & noise_mask_by_class if prune_method == 'both' else noise_mask


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

