
# coding: utf-8

# ## Latent Algebra
# 
# #### Contains mathematical functions relating the latent terms, $p(s), P_{s \vert y}, P_{y \vert s}, p(y)$, etc. together. For every function here, if the inputs are exact, the output is guaranteed to be exact. Every function herein is the computational equivalent of a mathematical equation having a closed, exact form. If the inputs are inexact, the error will of course propagate.

from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
import numpy as np

from cleanlab.util import value_counts, clip_values, clip_noise_rates
import warnings


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

    # For loop based implementation
    
    # Number of classes
    K = len(py)

    # 'ps' is p(s=k) = noise_matrix * p(y=k)
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

    joint = noise_matrix * py
    ps = joint.sum(axis = 1) if ps is None else ps
    inverse_noise_matrix = joint.T / ps
    
    # Clip inverse noise rates P(y=k_s|y=k_y) into proper range [0,1)
    return clip_noise_rates(inverse_noise_matrix)


def compute_noise_matrix_from_inverse(ps, inverse_noise_matrix, py = None):
    '''Compute the noise matrix P(s=k_s|y=k_y).
    
    # For loop based implementation
    
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
  
    joint = (inverse_noise_matrix * ps).T
    py = joint.sum(axis = 0) if py is None else py
    noise_matrix = joint / py
  
    # Clip inverse noise rates P(y=k_y|y=k_s) into proper range [0,1)
    return clip_noise_rates(noise_matrix)

  
def compute_py(ps, noise_matrix, inverse_noise_matrix, py_method = 'cnt', y_count = None):
    '''Compute py := P(y=k) from ps := P(s=k), noise_matrix, and inverse noise matrix.

    This method is ** ROBUST ** when py_method = 'cnt'
    It may work well even when the noise matrices are estimated
    poorly by using the diagonals of the matrices
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
        
    py_method : str (Options: ["cnt", "eqn", "marginal", "marginal_ps"])
        How to compute the latent prior p(y=k). Default is "cnt" as it often
        works well even when the noise matrices are estimated poorly by using
        the matrix diagonals instead of all the probabilities.
        
    y_count : np.array (shape (K, ) or (1, K)) 
        The marginal counts of the confident joint (like cj.sum(axis = 0))

    Output
    ------

    py : np.array (shape (K, ) or (1, K))
        The fraction (prior probability) of each observed, noisy class label, P(y = k).'''
  
    if len(np.shape(ps)) > 2 or (len(np.shape(ps)) == 2 and np.shape(ps)[0] != 1):
        w = 'Input parameter np.array ps has shape ' + str(np.shape(ps))
        w += ', but shape should be (K, ) or (1, K)'
        warnings.warn(w)
        
    if py_method == 'marginal' and y_count is None:
        err = 'py_method == "marginal" requires y_count, but y_count is None.'
        err += ' Provide parameter y_count.'
        raise ValueError(err)
    
    if py_method == 'cnt': 
        # Computing py this way avoids dividing by zero noise rates.
        # More robust bc error est_p(y|s) / est_p(s|y) ~ p(y|s) / p(s|y) 
        py = inverse_noise_matrix.diagonal() / noise_matrix.diagonal() * ps
        # Equivalently,
        # py = (y_count / s_count) * ps
    elif py_method == 'eqn':
        py = np.linalg.inv(noise_matrix).dot(ps)
    elif py_method == 'marginal':
        py = y_count / float(sum(y_count))
    elif py_method == 'marginal_ps':
        py = np.dot(inverse_noise_matrix, ps)
    else:
        err = 'py_method {}'.format(py_method)
        err += ' should be in [cnt, eqn, marginal, marginal_ps]'
        raise ValueError(err)
    
    # Clip py (0,1), .s.t. no class should have prob 0, hence 1e-5
    py = clip_values(py, low=1e-5, high=1.0, new_sum = 1.0)  
    return py


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
        raise ValueError("Input parameter np.array 'psx' has shape " + str(np.shape(psx)) +
                         ", but shape should be (N, K)")
  
    pyx = psx * inverse_noise_matrix.diagonal() / noise_matrix.diagonal()
    # Make sure valid probabilites that sum to 1.0
    return np.apply_along_axis(
        func1d=clip_values, 
        axis=1, 
        arr=pyx,
        **{"low":0.0, "high":1.0, "new_sum":1.0}
    )