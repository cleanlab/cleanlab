
# coding: utf-8

# ## Confident Learning Utilties
# 
# #### Contains ancillarly helper functions used throughout this package.

# In[ ]:


from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
import numpy as np


# In[ ]:


def assert_inputs_are_valid(X, s, psx = None): # pragma: no cover
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


# In[ ]:


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


# In[ ]:


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
        e.g. ["dog","dog","cat"] or [1,2,0,1,1,0,2]'''
    
    if type(x[0]) is int and (np.array(x) >= 0).all():
        return np.bincount(x)
    else:
        return np.unique(x, return_counts=True)[1] 


# In[ ]:


def estimate_pu_f1(s, prob_s_eq_1):
    '''Computes Claesen's estimate of f1 in the pulearning setting.
    
    Parameters
    ----------
    s : iterable (list or np.array)
      Binary label (whether each element is labeled or not) in pu learning.
      
    prob_s_eq_1 : iterable (list or np.array)
      The probability, for each example, whether it is s==1 P(s==1|x)
      
    Output (float)
    ------
    Claesen's estimate for f1 in the pulearning setting.'''
  
    pred = np.asarray(prob_s_eq_1) >= 0.5
    true_positives = sum((np.asarray(s) == 1) & (np.asarray(pred) == 1))
    all_positives = sum(s)
    recall = true_positives / float(all_positives)
    frac_positive = sum(pred) / float(len(s))
    return recall ** 2 / (2.0 * frac_positive) if frac_positive != 0 else np.nan


def confusion_matrix(y, s):
    '''Implements a confusion matrix assuming y as true classes
    and s as noisy (or sometimes predicted) classes.

    Results are identical (and similar computation time) to: 
        "sklearn.metrics.confusion_matrix"

    However, this function avoids the dependency on sklearn.
    
    Parameters
    ----------
    y : np.array 1d
      Contains non-negative integers 0, 1, 2... Labels are consecutive.
      For example y = [0, 1, 1, 2] is okay.
      But y = [0, 1, 3, 1] is BAD because there is no "2" class.
      
    s : np.array 1d
      Same as y'''
    
    y_classes = np.unique(y)
    s_classes = np.unique(s)
    K_y = len(y_classes) # Number of classes in y
    K_s = len(s_classes) # Number of classes in s    
    mapy = dict(zip(y_classes, range(K_y)))    
    maps = dict(zip(s_classes, range(K_s)))
    
    result = np.zeros((K_y, K_s))

    for i in range(len(y)):
        result[mapy[y[i]]][maps[s[i]]] += 1

    return result.astype(float) / result.sum(axis=0)  


# In[ ]:


def print_square_matrix(
    matrix, 
    left_name = 's', 
    top_name = 'y', 
    title = " A square matrix",
    short_title = 's,y',
):
    '''Pretty prints a matrix. 
    
    Parameters
    ----------
    matrix : np.array
        the matrix to be printed
    left_name : str
        the name of the variable on the left of the matrix
    top_name : str
        the name of the variable on the top of the matrix
    title : str
        Prints this string above the printed square matrix.
    short_title : str
        A short title (6 characters or less) like P(s|y) or P(s,y).'''
    
    short_title = short_title[:6]    
    K = len(matrix) # Number of classes
    # Make sure matrix is 2d array
    if len(np.shape(matrix)) == 1:
        matrix = np.array([matrix])
    print()
    print(title, 'of shape', matrix.shape)
    print(" "+short_title+"".join(['\t'+top_name+'='+str(i) for i in range(K)]))
    print('\t---'*K)
    for i in range(K):
        print(left_name+"="+str(i)+" |\t"+"\t".join([str(z) for z in list(matrix.round(2)[i,:])]))
    print("\tTrace(matrix) =", np.round(np.trace(matrix), 2))
    print()  
    
def print_noise_matrix(noise_matrix):
    '''Pretty prints the noise matrix.'''
    print_square_matrix(
        noise_matrix,
        title=' Noise Matrix (aka Noisy Channel) P(s|y)', 
        short_title = "p(s|y)",
    )
    
def print_inverse_noise_matrix(inverse_noise_matrix):
    '''Pretty prints the inverse noise matrix.'''
    print_square_matrix(
        inverse_noise_matrix, 
        left_name = 'y', 
        top_name = 's', 
        title=' Inverse Noise Matrix P(y|s)',
        short_title = "p(y|s)",
    )
    
def print_joint_matrix(joint_matrix):
    '''Pretty prints the joint label noise matrix.'''
    print_square_matrix(
        joint_matrix,
        title=' Joint Label Noise Distribution Matrix P(s,y)',
        short_title = "p(s,y)",
    )


# In[ ]:


def _python_version_is_compatible(
    warning_str = "pyTorch supports Python version 2.7, 3.5, 3.6, 3.7.",
    warning_already_issued = False,
    list_of_compatible_versions = [2.7, 3.5, 3.6],
):
    '''Helper function for VersionWarning class that issues
    a warning (if a warning has not already been issued),
    whenever the python version is not in the 
    list_of_compatible_versions.
    '''
    
    import sys
    v = sys.version_info[0] + 0.1 * sys.version_info[1]
    if v in list_of_compatible_versions:
        return True
    elif not warning_already_issued:
        import warnings
        warning = '''
        {}
        cleanlab supports Python versions 2.7, 3.4, 3.5, 3.6.
        You are using Python version {}.
        You'll need to use a version compatible with both.'''.format(warning_str, v)
        warnings.warn(warning)
        warning_already_issued = True
    return False


class VersionWarning(object):
    '''Functor that calls _python_version_is_compatible
    and manages the state of the bool variable
    warning_already_issued to make sure the same warning
    is never displayed multiple times. '''
    
    def __init__(self, warning_str, list_of_compatible_versions):
        self.warning_str = warning_str
        self.warning_already_issued = False
        self.list_of_compatible_versions = list_of_compatible_versions
    def is_compatible(self):
        compatible = _python_version_is_compatible(
            self.warning_str,
            self.warning_already_issued,
            self.list_of_compatible_versions,
        )
        if not compatible:
            self.warning_already_issued = True
        return compatible

