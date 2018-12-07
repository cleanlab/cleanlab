
# coding: utf-8

# In[1]:


from __future__ import print_function, absolute_import, division, unicode_literals, with_statement


# In[2]:


import numpy as np
from cleanlab import noise_generation
import pytest


# In[3]:


seed = 0
np.random.seed(0)


# In[31]:


def test_main_pipeline(
    verbose = False,
    n = 10,
    valid_noise_matrix = True,
):
    trace = 1.5
    py = [0.1, 0.1, 0.2, 0.6]
    K = len(py)
    y = [z for i,p in enumerate(py) for z in [i]*int(p*n)]
    nm = noise_generation.generate_noise_matrix_from_trace(
        K = K,
        trace = trace,
        py = py,
        seed = 0,
        valid_noise_matrix = valid_noise_matrix,
    )
    # Check that trace is what its supposed to be
    assert(abs(trace - np.trace(nm) < 1e-2))
    # Check that sum of probabilities is K
    assert(abs(nm.sum() - K) < 1e-4)
    # Check that sum of each column is 1
    assert(all(abs(nm.sum(axis = 0) - 1) < 1e-4))
    # Check that joint sums to 1.
    assert(abs(np.sum(nm*py) - 1 < 1e-4))
    s = noise_generation.generate_noisy_labels(y, nm, verbose)
    assert(noise_generation.noise_matrix_is_valid(nm, py, verbose))
    
test_main_pipeline()


# In[14]:


def test_main_pipeline_verbose(verbose = True, n = 10):
    test_main_pipeline(verbose = verbose, n = n)


# In[32]:


def test_main_pipeline_many(verbose = False, n = 1000):
    test_main_pipeline(verbose = verbose, n = n)
    
def test_main_pipeline_many_verbose_valid(verbose = True, n = 100):
    test_main_pipeline(verbose, n, valid_noise_matrix = True)
    
def test_main_pipeline_many_valid(verbose = False, n = 100):
    test_main_pipeline(verbose, n, valid_noise_matrix = True)


# In[7]:


def test_main_pipeline_many_verbose(verbose = True, n = 1000):
    test_main_pipeline(verbose = verbose, n = n)


# In[8]:


def test_invalid_inputs_verify():
    
    nm = np.array([
        [0.2, 0.5],
        [0.8, 0.5],
    ])
    py = [0.1, 0.8]
    assert(not noise_generation.noise_matrix_is_valid(nm, py))
    
    nm = np.array([
        [0.2, 0.5],
        [0.8, 0.4],
    ])
    py = [0.1, 0.9]
    assert(not noise_generation.noise_matrix_is_valid(nm, py))
    
    py = [0.1, 0.8]
    assert(not noise_generation.noise_matrix_is_valid(nm, py))


# In[9]:


def test_invalid_matrix():    
    nm = np.array([
        [0.1, 0.9],
        [0.9, 0.1],
    ])
    py = [0.1, 0.9]
    assert(not noise_generation.noise_matrix_is_valid(nm, py))


# In[12]:


def test_trace_less_than_1_error(trace = 0.5):
    try:
        noise_generation.generate_noise_matrix_from_trace(3, trace)
    except ValueError as e:
        assert('trace > 1' in str(e))
        with pytest.raises(ValueError) as e:
            noise_generation.generate_noise_matrix_from_trace(3, trace)


# In[13]:


def test_trace_equals_1_error(trace = 1):
    test_trace_less_than_1_error(trace)


# In[14]:


def test_valid_no_py_error():
    try:
        noise_generation.generate_noise_matrix_from_trace(
            K = 3, 
            trace = 2,
            valid_noise_matrix = True,
        )
    except ValueError as e:
        assert('py must be' in str(e))
        with pytest.raises(ValueError) as e:            
            noise_generation.generate_noise_matrix_from_trace(
                K = 3, 
                trace = 2,
                valid_noise_matrix = True,
            )


# In[15]:


def test_one_class_error():
    try:
        noise_generation.generate_noise_matrix_from_trace(
            K = 1, 
            trace = 1,
        )
    except ValueError as e:
        assert('must be >= 2' in str(e))
        with pytest.raises(ValueError) as e:            
            noise_generation.generate_noise_matrix_from_trace(
                K = 1, 
                trace = 1,
            )


# In[22]:


def test_two_class_gen_with_trace(valid = True):
    trace = 1.5
    nm = noise_generation.generate_noise_matrix_from_trace(
        K = 2, 
        trace = trace,
        valid_noise_matrix = valid,
    )
    assert(abs(trace - np.trace(nm) < 1e-2))


# In[23]:


def test_two_class_gen_with_trace_not_valid(valid = False):
    test_two_class_gen_with_trace(valid = valid)


# In[48]:


def test_deprecated_warning(verbose = False): 
    K = 3
    with pytest.warns(DeprecationWarning):
        nm = noise_generation.generate_noise_matrix(
            K = K, 
            verbose = verbose,
        )
    assert(abs(nm.sum() - K) < 1e-4)
    assert(all(abs(nm.sum(axis = 0) - 1) < 1e-4))


# In[49]:


def test_deprecated_warning_verbose(verbose = True):    
    test_deprecated_warning(verbose)   


# In[39]:


def test_gen_probs_sum_empty():
    f = noise_generation.generate_n_rand_probabilities_that_sum_to_m
    assert(len(f(n = 0, m = 1)) == 0)


# In[54]:


def test_gen_probs_max_error():    
    f = noise_generation.generate_n_rand_probabilities_that_sum_to_m
    try:
        f(n = 5, m = 1, max_prob = 0.1)
    except ValueError as e:
        assert('max_prob must be greater' in str(e))
        with pytest.raises(ValueError) as e:
            f(n = 5, m = 1, max_prob = 0.1)


# In[58]:


def test_gen_probs_min_error():    
    f = noise_generation.generate_n_rand_probabilities_that_sum_to_m
    try:
        f(n = 5, m = 1, min_prob = 0.9)
    except ValueError as e:
        assert('min_prob must be less' in str(e))
        with pytest.raises(ValueError) as e:
            f(n = 5, m = 1, min_prob = 0.9)


# In[61]:


def test_gen_probs_min_max_error():    
    f = noise_generation.generate_n_rand_probabilities_that_sum_to_m
    min_prob = 0.6
    max_prob = 0.4
    try:
        f(n = 5, m = 1, min_prob = min_prob, max_prob = max_prob)
    except ValueError as e:
        assert('min_prob must be less' in str(e))
        with pytest.raises(ValueError) as e:
            f(n = 5, m = 1, min_prob = min_prob, max_prob = max_prob)


# In[64]:


def test_balls_zero():    
    f = noise_generation.randomly_distribute_N_balls_into_K_bins
    K = 3
    result = f(N=0,K=K)
    assert(len(result) == K)
    assert(sum(result) == 0)


# In[81]:


def test_balls_params():    
    f = noise_generation.randomly_distribute_N_balls_into_K_bins
    N = 10
    K = 10
    for mx in [None, 1, 2, 3]:
        for mn in [None, 1, 2, 3]:
            r = f(
                N = N,
                K = K,
                max_balls_per_bin = mx,
                min_balls_per_bin = mn,
            )
            assert(sum(r) == K)
            assert(min(r) <= (K if mn is None else mn))
            assert(len(r) == K)

