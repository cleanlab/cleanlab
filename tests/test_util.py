
# coding: utf-8

# In[1]:


from __future__ import print_function, absolute_import, division, unicode_literals, with_statement


# In[2]:


from cleanlab import util
import numpy as np


# In[3]:


noise_matrix = np.array([
    [1.0, 0.0, 0.2],
    [0.0, 0.7, 0.2],
    [0.0, 0.3, 0.6]
])

noise_matrix_2 = np.array([
    [1.0, 0.3],
    [0.0, 0.7],
])

joint_matrix = np.array([
    [0.1, 0.0, 0.1],
    [0.1, 0.1, 0.1],
    [0.2, 0.1, 0.2]
])

joint_matrix_2 = np.array([
    [0.2, 0.3],
    [0.4, 0.1],
])

single_element = np.array([1])


# In[4]:


def test_print_inm():
    for m in [noise_matrix, noise_matrix_2, single_element]:
        util.print_inverse_noise_matrix(m)
    assert(True)


# In[5]:


def test_print_joint():
    for m in [joint_matrix, joint_matrix_2, single_element]:
        util.print_joint_matrix(m)
    assert(True)


# In[6]:


def test_print_square():
    for m in [noise_matrix, noise_matrix_2, single_element]:
        util.print_square_matrix(noise_matrix)
    assert(True)


# In[7]:


def test_print_noise_matrix():
    for m in [noise_matrix, noise_matrix_2, single_element]:
        util.print_noise_matrix(noise_matrix)
    assert(True)


# In[14]:


def test_pu_f1():
    s = [1, 1, 1, 0, 0, 0]
    p = [1, 1, 1, 0, 0, 0]
    assert(abs(util.estimate_pu_f1(s, p) - 1) < 1e-4)


# In[22]:


def test_value_counts_str():
    r = util.value_counts(['a','b','a'])
    assert(all(np.array([2, 1]) - r < 1e-4))


# In[43]:


def test_pu_remove_noise():
    nm = np.array([
        [0.9, 0.0, 0.0],
        [0.0, 0.7, 0.4],
        [0.1, 0.3, 0.6]
    ])
    r = util.remove_noise_from_class(nm, 0)
    assert(np.all(r - nm < 1e-4))

