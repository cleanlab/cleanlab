
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

