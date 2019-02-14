#!/usr/bin/env python
# coding: utf-8

# In[1]:


# These imports enhance Python2/3 compatibility.
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement


# In[2]:


import cleanlab
import numpy as np
import torch

# For visualizing images of label errors
from PIL import Image
from torchvision import datasets
from matplotlib import pyplot as plt

# urllib2 for python2 and python3
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen


# In[3]:


# CHANGE THIS TO CHANGE EXPERIMENT
# pyx_file = 'imagenet_val_out.npy' # NO FINE TUNING
pyx_file = 'imagenet_val_out_cv_10fold.npy' # fine tuned with 10fold cv

# where imagenet dataset is located
data_dir = '/media/ssd/datasets/datasets/imagenet/val/'
# Stored results directory
pyx_dir = '/media/ssd/datasets/pyx/imagenet/'

# Load in data
with open(pyx_dir + 'imagenet_val_out_cv_10fold.npy', 'rb') as f:
    out = np.load(f)
with open(pyx_dir + 'imagenet_val_labels.npy', 'rb') as f:
    labels = np.load(f)
pyx = torch.nn.functional.softmax(torch.from_numpy(out), dim = 1).numpy()


# In[4]:


url = 'https://gist.githubusercontent.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57/'
url += 'raw/aa66dd9dbf6b56649fa3fab83659b2acbf3cbfd1/map_clsloc.txt'
with urlopen(url) as f:
    lines = [x.decode('utf-8') for x in f.readlines()]    
    nid2name = dict([(l.split(" ")[0], l.split(" ")[2][:-1]) for l in lines])
    
dataset = datasets.ImageFolder(data_dir)
nid2idx = dataset.class_to_idx
idx2nid = {v: k for k, v in nid2idx.items()}
name2nid = {v: k for k, v in nid2name.items()}
idx2name = {k: nid2name[v] for k, v in idx2nid.items()}


# In[5]:


cj = cleanlab.latent_estimation.estimate_confident_joint_from_probabilities(labels, pyx)
py, nm, inv = cleanlab.latent_estimation.estimate_latent(cj, labels)


# In[6]:


cj_non_diag = cj - np.eye(len(cj)) * cj.diagonal()
largest_non_diag_raveled = np.argsort(cj_non_diag.ravel())[::-1]
largest_non_diag = np.unravel_index(largest_non_diag_raveled, cj_non_diag.shape)
largest_non_diag = list(zip(*(list(z) for z in largest_non_diag)))


# In[7]:


for i,j in largest_non_diag[:100]:
    print(cj[i,j], "|", idx2nid[i], idx2name[i], "|",  idx2nid[j], idx2name[j])

