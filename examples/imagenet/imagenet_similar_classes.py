
# coding: utf-8

# In[1]:


# These imports enhance Python2/3 compatibility.
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement


# In[4]:


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
    
# where imagenet dataset is located
train_dir = '/datasets/datasets/imagenet/val/'


# In[5]:


# Set-up name mapping for ImageNet train data
url = 'https://gist.githubusercontent.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57/'
url += 'raw/aa66dd9dbf6b56649fa3fab83659b2acbf3cbfd1/map_clsloc.txt'
with urlopen(url) as f:
    lines = [x.decode('utf-8') for x in f.readlines()]    
    nid2name = dict([(l.split(" ")[0], l.split(" ")[2][:-1]) for l in lines])
    
dataset = datasets.ImageFolder(train_dir)
nid2idx = dataset.class_to_idx
idx2nid = {v: k for k, v in nid2idx.items()}
name2nid = {v: k for k, v in nid2name.items()}
idx2name = {k: nid2name[v] for k, v in idx2nid.items()}


# ## Analyze the train set on ImageNet

# In[6]:


# CHANGE THIS TO CHANGE EXPERIMENT
# pyx_file = 'imagenet_val_out.npy' # NO FINE TUNING
pyx_file = 'imagenet__train__model_resnet50__pyx.npy' # trained from scratch with 10fold cv

# where imagenet dataset is located
train_dir = '/datasets/datasets/imagenet/train/'
# Stored results directory
pyx_dir = '/datasets/cgn/pyx/imagenet/'

# Load in data
pyx = np.load(pyx_dir + pyx_file)
imgs, labels = [list(z) for  z in zip(*datasets.ImageFolder(train_dir).imgs)]
labels = np.array(labels, dtype=int)


# In[38]:


cj = cleanlab.latent_estimation.estimate_confident_joint_from_probabilities(labels, pyx)


# In[47]:


joint = cleanlab.latent_estimation.estimate_joint(cj, labels, pyx)
joint_non_diag = joint - np.eye(len(joint)) * joint.diagonal()


# In[8]:


cj_non_diag = cj - np.eye(len(cj)) * cj.diagonal()
largest_non_diag_raveled = np.argsort(cj_non_diag.ravel())[::-1]
largest_non_diag = np.unravel_index(largest_non_diag_raveled, cj_non_diag.shape)
largest_non_diag = list(zip(*(list(z) for z in largest_non_diag)))


# In[69]:


# Checks that joint correctly has rows that are p(s)
assert(all(joint.sum(axis = 1) - np.bincount(labels) / len(labels) < 1e-4))


# In[123]:


print("Index of '{}' in sorted diagonal of cj: ".format(class_name), end = "")
print([nid2name[idx2nid[i]] for i in cj.diagonal().argsort()].index(class_name))

print("Index of '{}' in sorted diagonal of joint: ".format(class_name), end = "")
print([nid2name[idx2nid[i]] for i in joint.diagonal().argsort()].index(class_name))

print("Index of '{}' in sorted most noisy classes in cj: ".format(class_name), end = "")
print([nid2name[idx2nid[i]] for i in np.argsort(cj_non_diag.sum(axis = 0))[::-1]].index(class_name))

print("Index of '{}' in sorted most noisy classes in joint: ".format(class_name), end = "")
print([nid2name[idx2nid[i]] for i in np.argsort(joint_non_diag.sum(axis = 0))[::-1]].index(class_name))

print("Index of '{}' in sorted most noisy true classes in cj: ".format(class_name), end = "")
print([nid2name[idx2nid[i]] for i in np.argsort(cj_non_diag.sum(axis = 1))[::-1]].index(class_name))

print("Index of '{}' in sorted most noisy true classes in joint: ".format(class_name), end = "")
print([nid2name[idx2nid[i]] for i in np.argsort(joint_non_diag.sum(axis = 1))[::-1]].index(class_name))

idx = cj.diagonal().argmin()
print("Least confident class by diagonal of cj:", nid2name[idx2nid[idx]], idx)
idx = joint.diagonal().argmin()
print("Least confident class by diagonal of joint:", nid2name[idx2nid[idx]], idx)
idx = cj_non_diag.sum(axis = 0).argmax()
print("Least confident class by max sum of row of non-diagonal elements of cj:", nid2name[idx2nid[idx]], idx)
idx = joint_non_diag.sum(axis = 1).argmax()
print("Least confident class by max sum of column of non-diagonal elements of cj:", nid2name[idx2nid[idx]], idx)
print('Largest noise rate:', [(nid2name[idx2nid[z]], z) for z in largest_non_diag[0]])


# In[17]:


for i,j in largest_non_diag[:30]:
    print(int(round(cj[i,j])), "|", idx2nid[i], idx2name[i], "|",  idx2nid[j], idx2name[j])


# In[129]:


print("Top 30 row sums in confident joint (most noisy classes):\n")
[(idx2nid[i], idx2name[i]) for i in np.argsort(cj_non_diag.sum(axis = 0))[::-1][:30]]


# # Analye the validation set on ImageNet

# In[5]:


# CHANGE THIS TO CHANGE EXPERIMENT
# pyx_file = 'imagenet_val_out.npy' # NO FINE TUNING
pyx_file = 'imagenet_val_out_cv_10fold.npy' # fine tuned with 10fold cv

# where imagenet dataset is located
val_dir = '/datasets/datasets/imagenet/val/'
# Stored results directory
pyx_dir = '/datasets/cgn/pyx/imagenet/'

# Load in data
with open(pyx_dir + 'imagenet_val_out_cv_10fold.npy', 'rb') as f:
    out = np.load(f)
with open(pyx_dir + 'imagenet_val_labels.npy', 'rb') as f:
    labels = np.load(f)
pyx = torch.nn.functional.softmax(torch.from_numpy(out), dim = 1).numpy()


# In[7]:


# set up mapping for imagenet validation data
url = 'https://gist.githubusercontent.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57/'
url += 'raw/aa66dd9dbf6b56649fa3fab83659b2acbf3cbfd1/map_clsloc.txt'
with urlopen(url) as f:
    lines = [x.decode('utf-8') for x in f.readlines()]    
    nid2name = dict([(l.split(" ")[0], l.split(" ")[2][:-1]) for l in lines])
    
dataset = datasets.ImageFolder(val_dir)
nid2idx = dataset.class_to_idx
idx2nid = {v: k for k, v in nid2idx.items()}
name2nid = {v: k for k, v in nid2name.items()}
idx2name = {k: nid2name[v] for k, v in idx2nid.items()}


# In[8]:


cj = cleanlab.latent_estimation.estimate_confident_joint_from_probabilities(labels, pyx)
py, nm, inv = cleanlab.latent_estimation.estimate_latent(cj, labels)


# In[9]:


cj_non_diag = cj - np.eye(len(cj)) * cj.diagonal()
largest_non_diag_raveled = np.argsort(cj_non_diag.ravel())[::-1]
largest_non_diag = np.unravel_index(largest_non_diag_raveled, cj_non_diag.shape)
largest_non_diag = list(zip(*(list(z) for z in largest_non_diag)))


# In[39]:


print("Top 30 row sums in confident joint (most noisy classes):\n")
[(idx2nid[i], idx2name[i]) for i in np.argsort(cj_non_diag.sum(axis = 1))[::-1][:30]]


# In[127]:


for i,j in largest_non_diag[:30]:
    print(cj[i,j], "|", idx2nid[i], idx2name[i], "|",  idx2nid[j], idx2name[j])

