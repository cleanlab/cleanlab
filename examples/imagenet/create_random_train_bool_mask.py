
# coding: utf-8

# # This is the script used to determine the random examples to prune in the benchmarking figure (dotted red line) in the paper. This has no algorihtmic purpose but is included for reproducibility.

# In[1]:


import numpy as np


# In[2]:


import numpy as np
from torchvision import datasets
from sklearn.metrics import accuracy_score
import cleanlab
from IPython.display import Image, display
import json


# In[3]:


# urllib2 for python2 and python3
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen
    
# simple label names for ImageNet
url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
f = urlopen(url)
simple_labels = json.loads('\n'.join(i.decode('ascii') for i in f.readlines()))


# In[8]:


# Load psx, labels, and image locations
data_dir = "/home/cgn/datasets/datasets/imagenet/"
traindir = data_dir + "train/"
# psx = np.load(data_dir + "imagenet__train__model_resnet50__pyx.npy")
imgs, labels = [list(z) for  z in zip(*datasets.ImageFolder(traindir).imgs)]
labels = np.array(labels, dtype=int)
# print('Overall accuracy: {:.2%}'.format(accuracy_score(labels, psx.argmax(axis = 1))))


# In[ ]:


# Creates seperate files for the top 20% errors, 40% errors,...
for i in range(1,5):
    # Prepare arguments
    amt = str(100 * i // 5)
    end_idx = len(label_errors_idx) * i // 5
    partial_errors_idx = label_errors_idx[:end_idx]
    # Create new bool mask
    bool_mask = np.zeros(len(label_errors_bool), dtype=bool)
    bool_mask[partial_errors_idx] = True
    # Validate
    assert(all(np.array([i for i, b in enumerate(bool_mask) if b]) == sorted(partial_errors_idx)))
    print(amt, end_idx)
    np.save("imagenet_train_RANDOM_bool_mask__fraction_{}.npy".format(amt), ~bool_mask)
    
# Verify written files
for i in range(1, 5):
    amt = str(100 * i // 5)
    end_idx = len(label_errors_idx) * i // 5
    truth = np.array(sorted(label_errors_idx[:end_idx]))
    us = np.array([i for i, b in enumerate(~np.load("imagenet_train_RANDOM_bool_mask__fraction_{}.npy".format(amt))) if b])
    assert(all(truth == us))


# In[18]:


# Take the opposite of the stored file (errors should be true, not false)
label_errors_bool = ~np.load("imagenet_train_bool_mask.npy")
num_errors = sum(label_errors_bool)


# In[30]:


for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
    mask = np.ones(len(labels), dtype = bool)
    selection = np.random.choice(len(labels), int(num_errors * frac), replace = False)
    mask[selection] = False
    np.save("imagenet_train_RANDOM_bool_mask__fraction_{}.npy".format(frac), mask)
    assert(len(label_errors_bool) - int(num_errors * frac) == sum(mask))
    print(sum(mask))


# In[28]:



assert(len(label_errors_bool) - int(num_errors * 0.2) == sum(mask))

