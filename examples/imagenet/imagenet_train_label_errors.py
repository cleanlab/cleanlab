#!/usr/bin/env python
# coding: utf-8

# # Find label errors in ImageNet train set using confident learning
# 
# ### Note this code assumes that you've already computed psx -- the predicted probabilities for all examples in the training set using four-fold cross-validation. If you have no done that you will need to use `imagenet_train_crossval.py` to do this!
# 

# In[1]:


# These imports enhance Python2/3 compatibility.
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement


# In[2]:


# Common ML stuff
import numpy as np
from sklearn.metrics import accuracy_score

# Cleanlab
import cleanlab
from cleanlab import baseline_methods

# System modules
from IPython.display import Image, display
import json
import sys

# PyTorch modules
from torchvision.utils import make_grid
from torchvision import datasets

# Plotting
import matplotlib.pyplot as plt
import matplotlib
import torchvision.transforms as transforms
import matplotlib.image as mpimg


# In[3]:


def show(img, savefig=False):
    '''Show a grid of images.'''
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    if savefig:
        plt.savefig('imagenet_figure_32.png', dpi=300, pad_inches=0.0, bbox_inches='tight')
    
def make3d(img_arr):
    '''Reshape images to include a third dimensions for numpy.'''
    img_arr = np.asarray(img_arr)
    return np.stack((img_arr,)*3, -1) if len(img_arr.shape) < 3 else img_arr


# In[4]:


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


# In[5]:


# Load psx, labels, and image locations
data_dir = "/datasets/datasets/imagenet/"
traindir = data_dir + "train/"
psx = np.load(data_dir + "imagenet__train__model_resnet50__pyx.npy")
imgs, labels = [list(z) for  z in zip(*datasets.ImageFolder(traindir).imgs)]
labels = np.array(labels, dtype=int)
print('Overall accuracy: {:.2%}'.format(accuracy_score(labels, psx.argmax(axis = 1))))


# In[6]:


# This takes ~3 minutes on a 20-thread processor for ImageNet train set.
already_computed = True
if already_computed:
    label_errors_bool = ~np.load("/home/cgn/masks/imagenet_train_bool_mask.npy")
else:
    label_errors_bool = cleanlab.pruning.get_noise_indices(
        s = labels,
        psx = psx,
        prune_method = 'prune_by_noise_rate',
        sorted_index_method=None,
    )
#     np.save("imagenet_train_bool_mask.npy", ~label_errors_bool) # Store false for errors


# In[13]:


label_errors_idx = cleanlab.pruning.order_label_errors(
    label_errors_bool = label_errors_bool,
    psx = psx,
    labels = labels,
    sorted_index_method = 'normalized_margin',
)


# In[14]:


label_errors_idx


# In[15]:


# Number of errors to store labels and images for.
# Here we only print the first three, but we will print all of them later on.
num2print = 32

# Import this here because this version of Image is used to print a single image
from IPython.display import Image, display
fns = []
given = []
pred = []
for i, idx in enumerate(label_errors_idx[:num2print]):
    fn = imgs[idx]
    fns.append(fn)
    given.append(simple_labels[labels[idx]])
    pred.append(simple_labels[np.argmax(psx[idx])])
    # Print out the first 3 examples
    if i < 10:
        print("Given:", given[-1].upper()) 
        print("Guess:", pred[-1].upper())
        print(fn.split("/")[-1])
        sys.stdout.flush()
        display(Image(filename=fn))
        print("\n")


# In[9]:


# Now re-import Image from PIL. PIL Image module displays grids of images.
from PIL import Image
num2print = 32
image_size = 333
save_figure = False

rs = transforms.Resize((333,333))

plt.figure(figsize=(50,40))
fns = [imgs[i] for i in label_errors_idx[:num2print]]  # Filenames of errors
imglist = [transforms.ToTensor()(make3d(rs(Image.open(fn)))) for fn in fns]
show(make_grid(imglist, padding=1, normalize=True), savefig=save_figure)


# In[10]:


# Print rows of images with captions

def padtext(s, l = 27): # 27 works well
    return s + " " * (l - len(s))

rows2print = 4
num_per_row = int(num2print / rows2print)
for i in range(rows2print):
    plt.figure(figsize=(30,10))
    imglist = [transforms.ToTensor()(make3d(rs(Image.open(fn)))) for fn in fns[
        num_per_row * i : num_per_row * i + num_per_row]]
    show(make_grid(imglist, padding=1, normalize=True))
    plt.show()
    for z in given[num_per_row * i :num_per_row * i + num_per_row]:
        item = " GIVEN: " + z
        print(padtext(item), end = "")
    print()
    for z in pred[num_per_row * i :num_per_row * i + num_per_row]:
        item = " GUESS: " + z
        print(padtext(item), end = "")
    print()
    for z in fns[num_per_row * i:num_per_row * i + num_per_row]:
        item = " " + fn.split("/")[-1]
        print(padtext(item), end = "")


# # This concludes the tutorial for displaying imagenet label errors. The code below is for reproducing the label errors in the confiding learning paper: https://arxiv.org/abs/1911.00068

# ## Create multiple files storing the indices of errors for the top 20% of errors, top 40% of errors, etc.

# In[60]:


# Take the opposite of the stored file (errors should be true, not false)
label_errors_bool = ~np.load("imagenet_train_bool_mask.npy")


# In[61]:


label_errors_idx = cleanlab.pruning.order_label_errors(
    label_errors_bool = label_errors_bool,
    psx = psx,
    labels = labels,
    sorted_index_method = 'normalized_margin',
)


# In[62]:


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
    np.save("imagenet_train_bool_mask__fraction_{}.npy".format(amt), ~bool_mask)
    
# Verify written files
for i in range(1, 5):
    amt = str(100 * i // 5)
    end_idx = len(label_errors_idx) * i // 5
    truth = np.array(sorted(label_errors_idx[:end_idx]))
    us = np.array([i for i, b in enumerate(~np.load("imagenet_train_bool_mask__fraction_{}.npy".format(amt))) if b])
    assert(all(truth == us))


# ## Create these files for various methods of finding label errors

# In[5]:


# confident joint only method for getting label errors
label_error_mask = np.zeros(len(labels), dtype=bool)
label_error_indices = cleanlab.latent_estimation.compute_confident_joint(
    labels, psx, return_indices_of_off_diagonals=True
)[1]
for idx in label_error_indices:
    label_error_mask[idx] = True
label_errors_bool_cj_only = label_error_mask

label_errors_bool_both = cleanlab.pruning.get_noise_indices(
    s = labels,
    psx = psx,
    prune_method = 'both',
    sorted_index_method=None,
)

label_errors_bool_pbc = cleanlab.pruning.get_noise_indices(
    s = labels,
    psx = psx,
    prune_method = 'prune_by_class',
    sorted_index_method=None,
)

label_errors_bool_pbnr = cleanlab.pruning.get_noise_indices(
    s = labels,
    psx = psx,
    prune_method = 'prune_by_noise_rate',
    sorted_index_method=None,
)

label_errors_bool_argmax = baseline_methods.baseline_argmax(psx, labels)


# In[7]:


le_idx_both = cleanlab.pruning.order_label_errors(label_errors_bool_both, psx, labels)
le_idx_pbc = cleanlab.pruning.order_label_errors(label_errors_bool_pbc, psx, labels)
le_idx_pbnr = cleanlab.pruning.order_label_errors(label_errors_bool_pbnr, psx, labels)
le_idx_argmax = cleanlab.pruning.order_label_errors(label_errors_bool_argmax, psx, labels)
le_idx_cj_only = cleanlab.pruning.order_label_errors(label_errors_bool_cj_only, psx, labels)


# In[9]:


for key, label_errors_idx in {
#     'both': le_idx_both,
#     'argmax': le_idx_argmax,
#     'cj_only': le_idx_cj_only,
    'cl_pbnr': le_idx_pbnr,
    'cl_pbc': le_idx_pbc,
}.items():
    # Creates seperate files for the top 20% errors, 40% errors,...
    for i in range(1,6):
        # Prepare arguments
        amt = str(100 * i // 5)
        end_idx = len(label_errors_idx) * i // 5
        partial_errors_idx = label_errors_idx[:end_idx]
        # Create new bool mask
        bool_mask = np.zeros(len(label_errors_bool_both), dtype=bool)
        bool_mask[partial_errors_idx] = True
        # Validate
        assert(all(np.array([i for i, b in enumerate(bool_mask) if b]) == sorted(partial_errors_idx)))
        print(amt, end_idx)
        np.save("/home/cgn/masks/imagenet_train_bool_{}_mask__fraction_{}.npy".format(key, amt), ~bool_mask)

    # Verify written files
    for i in range(1, 6):
        amt = str(100 * i // 5)
        end_idx = len(label_errors_idx) * i // 5
        truth = np.array(sorted(label_errors_idx[:end_idx]))
        us = np.array([i for i, b in enumerate(~np.load("/home/cgn/masks/imagenet_train_bool_{}_mask__fraction_{}.npy".format(key, amt))) if b])
        assert(all(truth == us))

