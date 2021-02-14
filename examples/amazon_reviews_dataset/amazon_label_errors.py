#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cleanlab
import numpy as np
from cleanlab.models.fasttext import data_loader
import pandas as pd
from IPython.display import display, HTML
pd.options.display.max_colwidth = 1000


# In[4]:


# Stored results directory
pyx_dir = '/datasets/cgn/pyx/amazon/'
pyx_file = 'amazon_pyx_cv_3fold.npy'

# # Load pyx
# with open(pyx_dir + pyx_file, 'rb') as f:
#     pyx = np.load(f)
    
# Load pyx 
fn = pyx_dir + 'amazon_pyx_cv__folds_3__epochs_10__lr_1.0__ngram_3__dim_100.npy'
with open(fn, 'rb') as f:
    pyx = np.load(f)


# In[5]:


print("Fetched probabilities for", pyx.shape[0], 'examples and', pyx.shape[1], 'classes.')


# In[7]:


# Get data
s = np.empty(pyx.shape[0], dtype=int)
text = []
loc = '/datasets/datasets/amazon5core/amazon5core.txt'
bs = 1000000
label_map = {'__label__1':0, '__label__3':1, '__label__5':2}
for i, (l, t) in enumerate(data_loader(loc, batch_size=bs)):
    s[bs*i:bs*(i+1)] = [label_map[lab] for lab in l]
    text.append(t)
text = [t for lst in text for t in lst]


# In[8]:


crossval_acc = sum(pyx.argmax(axis=1) == s) / len(s)
print('Cross-val accuracy: {:.2%}'.format(crossval_acc))


# In[120]:


# Estimate the confident joint, a proxy for the joint distribution of label noise.
cj, cj_only_label_error_indices = cleanlab.latent_estimation.compute_confident_joint(
    s, pyx,
    return_indices_of_off_diagonals=True,
)
py, nm, inv = cleanlab.latent_estimation.estimate_latent(cj, s)

# If you want to get label errors using cj_only method.
cj_only_bool_mask = np.zeros(len(s), dtype=bool)
for idx in cj_only_label_error_indices:
    cj_only_bool_mask[idx] = True
label_errors_idx = cleanlab.pruning.order_label_errors(cj_only_bool_mask, pyx, s, sorted_index_method='normalized_margin')


# In[151]:


def print_errors(
    label_errors_idx,
    latex=False,
    num_to_examine=200,
    num_to_view=10,
):
    results = []
    for i, idx in enumerate(label_errors_idx):
        given_label = s[idx]
        if 'sex' in text[idx] or 'serrated' in text[idx]:
            continue  # Don't add profanity to our paper.
        if len(text[idx]) > 30 and len(text[idx]) < 120:
            given_str = ('5' if given_label == 2 else ('3' if given_label == 1 else '1')) + 'cgn'
            results.append({
                'Review': text[idx],
                'Given Label': given_str,
                'CL Guess': str([1,3,5][np.argmax(pyx[idx])]) + 'cgn',

            })
        if i > num_to_examine:
            break
    
    df = pd.DataFrame(results[:num_to_view])
    display(df.set_index('Review', drop=True))
    if latex:
        tex = df.to_latex(index=False).replace('5cgn', '$\star\star\star\star\star$')
        tex = tex.replace('3cgn', '$\star\star\star$')
        tex = tex.replace('1cgn', '$\star$')
        print(tex)


# In[149]:


# To estimate the label errors with confident learning
label_errors_idx = cleanlab.pruning.get_noise_indices(
    s=s,
    psx=pyx,
    confident_joint=cj,
    prune_method='both',
    sorted_index_method='normalized_margin',  # ['prob_given_label', 'normalized_margin']
)


# In[155]:


print_errors(label_errors_idx, num_to_view=20, latex=True)

