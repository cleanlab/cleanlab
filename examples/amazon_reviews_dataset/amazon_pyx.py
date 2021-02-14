#!/usr/bin/env python
# coding: utf-8

# # Filters on the original amazon reviews dataset:
# 1. coreset (reviewers who reviewed at least five things and products with at least five reviews)
# 2. helpful (reviews with more helpful upvotes than unhelpful upvotes - requires at least one upvote)
# 3. sentiment non-ambiguity (has to be rated 1, 3, or 5 -- no way to verify that a 2 is really a 2 ya know? its either positive middle or negative, but what really is a 4? so i drop all 2s and 4s)
# 4. non-empty
# 
# This results in ~ 10 million reviews.

# In[1]:


# These imports enhance Python2/3 compatibility.
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement


# In[2]:


import json
from cleanlab.models.fasttext import FastTextClassifier, data_loader
import cleanlab
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
import os
from datetime import datetime as dt


# In[3]:


data_dir = "/media/ssd/datasets/datasets/amazon5core/"
data_fn = "amazon5core.json"
write_fn = 'amazon5core.txt'
write_dir = "/home/curtis/"


# In[4]:


# Fetch and preprocess data

need_text_data = False
need_to_prepare_data_for_fasttext = False
if need_to_prepare_data_for_fasttext:
    # Convert amazon dataset to fasttext format
    # Only include reviews with more helpful votes than unhelpful votes
    # This takes about 6 minutes.
    labels = []
    with open(data_dir + data_fn, 'r') as rf:
        with open(write_dir + write_fn, 'w') as wf:
#             for i in range(1000000):
#                 d = json.loads(rf.readline())
            for line in rf:
                d = json.loads(line)
                h = d['helpful']
                if h[0] > h[1] // 2:
                    label = int(d['overall'])
                    if label in [1,3,5]:
                        text = d['reviewText']
                        if len(text) > 0:
                            wf.write("__label__{} {}\n".format(
                                label, 
                                text.strip().replace('\n', ' __newline__ '),
                            ))
                            labels.append(label)                          
    label_map = {1:0, 3:1, 5:2}
    labels = [label_map[l] for l in labels]
else:
    labels = np.empty(9996437, dtype=int)
    if need_text_data:
        text = []
    loc = write_dir + 'amazon5core.preprocessed.txt'
    bs = 1000000
    label_map = {'__label__1':0, '__label__3':1, '__label__5':2}
    for i, (l, t) in enumerate(data_loader(loc, batch_size=bs)):
        labels[bs*i:bs*(i+1)] = [label_map[lab] for lab in l]
        if need_text_data:
            text.append(t)
    if need_text_data:
        text = [t for lst in text for t in lst]


# In[8]:


param_list = ParameterGrid({
    "cv_n_folds" : [3],
    "lr" : [.01, .05, 0.1, 0.5, 1.0],
    "ngram" : [3],
    "epochs" : [1, 5, 10],
    "dim" : [100],
})
seed = 0


# In[ ]:


# Fasttext model selection.

start_time = dt.now()
scores = []
for i, params in enumerate(param_list):
    print(params)    
    if i > 0:
        elapsed = dt.now() - start_time
        total_time = elapsed * len(param_list) / float(i)
        remaining = total_time - elapsed
        print('Elapsed:', str(elapsed)[:-7], '| Remaining:', str(remaining)[:-7])
    ftc = FastTextClassifier(
        train_data_fn=write_dir + 'amazon5core.preprocessed.txt', 
        batch_size = 100000, 
        labels = [1, 3, 5],
        kwargs_train_supervised = {
            'epoch': params['epochs'],
            'thread': 12,
            'lr': params['lr'],
            'wordNgrams': params['ngram'],
            'bucket': 200000,
            'dim': params['dim'],
            'loss': 'softmax', #'softmax', # 'hs'
        },
    )
    pyx = cleanlab.latent_estimation.estimate_cv_predicted_probabilities(
        X=np.arange(len(labels)),
        labels=labels,
        clf=ftc,
        cv_n_folds=params['cv_n_folds'],
        seed=seed,
    )
    # Write out
    wfn = write_dir + 'amazon_pyx_cv__folds_{}__epochs_{}__lr_{}__ngram_{}__dim_{}.npy'.format(
        params['cv_n_folds'], params['epochs'], params['lr'], params['ngram'], params['dim'])
    with open(wfn, 'wb') as wf:
        np.save(wf, pyx)

    # Check that probabilities are good.
    print("pyx finished. Writing:", wfn)
    scores.append(accuracy_score(labels, np.argmax(pyx, axis = 1)))
    print('Acc:', np.round(scores[-1], 4))


# In[12]:


best_params = param_list[np.argmax(scores)]
print('best params', best_params)
wfn = write_dir + 'amazon_pyx_cv__folds_{}__epochs_{}__lr_{}__ngram_{}__dim_{}.npy'.format(
        params['cv_n_folds'], params['epochs'], params['lr'], params['ngram'], params['dim'])
print('located in:', wfn)


# In[18]:


train_from_scratch = False
# Train the best model from scratch
# No need to do this if you've already run the
# hyper-parameter optimization above.
if train_from_scratch:
    cv_n_folds = 10 # Increasing more improves pyx, at great cost.
    seed = 0
    lr = .01
    ngram = 3
    epochs = 10 # Increasing more doesn't do much.
    dim = 100

    ftc = FastTextClassifier(
        train_data_fn=write_dir + 'amazon5core.preprocessed.txt', 
        batch_size = 100000, 
        labels = [1, 3, 5],
        kwargs_train_supervised = {
            'epoch' : epochs,
            'thread' : 12,
            'lr' : lr,
            'wordNgrams' : ngram,
            'bucket' : 200000,
            'dim' : dim,
            'loss' : 'softmax', #'softmax', # 'hs'
        }
    )

    pyx = cleanlab.latent_estimation.estimate_cv_predicted_probabilities(
        X=np.arange(len(labels)),
        labels=labels,
        clf=ftc,
        cv_n_folds=cv_n_folds,
        seed=seed,
    )

    # Write out pyx
    wfn = write_dir + 'amazon_pyx_cv__folds_{}__epochs_{}__lr_{}__ngram_{}__dim_{}.npy'.format(
        cv_n_folds, epochs, lr, ngram, dim)
    with open(wfn, 'wb') as wf:
        np.save(wf, pyx)

    # Check that probabilities are good.
    print(wfn)
    accuracy_score(labels, np.argmax(pyx, axis = 1))

