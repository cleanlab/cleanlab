#!/usr/bin/env python
# coding: utf-8

# # Filters on the original amazon reviews dataset:
# 1. coreset (reviewers who reviewed at least five things and products with at least five reviews)
# 2. helpful (reviews with more helpful upvotes than unhelpful upvotes - requires at least one upvote)
# 3. sentiment non-ambiguity (has to be rated 1, 3, or 5 -- no way to verify that a 2 is really a 2 ya know? its either positive middle or negative, but what really is a 4? so i drop all 2s and 4s)
# 4. non-empty
# 
# This results in ~ 10 million reviews.

# In[3]:


# These imports enhance Python2/3 compatibility.
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement


# In[4]:


import json
from cleanlab.models.fasttext import FastTextClassifier, data_loader
import cleanlab
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import copy
from datetime import datetime as dt
import pickle
import scipy


# In[5]:


data_dir = "/media/ssd/datasets/datasets/amazon5core/"
data_fn = "amazon5core.json"
write_fn = 'amazon5core.txt'
write_dir = "/home/curtis/"


# In[9]:


tfidf_already_trained = False
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
    text = []
    loc = write_dir + 'amazon5core.preprocessed.txt'
    bs = 1000000
    label_map = {'__label__1':0, '__label__3':1, '__label__5':2}
    for i, (l, t) in enumerate(data_loader(loc, batch_size=bs)):
        labels[bs*i:bs*(i+1)] = [label_map[lab] for lab in l]
        if not tfidf_already_trained:
            text.append(t)
    if not tfidf_already_trained:
        text = [t for lst in text for t in lst]


# In[5]:


# Preprocess the data by running this
# cat amazon5core.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > amazon5core.preprocessed.txt


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Bag of words vectorizer on the entire corpus.\n\nif tfidf_already_trained:\n    with open('/home/curtis/amazon_text_vectorized.npz', 'rb') as rf:\n        X = scipy.sparse.load_npz(rf)\nelse:\n    # Takes about 20 minutes\n    tfidf = TfidfVectorizer(\n        stop_words='english', \n        ngram_range=(1,1), # (1, 2) takes too much mem\n#         max_features=2000000,\n    )\n    X = tfidf.fit_transform(text)\n    with open('/home/curtis/amazon_text_vectorized.npz', 'wb') as wf:\n        scipy.sparse.save_npz(wf, X)")


# In[ ]:


# text_already_vectorized = False
# if text_already_vectorized:
#     with open('/home/curtis/amazon_text_vectorized.npy', 'rb') as rf:
#         text = np.load(rf)
# else:
#     # Takes a while, maybe 25 minutes.
#     text = tfidf.transform(text)
#     with open('/home/curtis/amazon_text_vectorized.npy', 'wb') as wf:
#         np.save(wf, text)


# In[20]:


# Train data using cross-validation
seed = 0
cv_n_folds = 3
n = len(labels)
m = 3

# Create cross-validation object for out-of-sample predicted probabilities.
# CV folds preserve the fraction of noisy positive and
# noisy negative examples in each class.
kf = StratifiedKFold(n_splits = cv_n_folds, shuffle = True, random_state = seed)

# Intialize out array (output of trained network)
pyx = np.empty((n, m))

# Split data into "cv_n_folds" stratified folds.
for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(range(n), labels)):
    print(k, len(cv_train_idx), len(cv_holdout_idx))
    start = dt.now()
    clf = SGDClassifier(
        alpha=0.000001, loss='modified_huber',
        max_iter=50, n_jobs=12,
        penalty='l2', random_state=0, tol=0.0001,
    )
    X_train = X[cv_train_idx]
    print("X_train", str(dt.now() - start)[:-7])
    X_holdout = X[cv_holdout_idx]
    print("X_holdout", str(dt.now() - start)[:-7])
    y_train = labels[cv_train_idx]
    print("y_train", str(dt.now() - start)[:-7])
    y_holdout = labels[cv_holdout_idx]
    print("y_holdout", str(dt.now() - start)[:-7])
    clf.fit(X_train, y_train)
    print("clf_fit", str(dt.now() - start)[:-7])
    pyx[cv_holdout_idx] = clf.predict_proba(X_holdout)
    print("pyx[cv_holdout_idx]", str(dt.now() - start)[:-7])


# In[21]:


# Stored results directory
pyx_dir = '/media/ssd/datasets/pyx/amazon/'

# Write out
with open(write_dir + 'amazon_pyx_tfidf_cv_{}fold.npy'.format(cv_n_folds), 'wb') as wf:
    np.save(wf, pyx)


# In[23]:


# Load in pyx
with open(write_dir + 'amazon_pyx_tfidf_cv_{}fold.npy'.format(cv_n_folds), 'rb') as rf:
    pyx = np.load(rf)


# In[24]:


# Check that probabilities are good.
accuracy_score(labels, np.argmax(pyx, axis = 1))

