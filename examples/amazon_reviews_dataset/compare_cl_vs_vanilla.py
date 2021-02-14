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
import sys
import multiprocessing
from datetime import datetime as dt
import pickle5 as pickle  # Helps with backporting


# In[3]:


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# In[4]:


cpu_threads = 32
cv_n_folds = 5 # Increasing more improves pyx, at great cost.
seed = 0
lr = .01
ngram = 3
epochs = 10 # Increasing more doesn't do much.
dim = 100
data_dir = '/datasets/datasets/amazon5core/'
pyx_dir = '/datasets/cgn/pyx/amazon/'
cur_dir = '/home/cgn/cgn/cleanlab/examples/amazon_reviews_dataset/'


# # Compute cross-validated pred probs on train set.

# In[5]:


# Compute 5-fold cross-validated predicted probabilities on train set.
# Try two different train sets (every 10th vs 11th item sent to test set)
#   to make sure the choice of test set doesn't influence the results.
crossval_already_done = True
if not crossval_already_done:
    for test_split in [10, 11]:
        train_fn = data_dir + 'train_{}_amazon5core.preprocessed.txt'.format(test_split)
        # Get labels
        noisy_labels = np.empty(file_len(train_fn), dtype=int)
        bs = 1000000
        label_map = {'__label__1':0, '__label__3':1, '__label__5':2}
        for i, (l, t) in enumerate(data_loader(train_fn, batch_size=bs)):
            noisy_labels[bs*i:bs*(i+1)] = [label_map[lab] for lab in l]

        ftc = FastTextClassifier(
            train_data_fn=train_fn, 
            batch_size=100000, 
            labels=[1, 3, 5],
            kwargs_train_supervised = {
                'epoch': epochs,
                'thread': cpu_threads,
                'lr': lr,
                'wordNgrams': ngram,
                'bucket': 200000,
                'dim': dim,
                'loss': 'softmax', #'softmax', # 'hs'
            },
        )
        pyx = cleanlab.latent_estimation.estimate_cv_predicted_probabilities(
            X=np.arange(len(noisy_labels)),
            labels=noisy_labels,
            clf=ftc,
            cv_n_folds=cv_n_folds,
            seed=seed,
        )
        # Write out
        wfn = pyx_dir + 'amazon_pyx_train_{}_cv__folds_{}__epochs_{}__lr_{}__ngram_{}__dim_{}.npy'.format(
            test_split, cv_n_folds, epochs, lr, ngram, dim)
        with open(wfn, 'wb') as wf:
            np.save(wf, pyx)

        # Check that probabilities are good.
        print("pyx finished. Writing:", wfn)
        acc = accuracy_score(noisy_labels, np.argmax(pyx, axis=1))
        print('Acc: {:.2%}'.format(acc))


# # Find noise with confident learning

# In[6]:


finding_noise_already_done = True
if not finding_noise_already_done:
    for test_split in [10, 11]:
        print('Test split: every {}th item.'.format(test_split))
        train_fn = data_dir + 'train_{}_amazon5core.preprocessed.txt'.format(test_split)
        # Get labels
        noisy_labels = np.empty(file_len(train_fn), dtype=int)
        bs = 1000000
        label_map = {'__label__1':0, '__label__3':1, '__label__5':2}
        for i, (l, t) in enumerate(data_loader(train_fn, batch_size=bs)):
            noisy_labels[bs*i:bs*(i+1)] = [label_map[lab] for lab in l]

        # Read in cross-validated predicted probs
        rfn = pyx_dir + 'amazon_pyx_train_{}_cv__folds_{}__epochs_{}__lr_{}__ngram_{}__dim_{}.npy'.format(
            test_split, cv_n_folds, epochs, lr, ngram, dim)
        with open(rfn, 'rb') as rf:
            pyx = np.load(rf)
        acc = accuracy_score(noisy_labels, np.argmax(pyx, axis=1))
        print('Cross-val Acc: {:.2%}'.format(acc))

        # Find noise masks with confident learning methods
        # Estimate the confident joint, a proxy for the joint distribution of label noise.
        cj, cj_only_label_error_indices = cleanlab.latent_estimation.compute_confident_joint(
            noisy_labels, pyx,
            return_indices_of_off_diagonals=True,
        )
        py, nm, inv = cleanlab.latent_estimation.estimate_latent(cj, noisy_labels)

        # Five CL methods for finding label errors.
        cj_only_bool_mask = np.zeros(len(noisy_labels), dtype=bool)
        for idx in cj_only_label_error_indices:
            cj_only_bool_mask[idx] = True

        argmax_bool_mask = cleanlab.baseline_methods.baseline_argmax(pyx, noisy_labels)

        cl_pbc_bool_mask = cleanlab.pruning.get_noise_indices(
            noisy_labels, pyx, confident_joint=cj,
            prune_method='prune_by_class')

        cl_pbnr_bool_mask = cleanlab.pruning.get_noise_indices(
            noisy_labels, pyx, confident_joint=cj,
            prune_method='prune_by_noise_rate')

        cl_both_bool_mask = cleanlab.pruning.get_noise_indices(
            noisy_labels, pyx, confident_joint=cj,
            prune_method='both')
        
        # True if clean data, False if noisy
        cl_masks = {
            'cj_only': ~cj_only_bool_mask,
            'argmax': ~argmax_bool_mask,
            'pbc': ~cl_pbc_bool_mask,
            'pbnr': ~cl_pbnr_bool_mask,
            'both': ~cl_both_bool_mask,
        }

        # Find the errors that all CL methods agree on.
        common_errors = ~(list(cl_masks.values())[0])
        for l in cl_masks.values():
            common_errors = common_errors & ~l
        cl_masks['cl_intersection_all_methods'] = ~common_errors
        
        with open(cur_dir + 'pickles/cl_masks_{}.p'.format(test_split), 'wb') as handle:
            pickle.dump(cl_masks, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Generating cleaned versions of the dataset for each CL method.')
        for name, mask in list(cl_masks.items()):
            print(name, end=' | ')
            with open(data_dir + 'train_{}_amazon5core.preprocessed.txt'.format(
                    test_split), 'r') as rf:
                with open(data_dir + 'train_{}_amazon5core.{}.txt'.format(
                        test_split, name), 'w') as wf:          
                    for i, line in enumerate(rf):
                        if mask[i]:
                            print(line, end='', file=wf)
        print('Done!')


# # Benchmark CL methods versus Vanilla training
# * on a random 1 million example subset of Amazon Reviews data
# 
# ### Bash script example: running five trials of this script on five machines.
# 
# ```
# # On another machine
# seed=0
# trainsize=1000000
# epochs=5
# mkdir -p ~/amazon_reviews && cd ~/amazon_reviews
# { time python /home/cgn/cgn/cleanlab/examples/amazon_reviews_dataset/compare_cl_vs_vanilla.py $seed $trainsize $epochs ; } &> "out_seed_${seed}_trainsize_${trainsize}_epochs_${epochs}.log" & sleep 1 && tail -f "out_seed_${seed}_trainsize_${trainsize}_epochs_${epochs}.log"
# 
# # On another machine
# seed=1
# trainsize=1000000
# epochs=5
# mkdir -p ~/amazon_reviews && cd ~/amazon_reviews
# { time python /home/cgn/cgn/cleanlab/examples/amazon_reviews_dataset/compare_cl_vs_vanilla.py $seed $trainsize $epochs ; } &> "out_seed_${seed}_trainsize_${trainsize}_epochs_${epochs}.log" & sleep 1 && tail -f "out_seed_${seed}_trainsize_${trainsize}_epochs_${epochs}.log"
# 
# # On another machine
# seed=2
# trainsize=1000000
# epochs=5
# mkdir -p ~/amazon_reviews && cd ~/amazon_reviews
# { time python /home/cgn/cgn/cleanlab/examples/amazon_reviews_dataset/compare_cl_vs_vanilla.py $seed $trainsize $epochs ; } &> "out_seed_${seed}_trainsize_${trainsize}_epochs_${epochs}.log" & sleep 1 && tail -f "out_seed_${seed}_trainsize_${trainsize}_epochs_${epochs}.log"
# 
# # On another machine
# seed=3
# trainsize=1000000
# epochs=5
# mkdir -p ~/amazon_reviews && cd ~/amazon_reviews
# { time python /home/cgn/cgn/cleanlab/examples/amazon_reviews_dataset/compare_cl_vs_vanilla.py $seed $trainsize $epochs ; } &> "out_seed_${seed}_trainsize_${trainsize}_epochs_${epochs}.log" & sleep 1 && tail -f "out_seed_${seed}_trainsize_${trainsize}_epochs_${epochs}.log"
# 
# # On another machine
# seed=4
# trainsize=1000000
# epochs=5
# mkdir -p ~/amazon_reviews && cd ~/amazon_reviews
# { time python /home/cgn/cgn/cleanlab/examples/amazon_reviews_dataset/compare_cl_vs_vanilla.py $seed $trainsize $epochs ; } &> "out_seed_${seed}_trainsize_${trainsize}_epochs_${epochs}.log" & sleep 1 && tail -f "out_seed_${seed}_trainsize_${trainsize}_epochs_${epochs}.log"
# ```

# In[ ]:


try:
    seed = int(sys.argv[1])
except:
    seed = 0
try:
    TRAIN_SIZE = int(sys.argv[2])
except:
    TRAIN_SIZE = int(1e6)  # 1 million examples
try:
    epochs = int(sys.argv[3])
except:
    epochs = 20
try:
    cpu_threads = int(sys.argv[4])
except:
    cpu_threads = multiprocessing.cpu_count() // 2
    
print('Seed:', seed)
print('Train Size:', TRAIN_SIZE)
print('CPU threads:', cpu_threads)
print('Epochs:', epochs)
sys.stdout.flush()

lr = .01
ngram = 3
dim = 100
data_dir = '/datasets/datasets/amazon5core/'
pyx_dir = '/datasets/cgn/pyx/amazon/'

is_cl_method = lambda x: x != 'preprocessed'

results = []
# 'preprocessed' == vanilla baseline training with no CL.
cl_methods = ['preprocessed', 'cl_intersection_all_methods',
              'cj_only', 'argmax', 'pbc', 'pbnr', 'both']
for test_split in [10, 11]:
    np.random.seed(seed)
    # Prepare dataset
    with open(cur_dir + 'pickles/cl_masks_{}.p'.format(test_split), 'rb') as handle:
        cl_masks = pickle.load(handle)
    common_errors = ~cl_masks['cl_intersection_all_methods']
    print('Number of common errors among CL methods:', sum(common_errors))
    # Choose random subset of 1 million examples from the train data
    noisy_idx = np.arange(len(common_errors))[common_errors]
    clean_idx = np.arange(len(common_errors))[~common_errors]
    train_idx = np.concatenate([noisy_idx, np.random.choice(
        clean_idx, size=TRAIN_SIZE - len(noisy_idx), replace=False)])
    np.random.shuffle(train_idx)
    # Train
    for method in cl_methods:
        train_fn = data_dir + 'train_{}_amazon5core.preprocessed.txt'.format(test_split)
        test_fn = data_dir + 'test_{}_amazon5core.preprocessed.txt'.format(test_split)
        if is_cl_method(method):
            # Remove CL detected label errors
            cl_noise_idx = np.arange(len(cl_masks[method]))[~cl_masks[method]]
            clean_idx = np.array(list(set(train_idx).difference(cl_noise_idx)))
        # Set-up fast-text classifer
        ftc = FastTextClassifier(
            train_data_fn=train_fn,
            test_data_fn=test_fn,
            batch_size=100000, 
            labels=[1, 3, 5],
            kwargs_train_supervised = {
                'epoch': epochs,
                'thread': cpu_threads,
                'lr': lr,
                'wordNgrams': ngram,
                'bucket': 200000,
                'dim': dim,
                'loss': 'softmax', #'softmax', # 'hs'
            },
        )
        X = clean_idx if is_cl_method(method) else train_idx
        X_size = len(X)
        ftc.fit(X=X)
#         pred, test_labels = ftc.predict(train_data=False, return_labels=True)
#         acc = sum(pred == test_labels) / len(test_labels)
        results.append({
            'test_split': test_split,
            'method': 'vanilla' if method == 'preprocessed' else method,
            'acc': ftc.score(k=1),
            'train_size': X_size,
            'data_removed': TRAIN_SIZE - X_size,
        })
        print(results[-1])
        sys.stdout.flush()
print(results)
sys.stdout.flush()


# # Notes

# In[ ]:


# Results for entire 10 million Amazon Reviews dataset.
# While CL does outperform vanilla training, there is too
# much data (10 million excamples) and not enough noise
# for it CL to matter much.
results = [
    {'test_split': 10,
    'method': 'vanilla',
    'acc': 0.9088874728277995,
    },
    {'test_split': 10,
    'method': 'cj_only',
    'acc': 0.9090925460389359,
    },
    {'test_split': 10,
    'method': 'argmax',
    'acc': 0.9082052292668482,
    },
    {'test_split': 10,
    'method': 'pbc',
    'acc': 0.9099428495973062,
    },
    {'test_split': 10,
    'method': 'pbnr',
    'acc': 0.9093226281782596,
    }, 
    {'test_split': 10,
    'method': 'both',
    'acc': 0.9097417778146798,
    },
    {'test_split': 11,
    'method': 'vanilla',
    'acc': 0.9093695083558272,
    },
    {'test_split': 11,
     'method': 'cj_only',
     'acc': 0.9094795475627966,
    },
    {'test_split': 11,
     'method': 'argmax',
     'acc': 0.9084814919555838,
    },
    {'test_split': 11,
     'method': 'pbc',
     'acc': 0.9101496863332406,
    },
    {'test_split': 11,
     'method': 'pbnr',
     'acc': 0.9097271357784779,
    },
    {'test_split': 11,
     'method': 'both',
     'acc': 0.9102553239719312,
    },
]

