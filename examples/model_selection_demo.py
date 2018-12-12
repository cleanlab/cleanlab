
# coding: utf-8

# # Hyperparameter Optimization Tutorial
# 
# This tutorial will show you the main hyper-parameters for LearningWithNoisyLabels. There are only three!
# 
# 1. `prune_method` : str (default: `'prune_by_noise_rate'`), Method used for pruning.
#     * Values: [`'prune_by_class'`, `'prune_by_noise_rate'`, or `'both'`]. 
#     * `'prune_by_noise_rate'`: works by removing examples with *high probability* of being mislabeled for every non-diagonal in the prune_counts_matrix (see pruning.py).
#     * `'prune_by_class'`: works by removing the examples with *smallest probability* of belonging to their given class label for every class.
#     * `'both'`: Finds the examples satisfying (1) AND (2) and removes their set conjunction. 
# 
# 
# 2. `prune_count_method` : str (default `'inverse_nm_dot_s'`)
#     * Values: [`'inverse_nm_dot_s'` or `'calibrate_confident_joint'`]
#     * DO NOT USE 'calibrate_confident_joint' if you already know the noise matrix and will call .fit(noise_matrix = known_noise_matrix) or .fit(inverse_noise_matrix = known_inverse_noise_matrix) because 'calibrate_confident_joint' will estimate the noise without using this information.
#     * IN ALL OTHER CASES, We recommend always using 'calibrate_confident_joint' because it is faster and more robust when no noise matrix info is given.
#     * Determines the method used to estimate the counts of the joint P(s, y) that will be used to determine how many examples to prune for every class that are flipped to every other class, as follows:
#     
#     ```python
#     if prune_count_method == 'inverse_nm_dot_s':
#         # Matrix of counts(y=k and s=l)
#         prune_count_matrix = inverse_noise_matrix * s_counts 
#     elif prune_count_method == 'calibrate_confident_joint':
#         # Calibrate so (first make sum to 1, then make sum to total number of examples)
#         prune_count_matrix = confident_joint.T / float(confident_joint.sum()) * len(s)
#       ```
# 
# 
# 3. converge_latent_estimates : bool (Default: False)
#     * If true, forces numerical consistency of latent estimates. Each is estimated independently, but they are related mathematically with closed form  equivalences. This will iteratively enforce mathematically consistency.
# 
# ## This tutorial uses hypopt for faster hyper-optimization using a validation set (instead of slow cross validation).
# ### `$ pip install hypopt`

# In[5]:


# Python 2 and 3 compatibility
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement


# In[50]:


from hypopt.model_selection import GridSearch
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.noise_generation import generate_noise_matrix_from_trace
from cleanlab.noise_generation import generate_noisy_labels
from cleanlab.util import print_noise_matrix
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import numpy as np
import copy


# In[35]:


def make_linear_dataset(n_classes = 3, n_samples = 300):
    X, y = make_classification(n_samples = n_samples, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1, n_classes=n_classes)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return (X, y)


# In[43]:


param_grid = {
    'prune_method': ['prune_by_class', 'prune_by_noise_rate', 'both'],
    'prune_count_method': ['inverse_nm_dot_s', 'calibrate_confident_joint'],
    'converge_latent_estimates': [True, False],
}


# In[44]:


# Set the sparsity of the noise matrix.
frac_zero_noise_rates = 0.0 # Consider increasing to 0.5
# A proxy for the fraction of labels that are correct.
avg_trace = 0.65 # ~35% wrong labels. Increasing makes the problem easier.
# Amount of data for each dataset.
dataset_size = 250 # Try 250 or 400 to use less or more data.
num_classes = 3

ds = make_linear_dataset(n_classes=num_classes, n_samples=num_classes*dataset_size)
X, y = ds
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.25, random_state=1)


# In[72]:


for name, clf in [
    (
        "Naive Bayes", 
        GaussianNB(),
    ),
    (
        "Logistic Regression", 
         LogisticRegression(random_state=0, solver = 'lbfgs', multi_class = 'auto'),
    ),
]:
    print("\n", "="*len(name), "\n", name, '\n', "="*len(name))
    np.random.seed(seed=0)
    clf_copy = copy.deepcopy(clf)
    # Compute p(y=k), the ground truth class prior on the labels.
    py = np.bincount(y_train) / float(len(y_train))
    # Generate the noisy channel to characterize the label errors.
    noise_matrix = generate_noise_matrix_from_trace(
        K = num_classes,
        trace = num_classes * avg_trace, 
        py = py,
        frac_zero_noise_rates = frac_zero_noise_rates,
    )
    print_noise_matrix(noise_matrix)
    # Create the noisy labels. This method is exact w.r.t. the noise_matrix.
    y_train_with_errors = generate_noisy_labels(y_train, noise_matrix)
    lnl_cv = GridSearch(
        model = LearningWithNoisyLabels(clf), 
        param_grid = param_grid, 
        num_threads = 4,
        seed = 0,
    )
    lnl_cv.fit(
        X_train = X_train, 
        y_train = y_train_with_errors,
        X_val = X_val,
        y_val = y_val,
        verbose = False,
    )
    # Also compute the test score with default parameters
    clf_copy.fit(X_train, y_train_with_errors)
    score_opt = lnl_cv.model.score(X_test, y_test)
    score_default = clf_copy.score(X_test, y_test)
    print("Accuracy with default parameters:", np.round(score_default, 2))
    print("Accuracy with optimized parameters:", np.round(score_opt, 2))
    print()
    s = "Optimal parameter settings using {}".format(name)
    print(s)
    print("-"*len(s))
    for key in lnl_cv.best_params.keys():
        print(key, ":", lnl_cv.best_params[key])

