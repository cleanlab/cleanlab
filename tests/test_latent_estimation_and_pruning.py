
# coding: utf-8

# In[ ]:


from __future__ import print_function, absolute_import, division, unicode_literals, with_statement


# In[ ]:


from cleanlab import latent_estimation
from cleanlab.noise_generation import generate_noise_matrix_from_trace
from cleanlab.noise_generation import generate_noisy_labels
from cleanlab.latent_algebra import compute_inv_noise_matrix
from cleanlab import pruning
from cleanlab.util import value_counts
import numpy as np
import pytest


# In[ ]:


seed = 1


# In[ ]:


def make_data(
    means = [ [3, 2], [7, 7], [0, 8] ],
    covs = [ [[5, -1.5],[-1.5, 1]] , [[1, 0.5],[0.5, 4]], [[5, 1],[1, 5]] ],
    sizes = [ 400, 200, 200 ],
    avg_trace = 0.8,
    seed = 1, # set to None for non-reproducible randomness
):
    
    np.random.seed(seed = seed)

    m = len(means) # number of classes
    n = sum(sizes)
    data = []
    labels = []
    test_data = []
    test_labels = []

    for idx in range(m):
        data.append(np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx]))
        test_data.append(np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx]))
        labels.append(np.array([idx for i in range(sizes[idx])]))
        test_labels.append(np.array([idx for i in range(sizes[idx])]))
    X_train = np.vstack(data)
    y_train = np.hstack(labels)
    X_test = np.vstack(test_data)
    y_test = np.hstack(test_labels) 

    # Compute p(y=k)
    py = np.bincount(y_train) / float(len(y_train))

    noise_matrix = generate_noise_matrix_from_trace(
      m, 
      trace = avg_trace * m,
      py = py,
      valid_noise_matrix = True,
      seed = seed,
    )

    # Generate our noisy labels using the noise_marix.
    s = generate_noisy_labels(y_train, noise_matrix)
    ps = np.bincount(s) / float(len(s))
    
    # Compute inverse noise matrix
    inv = compute_inv_noise_matrix(py, noise_matrix, ps)
    
    # Estimate psx
    latent = latent_estimation.estimate_py_noise_matrices_and_cv_pred_proba(
        X = X_train,
        s = s,
        cv_n_folds = 3,
    )
    
    return {
        "X_train" : X_train,
        "y_train" : y_train,
        "X_test" : X_test,
        "y_test" : y_test,
        "s" : s,
        "ps" : ps,
        "py" : py,
        "noise_matrix" : noise_matrix,
        "inverse_noise_matrix" : inv,
        "est_py" : latent[0],
        "est_nm" : latent[1],
        "est_inv" : latent[2],
        "cj" : latent[3],
        "psx" : latent[4],
        "m" : m,
        "n" : n,
    }


# In[ ]:


data = make_data(seed = seed)


# In[ ]:


def test_invalid_prune_count_method():
    try:
        pruning.get_noise_indices(
            s = data['s'],
            psx = data['psx'],
            prune_count_method = 'INVALID_METHOD',
        )
    except ValueError as e:
        assert('should be' in str(e))
        with pytest.raises(ValueError) as e:
            pruning.get_noise_indices(
                s = data['s'],
                psx = data['psx'],
                prune_count_method = 'INVALID_METHOD',
            )


# In[ ]:


def test_invalid_prune_count_method():
    remove = 5
    s = data['s']
    noise_idx = pruning.get_noise_indices(
        s = s,
        psx = data['psx'],
        num_to_remove_per_class = remove
    )
    assert(all(value_counts(s[noise_idx]) == remove))


# In[ ]:


def test_pruning_both():
    remove = 5
    s = data['s']
    class_idx = pruning.get_noise_indices(
        s = s,
        psx = data['psx'],
        num_to_remove_per_class = remove,
        prune_method = 'prune_by_class',
    )
    nr_idx = pruning.get_noise_indices(
        s = s,
        psx = data['psx'],
        num_to_remove_per_class = remove,
        prune_method = 'prune_by_noise_rate',
    )
    both_idx = pruning.get_noise_indices(
        s = s,
        psx = data['psx'],
        num_to_remove_per_class = remove,
        prune_method = 'both',
    )
    assert(all(s[both_idx] == s[class_idx & nr_idx]))

