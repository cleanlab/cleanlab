# coding: utf-8

# In[ ]:


from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
from cleanlab import latent_estimation
from cleanlab import pruning
from cleanlab.latent_algebra import compute_inv_noise_matrix
from cleanlab.noise_generation import generate_noise_matrix_from_trace
from cleanlab.noise_generation import generate_noisy_labels
from cleanlab.util import value_counts
import numpy as np
import scipy
import pytest

# In[ ]:


seed = 1


# In[ ]:


def make_data(
        sparse=False,
        means=[[3, 2], [7, 7], [0, 8]],
        covs=[[[5, -1.5], [-1.5, 1]], [[1, 0.5], [0.5, 4]], [[5, 1], [1, 5]]],
        sizes=[400, 200, 200],
        avg_trace=0.8,
        seed=1,  # set to None for non-reproducible randomness
):
    np.random.seed(seed=seed)

    m = len(means)  # number of classes
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

    if sparse:
        X_train = scipy.sparse.csr_matrix(X_train)
        X_test = scipy.sparse.csr_matrix(X_test)

    # Compute p(y=k)
    py = np.bincount(y_train) / float(len(y_train))

    noise_matrix = generate_noise_matrix_from_trace(
        m,
        trace=avg_trace * m,
        py=py,
        valid_noise_matrix=True,
        seed=seed,
    )

    # Generate our noisy labels using the noise_marix.
    s = generate_noisy_labels(y_train, noise_matrix)
    ps = np.bincount(s) / float(len(s))

    # Compute inverse noise matrix
    inv = compute_inv_noise_matrix(py, noise_matrix, ps)

    # Estimate psx
    latent = latent_estimation.estimate_py_noise_matrices_and_cv_pred_proba(
        X=X_train,
        s=s,
        cv_n_folds=3,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "s": s,
        "ps": ps,
        "py": py,
        "noise_matrix": noise_matrix,
        "inverse_noise_matrix": inv,
        "est_py": latent[0],
        "est_nm": latent[1],
        "est_inv": latent[2],
        "cj": latent[3],
        "psx": latent[4],
        "m": m,
        "n": n,
    }


# In[ ]:


def test_exact_prune_count():
    remove = 5
    s = data['s']
    noise_idx = pruning.get_noise_indices(
        s=s,
        psx=data['psx'],
        num_to_remove_per_class=remove,
        prune_method='prune_by_class'
    )
    assert (all(value_counts(s[noise_idx]) == remove))


# In[ ]:


def test_pruning_both():
    data = make_data(sizes=[4, 4, 4])
    remove = 5
    s = data['s']
    class_idx = pruning.get_noise_indices(
        s=s,
        psx=data['psx'],
        num_to_remove_per_class=remove,
        prune_method='prune_by_class',
    )
    nr_idx = pruning.get_noise_indices(
        s=s,
        psx=data['psx'],
        num_to_remove_per_class=remove,
        prune_method='prune_by_noise_rate',
    )
    both_idx = pruning.get_noise_indices(
        s=s,
        psx=data['psx'],
        num_to_remove_per_class=remove,
        prune_method='both',
    )
    assert (all(s[both_idx] == s[class_idx & nr_idx]))


# In[ ]:


def test_prune_on_small_data():
    data = make_data(sizes=[4, 4, 4])
    for pm in ['prune_by_noise_rate', 'prune_by_class', 'both']:
        noise_idx = pruning.get_noise_indices(
            s=data['s'],
            psx=data['psx'],
            prune_method=pm,
        )
        # Num in each class < 5. Nothing should be pruned.
        assert (not any(noise_idx))


# In[ ]:


def test_cj_from_probs():
    data = make_data(sizes=[4, 4, 4])
    cj = latent_estimation.estimate_confident_joint_from_probabilities(
        s=data["s"],
        psx=data["psx"],
        force_ps=10,
    )
    true_ps = data["ps"] * data["n"]
    forced = cj.sum(axis=1)

    cj = latent_estimation.estimate_confident_joint_from_probabilities(
        s=data["s"],
        psx=data["psx"],
        force_ps=1,
    )
    forced1 = cj.sum(axis=1)

    cj = latent_estimation.estimate_confident_joint_from_probabilities(
        s=data["s"],
        psx=data["psx"],
        force_ps=False,
    )
    regular = cj.sum(axis=1)
    # Forcing ps should make ps more similar to the true ps.
    assert (np.mean(true_ps - forced) <= np.mean(true_ps - regular))
    # Check that one iteration is the same as not forcing ps
    assert (np.mean(true_ps - forced1) - np.mean(true_ps - regular) < 2e-4)


# In[ ]:


def test_estimate_latent_py_method():
    data = make_data(sizes=[4, 4, 4])
    for py_method in ["cnt", "eqn", "marginal"]:
        py, nm, inv = latent_estimation.estimate_latent(
            confident_joint=data['cj'],
            s=data['s'],
            py_method=py_method,
        )
        assert (sum(py) - 1 < 1e-4)
    try:
        py, nm, inv = latent_estimation.estimate_latent(
            confident_joint=data['cj'],
            s=data['s'],
            py_method='INVALID',
        )
    except ValueError as e:
        assert ('should be' in str(e))
        with pytest.raises(ValueError) as e:
            py, nm, inv = latent_estimation.estimate_latent(
                confident_joint=data['cj'],
                s=data['s'],
                py_method='INVALID',
            )


# In[ ]:


def test_estimate_latent_converge():
    data = make_data(sizes=[4, 4, 4])
    py, nm, inv = latent_estimation.estimate_latent(
        confident_joint=data['cj'],
        s=data['s'],
        converge_latent_estimates=True,
    )

    py2, nm2, inv2 = latent_estimation.estimate_latent(
        confident_joint=data['cj'],
        s=data['s'],
        converge_latent_estimates=False,
    )
    # Check results are similar, but not the same.
    assert (np.any(inv != inv2))
    assert (np.any(py != py2))
    assert (np.all(abs(py - py2) < 0.1))
    assert (np.all(abs(nm - nm2) < 0.1))
    assert (np.all(abs(inv - inv2) < 0.1))


# In[ ]:

@pytest.mark.parametrize("sparse", [True, False])
def test_estimate_noise_matrices(sparse):
    data = make_data(sparse=sparse, seed=seed)
    nm, inv = latent_estimation.estimate_noise_matrices(
        X=data["X_train"],
        s=data["s"],
    )
    assert (np.all(abs(nm - data["est_nm"]) < 0.1))
    assert (np.all(abs(inv - data["est_inv"]) < 0.1))