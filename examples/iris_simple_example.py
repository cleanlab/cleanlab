
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as logreg
import numpy as np
from confidentlearning.classification import RankPruning
from confidentlearning.noise_generation import generate_noisy_labels
from confidentlearning.util import value_counts
from confidentlearning.latent_algebra import compute_inv_noise_matrix


# In[2]:


# Seed for reproducibility
seed = 2
rp = RankPruning(clf = logreg(), seed = seed)
np.random.seed(seed = seed)

# Get iris dataset
iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Generate lots of noise.
noise_matrix = np.array([
    [0.5, 0.0, 0.0],
    [0.5, 1.0, 0.5],
    [0.0, 0.0, 0.5],
])

py = value_counts(y_train)
s = generate_noisy_labels(y_train, noise_matrix)

print('WITHOUT confident learning,', end=" ")
clf = logreg()
clf.fit(X_train, s)
pred = clf.predict(X_test)
print("Iris dataset test accuracy:", round(accuracy_score(pred, y_test), 2))

print("\nNow we show the improvement using confident learning to characterize the noise")
print("and learn on the data that is (with high confidence) labeled correctly.")
print()
print('WITH confident learning (noise matrix given),', end=" ")
rp.fit(X_train, s, noise_matrix = noise_matrix)
pred = rp.predict(X_test)
print("Iris dataset test accuracy:", round(accuracy_score(pred, y_test),2))

print('WITH confident learning (noise matrix and inverse noise matrix given),', end=" ")
rp.fit(X_train, s, noise_matrix = noise_matrix, inverse_noise_matrix=compute_inv_noise_matrix(py, noise_matrix))
pred = rp.predict(X_test)
print("Iris dataset test accuracy:", round(accuracy_score(pred, y_test),2))

print('WITH confident learning (using latent noise matrix estimation),', end=" ")
rp.fit(X_train, s, prune_count_method='inverse_nm_dot_s')
pred = rp.predict(X_test)
print("Iris dataset test accuracy:", round(accuracy_score(pred, y_test),2))

print('WITH confident learning (using calibrated confident joint),', end=" ")
rp.fit(X_train, s, prune_count_method='calibrate_confident_joint')
pred = rp.predict(X_test)
print("Iris dataset test accuracy:", round(accuracy_score(pred, y_test),2))


# ### Finally, we show the performance of confident learning across all combinations of parameter settings.

# In[3]:


from itertools import product

params = {
    "prune_count_method": ["calibrate_confident_joint", "inverse_nm_dot_s"],
    "prune_method": ["prune_by_noise_rate", "prune_by_class", "both"],
    "converge_estimates": [True, False],
}

keys, values = zip(*params.items())
for v in product(*values):
    job = dict(zip(keys, v))
    print("Param settings:", job)
    rp.fit(
        X_train, 
        s, 
        prune_method = job['prune_method'],
        prune_count_method = job['prune_count_method'],
        converge_latent_estimates = job['converge_estimates'],
    )
    pred = rp.predict(X_test)
    print("Iris dataset test accuracy (using confident learning):\t", round(accuracy_score(pred, y_test),2))
    print()

