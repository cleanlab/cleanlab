
# coding: utf-8

# In[1]:


from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
from numpy.random import multivariate_normal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

import cleanlab
from cleanlab.noise_generation import generate_noise_matrix_from_trace, generate_noisy_labels
from cleanlab.util import print_noise_matrix
from cleanlab.latent_estimation import estimate_confident_joint_and_cv_pred_proba, estimate_latent
from cleanlab.pruning import get_noise_indices
from cleanlab.classification import LearningWithNoisyLabels


# In[2]:


def show_decision_boundary(clf, title, show_noise = True):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
    y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])


    _ = plt.figure(figsize=(15, 12))
    plt.axis('off')

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, alpha=0.015)

    # Plot the distribution for viewing.
    if show_noise:
        # Plot noisy labels are circles around label errors
        for k in range(K):
            X_k = X_train[y_train == k] # data for class k
            _ = plt.scatter(X_k[:,0], X_k[:, 1], color=[color_map[noisy_k] for noisy_k in s[y_train==k]], s=150, marker=r"${a}$".format(a=str(k)), linewidth=1)
        _ = plt.scatter(X_train[:,0][s != y_train], X_train[:,1][s != y_train], s=400, facecolors='none', edgecolors='black', linewidth=0.8)
    else:
        # Plot the actual labels.
        for k in range(K):
            X_k = X_train[y_train==k] # data for class k
            plt.scatter(X_k[:,0], X_k[:, 1], color=colors[k], s=150, marker=r"${a}$".format(a=str(k)), linewidth=1)
    plt.title(title, fontsize=25)
    plt.show()


# In[3]:


seed = 1 # Seeded for reproducibility - remove to created random noise and distributions.
np.random.seed(seed = seed)

means = [ [3, 2], [7, 7], [0, 8] ]
covs = [ [[5, -1.5],[-1.5, 1]] , [[1, 0.5],[0.5, 4]], [[5, 1],[1, 5]] ]

K = len(means) # number of classes
sizes = [ 800, 400, 400 ]
data = []
labels = []
test_data = []
test_labels = []

for idx in range(len(means)):
    data.append(multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx]))
    test_data.append(multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx]))
    labels.append(np.array([idx for i in range(sizes[idx])]))
    test_labels.append(np.array([idx for i in range(sizes[idx])]))
X_train = np.vstack(data)
y_train = np.hstack(labels)
X_test = np.vstack(test_data)
y_test = np.hstack(test_labels) 

# Compute p(y=k)
py = np.bincount(y_train) / float(len(y_train))

noise_matrix = generate_noise_matrix_from_trace(
  K, 
  trace=1.5,
  py=py,
  valid_noise_matrix=True,
)

# Generate our noisy labels using the noise_marix.
s = generate_noisy_labels(y_train, noise_matrix)
ps = np.bincount(s) / float(len(s))

confident_joint, psx = estimate_confident_joint_and_cv_pred_proba(X_train, s, seed=seed)
est_py, est_noise_matrix, est_inverse_noise_matrix = estimate_latent(confident_joint, s)
idx_errors = get_noise_indices(s, psx)


# #### To show off the power of **cleanlab**, we've chosen an example of multiclass learning with noisy labels in which over 50% of the training labels are wrong.
# Toggle the ```trace``` parameter in ```generate_noise_matrix_from_trace``` above to try out different amounts of noise. Note, as we prove in our paper, learning becomes impossible if the ```trace <= 1```, so choose a value greater than 1, but less than, or equal to, the number of classes (3).

# In[4]:


est_joint = cleanlab.latent_estimation.estimate_joint(
    s=s,
    psx=psx,
    confident_joint=confident_joint, 
)
true_joint_distribution_of_label_errors = (noise_matrix * py)
percent_error_str = 'Percent of training examples that have wrong labels: ' +       str(int(round(100 - 100*true_joint_distribution_of_label_errors.trace()))) + "%"

colors = [(31 / 255., 119 / 255., 180 / 255.), (255 / 255., 127 / 255., 14 / 255.), (44 / 255., 160 / 255., 44 / 255.)]
color_map = dict(zip(range(len(colors)), colors))
try:
# Plot the distribution for your viewing.
    get_ipython().run_line_magic('matplotlib', 'inline')
    from matplotlib import pyplot as plt
    _ = plt.figure(figsize=(15, 12))
    _ = plt.axis('off')
    for k in range(K):
        X_k = X_train[y_train==k] # data for class k
        _ = plt.scatter(X_k[:,0], X_k[:, 1], color=colors[k], s=150, marker=r"${a}$".format(a=str(k)), linewidth=1)
    _ = plt.title("Original (unobserved) distribution, without any label errors.", fontsize=30)

    print("\n\n\n\n")

    # Plot the noisy distribution for viewing.
    _ = plt.figure(figsize=(15, 12))
    _ = plt.axis('off')
    for k in range(K):
        X_k = X_train[y_train == k] # data for class k
        _ = plt.scatter(X_k[:,0], X_k[:, 1], color=[color_map[noisy_k] for noisy_k in s[y_train==k]], s=150, marker=r"${a}$".format(a=str(k)), linewidth=1)
    _ = plt.scatter(X_train[:,0][s != y_train], X_train[:,1][s != y_train], s=400, facecolors='none', edgecolors='black', linewidth=2, alpha = 0.5)
    _ = plt.title('Observed distribution, with label errors circled.\nColors are the given labels, the numbers are latent.\n'+percent_error_str, fontsize=30)
    plt.show()

    print("\n\n\n\n")

    # Plot the noisy distribution for viewing.
    _ = plt.figure(figsize=(15, 12))
    _ = plt.axis('off')
    for k in range(K):
        X_k = X_train[idx_errors & (y_train == k)] # data for class k
        _ = plt.scatter(X_k[:,0], X_k[:, 1], color=[color_map[noisy_k] for noisy_k in s[y_train==k]], s=150, marker=r"${a}$".format(a=str(k)), linewidth=1)
    _ = plt.scatter(X_train[:,0][s != y_train], X_train[:,1][s != y_train], s=400, facecolors='none', edgecolors='black', linewidth=2, alpha = 0.5)
    _ = plt.title('Label errors detected using confident learning.\nEmpty circles show undetected label errors.\nUncircled data depicts false positives.', fontsize=30)
    plt.show()


    print("\n\n\n\n")

    _ = plt.figure(figsize=(15, 12))
    _ = plt.axis('off')
    for k in range(K):
        X_k = X_train[~idx_errors & (y_train == k)] # data for class k
        _ = plt.scatter(X_k[:,0], X_k[:, 1], color=[color_map[noisy_k] for noisy_k in s[y_train==k]], s=150, marker=r"${a}$".format(a=str(k)), linewidth=1)
    _ = plt.scatter(X_train[~idx_errors][:,0][s[~idx_errors] != y_train[~idx_errors]], X_train[~idx_errors][:,1][s[~idx_errors] != y_train[~idx_errors]], s=400, facecolors='none', edgecolors='black', linewidth=2, alpha = 0.5)
    _ = plt.title('Dataset after pruning detected label errors.', fontsize=30)
    plt.show()
except:
    print("Plotting is only supported in an iPython interface.")

print('The actual, latent, underlying noise matrix.')
print_noise_matrix(noise_matrix)
print('Our estimate of the noise matrix.')
print_noise_matrix(est_noise_matrix)
print()
print('The actual, latent, underlying joint distribution matrix.')
cleanlab.util.print_joint_matrix(true_joint_distribution_of_label_errors)
print('Our estimate of the joint distribution matrix.')
cleanlab.util.print_joint_matrix(est_joint)
print("Accuracy Comparison")
print("-------------------")
clf = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')
baseline_score = accuracy_score(y_test, clf.fit(X_train, s).predict(X_test))
print("Logistic regression:", baseline_score)
rp = LearningWithNoisyLabels(seed = seed)
rp_score = accuracy_score(y_test, rp.fit(X_train, s, psx=psx).predict(X_test))
print("Logistic regression (+rankpruning):", rp_score)
diff = rp_score - baseline_score
clf = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')
print('Fit on denoised data without re-weighting:', accuracy_score(y_test, clf.fit(X_train[~idx_errors], s[~idx_errors]).predict(X_test)))



try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    from matplotlib import pyplot as plt
    
    print("\n\n\n\n\n\n")
    
    clf = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')
    _ = clf.fit(X_train, s)
    show_decision_boundary(clf, 'Decision boundary for logistic regression trained with noisy labels.\n Test Accuracy: ' + str(round(baseline_score, 3)))

    _ = clf.fit(X_train, y_train)
    max_score = accuracy_score(y_test, clf.predict(X_test))
    show_decision_boundary(clf, 'Decision boundary for logistic regression trained with no label errors.\n Test Accuracy: ' + str(round(max_score, 3)), show_noise = False)

    show_decision_boundary(rp.clf, 'Decision boundary for LogisticRegression (+rankpruning) trained with noisy labels.\n Test Accuracy: ' + str(round(rp_score, 3)))
except:
    print("Plotting is only supported in an iPython interface.")


# In[5]:


param_grid = {
    "prune_method": ["prune_by_noise_rate", "prune_by_class", "both"],
    "converge_latent_estimates": [True, False],
}

# Fit LearningWithNoisyLabels across all parameter settings.
from sklearn.model_selection import ParameterGrid
params = ParameterGrid(param_grid)
scores = []
for param in params:
    clf = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')
    rp = LearningWithNoisyLabels(clf = clf, **param)
    _ = rp.fit(X_train, s) # s is the noisy y_train labels
    scores.append(accuracy_score(rp.predict(X_test), y_test))

# Print results sorted from best to least
for i in np.argsort(scores)[::-1]:
    print("Param settings:", params[i])
    print(
        "Accuracy (using confident learning):\t", 
        round(scores[i], 2),
        "\n"
    )
    
    # Print noise matrix for highest/lowest scoring models
    if i == np.argmax(scores) or i == np.argmin(scores):
        # Retrain with best parameters and show noise matrix estimation
        clf = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')
        rp = LearningWithNoisyLabels(clf = clf, **param)
        _ = rp.fit(X_train, s) # s is the noisy y_train labels
        print('The actual, latent, underlying noise matrix:', end = "")
        print_noise_matrix(noise_matrix)
        print('LearningWithNoisyLabels best estimate of the noise matrix:', end = "")
        print_noise_matrix(rp.noise_matrix)
        


# ### In the above example, notice the robustness to hyper-parameter choice and the stability of the algorithms across parameters. No setting of parameters dramatically affects the results. In fact, in certain non-trivial cases, we can prove that certain settings of parameters are equivalent.
# 
# ### In summary, the default setting of parameters tends to work well, but optimize across their settings freely.

# In[6]:


joint = cleanlab.latent_estimation.estimate_joint(s, psx, confident_joint)
print(joint)

print('\nThe above output should look like this:')
print(confident_joint / len(s)) 

