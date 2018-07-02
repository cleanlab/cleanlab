# ```confidentlearning```
A Python package for Confident Learning with state-of-the-art algorithms for multiclass learning with noisy labels, detection of label errors in massive datasets, latent noisy channel estimation, latent prior estimation, and much more.

#### Confident learning theory and algorithms are:
1. fast - only two hours (on cpu-based laptop) to find the label errors in the 2012 ImageNet Validation set 
2. robust - provable generalization and risk minimimzation guarantees with imperfect probability estimation
3. general - works with any probablistic classifier, Faster R-CNN, logistic regression, LSTM, etc.

Check out these [examples](examples).

## Note to the reviewers of our NIPS 2018 manuscript
Please view the [supplementary materials of our NIPS2018 submission by clicking here](https://www.dropbox.com/s/n8hydz9zj6skqwg/nips2018_confident_learning_supplements.pdf?dl=0). The password is the first word of the title of our submission, all lowercased. A system error prevented inclusion at submit-time. 



## Installation

Python 2.7 and Python 3.5 are supported.

To install the **confidentlearning** package with pip, just run:

```
$ pip install git+https://github.com/cgnorthcutt/confidentlearning.git
```

If you have issues, you can also clone the repo and install:

```
$ conda update pip # if you use conda
$ git clone https://github.com/cgnorthcutt/confidentlearning.git
$ cd confidentlearning
$ pip install -e .
```

## Get started with easy, quick examples.

New to **confidentlearing**? Start with:

1. [Visualizing confident learning](examples/visualizing_confident_learning.ipynb)
2. [A simple example of learning with noisy labels on the multiclass Iris dataset](examples/iris_simple_example.ipynb). 

These examples show how easy it is to characterize label noise in datasets, learn with noisy labels, identify label errors, estimate latent priors and noisy channels, and more.

## Automatically identify ~50 label errors in MNIST with confident learning. [[link]](examples/finding_MNIST_label_errors).
![Image depicting label errors in MNIST train set.](https://raw.githubusercontent.com/cgnorthcutt/confidentlearning/master/img/mnist_training_label_errors24_prune_by_noise_rate.png)
Label errors of the original MNIST **train** dataset identified algorithmically using the rankpruning algorithm. Depicts the 24 least confident labels, ordered left-right, top-down by increasing self-confidence (probability of belonging to the given label), denoted conf in teal. The label with the largest predicted probability is in green. Overt errors are in red.

![Image depicting label errors in MNIST test set.](https://raw.githubusercontent.com/cgnorthcutt/confidentlearning/master/img/mnist_test_label_errors8.png)
 Selected label errors in the MNIST **test** dataset ordered by increasing self-confidence (in teal).

## Automatically identify ~5k (of 50k) validation set label errors in ImageNet. [[link]](examples/finding_ImageNet_label_errors).
![Image depicting label errors in ImageNet validation set.](https://raw.githubusercontent.com/cgnorthcutt/confidentlearning/master/img/imagenet_validation_label_errors_96_prune_by_noise_rate.jpg)
Label errors in the 2012 ImageNet validation dataset identified automatically with confident learning using a pre-trained resnet18. Displayed are the 96 least confident labels. We see that ImageNet contains numerous multi-label images, although it is used widely by the machine learning and vision communities as a single-label benchmark dataset.


## Documentation by Example - Quick Tutorials

Many of these methods have default parameters that won't cover here. Check out the method docstrings for full documentation.

### Multiclass learning with noisy labels (in **3** lines of code):
**rankpruning** is a fast, general, robust algorithm for multiclass learning with noisy labels. It adds minimal overhead, needing only *O(nm<sup>2</sup>)* time for n training examples and m classes, works with any classifier, and is easy to use.
```python
from confidentlearning.classification import RankPruning
# RankPruning uses logreg by default, so this is unnecessary. 
# We include it here for clarity, but this step is omitted below.
from sklearn.linear_model import LogisticRegression as logreg

# Wrap around any classifier. Yup, neural networks work, too.
rp = RankPruning(clf=logreg()) 
# X_train is numpy matrix of training examples (integers for large data)
# train_labels_with_errors is a numpy array of labels of length n (# of examples), usually denoted 's'.
rp.fit(X_train, train_labels_with_errors) 
# Estimate the predictions you would have gotten by training with *no* label errors.
predicted_test_labels = rp.predict(X_test)
``` 

### Estimate the confident joint, the latent noisy channel matrix, *P<sub>s | y</sub>* and inverse, *P<sub>y | s</sub>*, the latent prior of the unobserved, actual true labels, *p(y)*, and the predicted probabilities.:
where *s* denotes a random variable that represents the observed, noisy label and *y* denotes a random variable representing the hidden, actual labels. Both *s* and *y* take any of the m classes as values. The **confidentlearning** package supports different levels of granularity for computation depending on the needs of the user. Because of this, we support multiple alternatives, all no more than a few lines, to estimate these latent distribution arrays, enabling the user to reduce computation time by only computing what they need to compute, as seen in the examples below.

Throughout these examples, you'll see a variable called *confident_joint*. The confident joint is an m x m matrix (m is the number of classes) that counts, for every observed, noisy class, the number of examples that confidently belong to every latent, hidden class. It counts the number of examples that we are confident are labeled correctly or incorrectly for every pair of obseved and unobserved classes. The confident joint is an unnormalized estimate of the complete-information latent joint distribution, *P<sub>s,y</sub>*. Most of the methods in the **confidentlearing** package start by first estimating the *confident_joint*.

#### Option 1: Compute the confident joint and predicted probs first. Stop if that's all you need.
```python
from confidentlearning.latent_estimation import estimate_latent
from confidentlearning.latent_estimation import estimate_confident_joint_and_cv_pred_proba

# Compute the confident joint and the n x m predicted probabilities matrix (psx),
# for n examples, m classes. Stop here if all you need is the confident joint.
confident_joint, psx = estimate_confident_joint_and_cv_pred_proba(
    X=X_train, 
    s=train_labels_with_errors,
    clf = logreg(), # default, you can use any classifier
)

# Estimate latent distributions: p(y) as est_py, P(s|y) as est_nm, and P(y|s) as est_inv
est_py, est_nm, est_inv = estimate_latent(confident_joint, s=train_labels_with_errors)
```

#### Option 2: Estimate the latent distribution matrices in a single line of code.
```python
from confidentlearning.latent_estimation import estimate_py_noise_matrices_and_cv_pred_proba
est_py, est_nm, est_inv, confident_joint, psx = estimate_py_noise_matrices_and_cv_pred_proba(
    X=X_train,
    s=train_labels_with_errors,
)
```

#### Option 3: Skip computing the predicted probabilities if you already have them
```python
# Already have psx? (n x m matrix of predicted probabilities)
# For example, you might get them from a pre-trained model (like resnet on ImageNet)
# With the confidentlearning package, you estimate directly with psx.
from confidentlearning.latent_estimation import estimate_py_and_noise_matrices_from_probabilities
est_py, est_nm, est_inv, confident_joint = estimate_py_and_noise_matrices_from_probabilities(
    s=train_labels_with_errors, 
    psx=psx,
)

``` 

### Estimate label errors in a dataset:
With the **confidentlearning** package, we can instantly fetch the indices of all estimated label errors, with nothing provided by the user except a classifier, examples, and their noisy labels. Like the previous example, there are various levels of granularity.

```python
from confidentlearning.pruning import get_noise_indices
# We computed psx, est_inv, confident_joint in the previous example.
label_errors = get_noise_indices(
    s=train_labels_with_errors, # required
    psx=psx, # required
    inverse_noise_matrix=est_inv, # not required, include to avoid recomputing
    confident_joint=confident_joint, # not required, include to avoid recomputing
)
``` 


### Estimate the latent joint probability distribution matrix of the noisy and true labels, *P<sub>s,y</sub>*: 
There are two methods to compute *P<sub>s,y</sub>*, the complete-information distribution matrix that captures the number of pairwise label flip errors when multipled by the total number of examples as *n * P<sub>s,y</sub>*.

#### Method 1: Guarantees the rows of *P<sub>s,y</sub>* correctly sum to *p(s)*, by first computing *P<sub>y | s</sub>*. 
This method occurs when hyperparameter prune_count_method = 'inverse_nm_dot_s' in RankPruning.fit() and get_noise_indices(). 

```python
from confidentlearning.util import value_counts
# *p(s)* is the prior of the observed, noisy labels and an array of length m (# of classes)
ps = value_counts(s) / float(len(s))
# We computed est_inv (estimated inverse noise matrix) in the previous example (two above).
psy = np.transpose(est_inv * ps) # Matrix of prob(s=l and y=k)
```


#### Method 2: Simplest. Compute by re-normalizing the confident joint. Rows won't sum to *p(s)*
This method occurs when hyperparameter prune_count_method = 'calibrate_confident_joint' in RankPruning.fit() and get_noise_indices().
```python
from confidentlearning.util import value_counts
# *p(s)* is the prior of the observed, noisy labels and an array of length m (# of classes)
ps = value_counts(s) / float(len(s))
# We computed confident_joint in the previous example (two above).
psy = confident_joint / float(confident_joint.sum()) # calibration, i.e. re-normalization
```

### Generate valid, class-conditional, unformly random noisy channel matrices:

```python
# Generate a valid (necessary conditions for learnability are met) noise matrix for any trace > 1
from confidentlearning.noise_generation import generate_noise_matrix_from_trace
noise_matrix = generate_noise_matrix_from_trace(
    K = number_of_classes, 
    trace = float_value_greater_than_1_and_leq_K,
    py = prior_of_y_actual_labels_which_is_just_an_array_of_length_K,
    frac_zero_noise_rates = float_from_0_to_1_controlling_sparsity,
)

# Check if a noise matrix is valid (necessary conditions for learnability are met)
from confidentlearning.noise_generation import noise_matrix_is_valid
is_valid = noise_matrix_is_valid(noise_matrix, prior_of_y_which_is_just_an_array_of_length_K)

```

### Support for numerous *weak supervision* and *learning with noisy labels* functionalities:

```python
# Generate noisy labels using the noise_marix. Guarantees exact amount of noise in labels.
from confidentlearning.noise_generation import generate_noisy_labels
s_noisy_labels = generate_noisy_labels(y_hidden_actual_labels, noise_matrix)

# This package is a full of other useful methods for learning with noisy labels.
# The tutorial stops here, but you don't have to. Inspect method docstrings for full docs.
```



## The Polyplex 
### The key to learning in the presence of label errors is estimating the joint distribution between the actual, hidden labels '*y*' and the observed, noisy labels '*s*'. Using confident learning, we can completely characterize the trace of the latent joint distribution, *trace(P<sub>s,y</sub>)*, given *p(y)*, for any fraction of label errors, i.e. for any trace of the noisy channel, *trace(P<sub>s|y</sub>)*.
You can check out how to do this yourself here:
1. [Drawing Polyplices](examples/drawing_polyplices.ipynb)
2. [Computing Polyplices](confidentlearning/polyplex.ipynb)
