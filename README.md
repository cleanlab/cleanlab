![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/cleanlab_logo_open_source_transparent_optimized_size.png)

cleanlab automatically finds and fixes errors in any ML dataset. This data-centric AI package facilitates **machine learning with messy, real-world data** by providing **clean lab**els during training.

```python

# cleanlab works with **any classifier**. Yup, you can use sklearn/PyTorch/TensorFlow/XGBoost/etc.
cl = cleanlab.classification.CleanLearning(sklearn.YourFavoriteClassifier())

# cleanlab finds data and label issues in **any dataset**... in ONE line of code!
label_issues = cl.find_label_issues(data, labels)

# cleanlab trains a robust version of your model that works more reliably with noisy data.
cl.fit(data, labels)

# cleanlab estimates the predictions you would have gotten if you had trained with *no* label issues.
cl.predict(test_data)

# A true data-centric AI package, cleanlab quantifies class-level issues and overall data quality, for any dataset.
cleanlab.dataset.health_summary(labels, confident_joint=cl.confident_joint)
```

Get started with: [documentation](https://docs.cleanlab.ai/), [tutorials](https://docs.cleanlab.ai/stable/tutorials/image.html), [examples](https://github.com/cleanlab/examples), and [blogs](https://cleanlab.ai/blog/).

 - [Learn how to](https://docs.cleanlab.ai/stable/tutorials/index) run cleanlab on your own data in just 5 minutes!
 - Quickstart with 5-minute tutorials for classification with: [image](https://docs.cleanlab.ai/stable/tutorials/image.html), [text](https://docs.cleanlab.ai/stable/tutorials/text.html), [audio](https://docs.cleanlab.ai/stable/tutorials/audio.html), and [tabular](https://docs.cleanlab.ai/stable/tutorials/tabular.html) data.


[![pypi](https://img.shields.io/pypi/v/cleanlab.svg)](https://pypi.org/pypi/cleanlab/)
[![os](https://img.shields.io/badge/platform-noarch-lightgrey)](https://pypi.org/pypi/cleanlab/)
[![py\_versions](https://img.shields.io/badge/python-3.6%2B-blue)](https://pypi.org/pypi/cleanlab/)
[![build\_status](https://github.com/cleanlab/cleanlab/workflows/CI/badge.svg)](https://github.com/cleanlab/cleanlab/actions?query=workflow%3ACI)
[![coverage](https://codecov.io/gh/cleanlab/cleanlab/branch/master/graph/badge.svg)](https://app.codecov.io/gh/cleanlab/cleanlab)
[![docs](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=cleanlab)](https://docs.cleanlab.ai/)
[![Slack Community](https://img.shields.io/static/v1?logo=slack&style=flat&color=white&label=slack&message=community)](https://join.slack.com/t/cleanlab-community/shared_invite/zt-17lszn4hv-gg2FhZPXYfljq_l01uo92g)
[![Twitter](https://img.shields.io/twitter/follow/CleanlabAI?style=social)](https://twitter.com/CleanlabAI)

-----

<details><summary><b>News! (2022) </b> -- cleanlab made accessible for everybody, not just ML researchers (<b>click to learn more</b>) </summary>
<p>
<ul>
<li> <b>April 2022 📖</b> cleanlab 2.0.0 released! Lays foundations for this library to grow into a general-purpose data-centric AI toolkit. </li>
<li> <b>March 2022 📖</b>  Documentation migrated to new website: <a href="https://docs.cleanlab.ai/">docs.cleanlab.ai</a> with quickstart tutorials for image/text/audio/tabular data.</li>
<li> <b>Feb 2022 💻</b> <a href="https://docs.cleanlab.ai/master/migrating/migrate_v2.html">APIs simplified</a> to make cleanlab accessible for everybody, not just ML researchers </li>
</ul>
</p>
</details>

<details><summary><b>News! (2021) </b> -- cleanlab finds pervasive label errors in the most common ML datasets (<b>click to learn more</b>) </summary>
<p>
<ul>
<li> <b>Dec 2021 🎉</b>  NeurIPS published the <a href="https://arxiv.org/abs/2103.14749">label errors paper (Northcutt, Athalye, & Mueller, 2021)</a>.</li>
<li> <b>Apr 2021 🎉</b>  Journal of AI Research published the <a href="https://jair.org/index.php/jair/article/view/12125">confident learning paper (Northcutt, Jiang, & Chuang, 2021)</a>.</li>
<li> <b>Mar 2021 😲</b>  cleanlab used to find and fix label issues in 10 of the most common ML benchmark datasets, published in: <a href="https://neurips.cc/Conferences/2021/ScheduleMultitrack?event=22763">NeurIPS 2021</a>. Along with <a href="https://arxiv.org/abs/2103.14749">the paper (Northcutt, Athalye, & Mueller, 2021)</a>, the authors launched <a href="https://labelerrors.com">labelerrors.com</a> where you can view the label issues in these datasets.</li>
</ul>
</p>
</details>

<details><summary><b>News! (2020) </b> -- cleanlab adds support for all OS, achieves state-of-the-art, supports co-teaching and more (<b>click to learn more</b>) </summary>
<p>
<ul>
<li> <b>Dec 2020 🎉</b>  cleanlab supports NeurIPS workshop paper <a href="https://securedata.lol/camera_ready/28.pdf">(Northcutt, Athalye, & Lin, 2020)</a>.</li>
<li> <b>Dec 2020 🤖</b>  cleanlab supports <a href="https://github.com/cleanlab/cleanlab/blob/master/cleanlab/classification.py#L215">PU learning</a>.</li>
<li> <b>Feb 2020 🤖</b>  cleanlab now natively supports Mac, Linux, and Windows.</li>
<li> <b>Feb 2020 🤖</b>  cleanlab now supports <a href="https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/coteaching.py">Co-Teaching</a> <a href="https://arxiv.org/abs/1804.06872">(Han et al., 2018)</a>.</li>
<li> <b>Jan 2020 🎉</b> cleanlab achieves state-of-the-art on CIFAR-10 with noisy labels. Code to reproduce:  <a href="https://github.com/cleanlab/examples/tree/master/contrib/v1/cifar10">examples/cifar10</a>. This is a great place to see how to use cleanlab on real datasets (with predicted probabilities from trained model already precomputed for you).</li>
</ul>
</p>
</details>

Release notes for past versions are available [here](https://github.com/cleanlab/cleanlab/releases). Details behind certain updates are explained in our [blog](https://cleanlab.ai/blog/).

**Long-time cleanlab user?**

* Here's a [guide](https://docs.cleanlab.ai/v2.0.0/migrating/migrate_v2.html) on how to migrate to cleanlab 2.0.0.

## So fresh, so cleanlab

cleanlab **clean**s your data's **lab**els via state-of-the-art *confident learning* algorithms, published in this [paper](https://jair.org/index.php/jair/article/view/12125) and [blog](https://l7.curtisnorthcutt.com/confident-learning). See datasets cleaned with cleanlab at [labelerrors.com](https://labelerrors.com). This package helps you find all the label issues lurking in your data and train more reliable ML models.

cleanlab is:

1. **backed by theory**
   - with [provable guarantees](https://arxiv.org/abs/1911.00068) of exact noise estimation and label error finding in realistic cases with imperfect models.
2. **fast**
   - Code is optimized and parallel-threaded (< 1 second to find label issues in ImageNet with pre-computed probabilities).
4. **easy-to-use**
   - Find label issues or train noise-robust models in one line of code. By default, cleanlab requires no hyper-parameters.
6. **general**
   - Works with **[any dataset](https://labelerrors.com/)** and **any model**, e.g., TensorFlow, PyTorch, sklearn, xgboost, etc.
<br/>

![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/label-errors-examples.png)
<p align="center">
Examples of incorrect given labels in various image datasets <a href="https://l7.curtisnorthcutt.com/label-errors">found and corrected</a> using cleanlab.
</p>

## Run cleanlab

cleanlab supports Linux, macOS, and Windows and runs on Python 3.6+.

- Get started [here](https://docs.cleanlab.ai/)! Install via `pip` or `conda` as described [here](https://docs.cleanlab.ai/).
- Developers who install the bleeding-edge master branch from source should refer to [this master version of documentation](https://docs.cleanlab.ai/master/index.html).

<details><summary>
cleanlab core package components
(<b>click to learn more</b>)
</summary>
<br/>

Many methods have default parameters not covered here. Check out the [documentation for the master branch version](https://docs.cleanlab.ai/master/)

### cleanlab Core Package Components

1. **cleanlab/classification.py** - [CleanLearning()](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/classification.py#L141) class for learning with noisy labels.
2. **cleanlab/count.py** - Estimates and fully characterizes all variants of label noise.
3. **cleanlab/filter.py** - Finds the examples with label issues in a dataset.
4. **cleanlab/rank.py** - Rank every example in a dataset with various label quality scores.
5. **cleanlab.dataset.py** - Provides dataset-level and class-level overviews of issues in your dataset.
6. **cleanlab/benchmarking/noise\_generation.py** - Generate noisy labels for benchmarking, reproduction, and ML research.

<br/>
</details>

## Use cleanlab with any model (TensorFlow, PyTorch, sklearn, xgboost, etc.)

All features of cleanlab work with **any dataset** and **any model**. Yes, any model: scikit-learn, PyTorch, Tensorflow, Keras, JAX, HuggingFace, MXNet, XGBoost, etc.
If you use a sklearn-compatible classifier, cleanlab methods work out-of-the-box.

<details><summary>
It’s also easy to use your favorite non-sklearn-compatible model (<b>click to learn more</b>)
</summary>
<br/>

There's nothing you need to do if your model already has `.fit()`, `.predict()`, and `.predict_proba()` methods.
Otherwise, just wrap your custom model into a Python class that inherits the `sklearn.base.BaseEstimator`:

``` python
from sklearn.base import BaseEstimator
class YourFavoriteModel(BaseEstimator): # Inherits sklearn base classifier
    def __init__(self, ):
        pass  # ensure this re-initializes parameters for neural net models
    def fit(self, X, y, sample_weight=None):
        pass
    def predict(self, X):
        pass
    def predict_proba(self, X):
        pass
    def score(self, X, y, sample_weight=None):
        pass
```

This inheritance allows to apply a wide range of sklearn functionality like hyperparameter-optimization to your custom model.
Now you can use your model with every method in cleanlab. Here's one example:

``` python
from cleanlab.classification import CleanLearning
cl = CleanLearning(clf=YourFavoriteModel())  # has all the same methods of YourFavoriteModel
cl.fit(train_data, train_labels_with_errors)
cl.predict(test_data)
```

#### Want to see a working example? [Here’s a compliant PyTorch MNIST CNN class](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/mnist_pytorch.py)

More details are provided in documentation of [cleanlab.classification.CleanLearning](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/classification.py#L141).

Note, some libraries exist to give you sklearn-compatibility for free. For PyTorch, check out the [skorch](https://skorch.readthedocs.io/) Python library which will wrap your PyTorch model into a sklearn-compatible model ([example](https://docs.cleanlab.ai/master/tutorials/image.html)). For TensorFlow/Keras, check out [SciKeras](https://www.adriangb.com/scikeras/) ([example](https://docs.cleanlab.ai/master/tutorials/text.html)). Many libraries also already offer a special scikit-learn API, for example: [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn) or [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html).

<br/>
</details>

## Cool cleanlab applications

<details><summary>
Reproducing results in <a href="https://arxiv.org/abs/1911.00068">Confident Learning paper</a>
(<b>click to learn more</b>)
</summary>
<br/>

For additional details, check out the: [confidentlearning-reproduce repository](https://github.com/cgnorthcutt/confidentlearning-reproduce).

### State of the Art Learning with Noisy Labels in CIFAR

A step-by-step guide to reproduce these results is available [here](https://github.com/cleanlab/examples/tree/master/contrib/v1/cifar10). This guide is also a good tutorial for using cleanlab on any large dataset. You'll need to `git clone`
[confidentlearning-reproduce](https://github.com/cgnorthcutt/confidentlearning-reproduce) which contains the data and files needed to reproduce the CIFAR-10 results.

![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/cifar10_benchmarks.png)

Comparison of confident learning (CL), as implemented in cleanlab, versus seven recent methods for learning with noisy labels in CIFAR-10. Highlighted cells show CL robustness to sparsity. The five CL methods estimate label issues, remove them, then train on the cleaned data using [Co-Teaching](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/coteaching.py).

Observe how cleanlab (i.e. the CL method) is robust to large sparsity in label noise whereas prior art tends to reduce in performance for increased sparsity, as shown by the red highlighted regions. This is important because real-world label noise is often sparse, e.g. a tiger is likely to be mislabeled as a lion, but not as most other classes like airplane, bathtub, and microwave.

### Find label issues in ImageNet

Use cleanlab to identify \~100,000 label errors in the 2012 ILSVRC ImageNet training dataset: [examples/imagenet](https://github.com/cleanlab/examples/tree/master/contrib/v1/imagenet).

![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/imagenet_train_label_errors_32.jpg)

Label issues in ImageNet train set found via cleanlab. Label Errors are boxed in red. Ontological issues in green. Multi-label images in blue.

### Find Label Errors in MNIST

Use cleanlab to identify \~50 label errors in the MNIST dataset: [examples/mnist](https://github.com/cleanlab/examples/tree/master/contrib/v1/mnist).

![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/mnist_training_label_errors24_prune_by_noise_rate.png)

Top 24 least-confident labels in the original MNIST **train** dataset, algorithmically identified via cleanlab. Examples are ordered left-right, top-down by increasing self-confidence (predicted probability that the **given** label is correct), denoted **conf** in teal. The most-likely correct label (with largest predicted probability) is in green. Overt label errors highlighted in red.

<br/>
</details>

<details><summary>
cleanlab performance across 4 data distributions and 9 classifiers
(<b>click to learn more</b>)
</summary>
<br/>

cleanlab is a general tool that can learn with noisy labels regardless of dataset distribution or classifier type: [examples/classifier\_comparison](https://github.com/cleanlab/examples/blob/master/classifier_comparison.ipynb).

![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/demo_cleanlab_across_datasets_and_classifiers.png)

Each sub-figure above depicts the decision boundary learned using [cleanlab.classification.CleanLearning](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/classification.py#L141) in the presence of extreme (\~35%) label errors (circled in green). Label noise is class-conditional (not uniformly random). Columns are organized by the classifier used, except the left-most column which depicts the ground-truth data distribution. Rows are organized by dataset.

Each sub-figure depicts accuracy scores on a test set (with correct non-noisy labels) as decimal values:

* LEFT (in black): The classifier test accuracy trained with perfect labels (no label errors).
* MIDDLE (in blue): The classifier test accuracy trained with noisy labels using cleanlab.
* RIGHT (in white): The baseline classifier test accuracy trained with noisy labels.

As an example, the table below is the noise matrix (noisy channel) *P(s | y)
characterizing the label noise for the first dataset row in the figure. *s* represents the observed noisy labels and *y* represents the latent, true labels. The trace of this matrix is 2.6. A trace of 4 implies no label noise. A cell in this matrix is read like: "Around 38% of true underlying '3' labels were randomly flipped to '2' labels in the
observed dataset."

| `p(label︱y)` | y=0  | y=1  | y=2  | y=3  |
|--------------|------|------|------|------|
| label=0      | 0.55 | 0.01 | 0.07 | 0.06 |
| label=1      | 0.22 | 0.87 | 0.24 | 0.02 |
| label=2      | 0.12 | 0.04 | 0.64 | 0.38 |
| label=3      | 0.11 | 0.08 | 0.05 | 0.54 |

<br/>
</details>

<details><summary>
ML research using cleanlab
(<b>click to learn more</b>)
</summary>
<br/>

Researchers may find some components of this package useful for evaluating algorithms for ML with noisy labels. For additional details/notation, refer to [the Confident Learning paper](https://jair.org/index.php/jair/article/view/12125).

### Methods to Standardize Research with Noisy Labels

cleanlab supports a number of functions to generate noise for benchmarking and standardization in research. This next example shows how to generate valid, class-conditional, uniformly random noisy channel matrices:

``` python
# Generate a valid (necessary conditions for learnability are met) noise matrix for any trace > 1
from cleanlab.benchmarking.noise_generation import generate_noise_matrix_from_trace
noise_matrix=generate_noise_matrix_from_trace(
    K=number_of_classes,
    trace=float_value_greater_than_1_and_leq_K,
    py=prior_of_y_actual_labels_which_is_just_an_array_of_length_K,
    frac_zero_noise_rates=float_from_0_to_1_controlling_sparsity,
)

# Check if a noise matrix is valid (necessary conditions for learnability are met)
from cleanlab.benchmarking.noise_generation import noise_matrix_is_valid
is_valid=noise_matrix_is_valid(
    noise_matrix,
    prior_of_y_which_is_just_an_array_of_length_K,
)
```

For a given noise matrix, this example shows how to generate noisy labels. Methods can be seeded for reproducibility.

``` python
# Generate noisy labels using the noise_marix. Guarantees exact amount of noise in labels.
from cleanlab.benchmarking.noise_generation import generate_noisy_labels
s_noisy_labels = generate_noisy_labels(y_hidden_actual_labels, noise_matrix)

# This package is a full of other useful methods for learning with noisy labels.
# The tutorial stops here, but you don't have to. Inspect method docstrings for full docs.
```

<br/>
</details>

<details><summary>
cleanlab for advanced users
(<b>click to learn more</b>)
</summary>
<br/>

Many methods and their default parameters are not covered here. Check out the [documentation for the master branch version](https://docs.cleanlab.ai/master/) for the full suite of features supported by the cleanlab API.

## Use any custom model's predicted probabilities to find label errors in 1 line of code

pred_probs (num_examples x num_classes matrix of predicted probabilities) should already be computed on your own, with any classifier. pred_probs must be obtained in a holdout/out-of-sample manner (e.g. via cross-validation).
* cleanlab can do this for you via [`cleanlab.count.estimate_cv_predicted_probabilities`](https://docs.cleanlab.ai/master/cleanlab/count.html)]
* Tutorial with more info: [[here](https://docs.cleanlab.ai/master/tutorials/pred_probs_cross_val.html)]
* Examples how to compute pred_probs with: [[CNN image classifier (PyTorch)](https://docs.cleanlab.ai/stable/tutorials/image.html)], [[NN text classifier (TensorFlow)](https://docs.cleanlab.ai/stable/tutorials/text.html)]

```python
# label issues are ordered by likelihood of being an error. First index is most likely error.
from cleanlab.filter import find_label_issues

ordered_label_issues = find_label_issues(  # One line of code!
    labels=numpy_array_of_noisy_labels,
    pred_probs=numpy_array_of_predicted_probabilities,
    return_indices_ranked_by='normalized_margin', # Orders label issues
 )
```

Pre-computed **out-of-sample** predicted probabilities for CIFAR-10 train set are available: [here](https://github.com/cleanlab/examples/tree/master/contrib/v1/cifar10#pre-computed-psx-for-every-noise--sparsity-condition).

## Fully characterize label noise and uncertainty in your dataset.

*s* denotes a random variable that represents the observed, noisy label and *y* denotes a random variable representing the hidden, actual labels. Both *s* and *y* take any of the m classes as values. The cleanlab package supports different levels of granularity for computation depending on the needs of the user. Because of this, we support multiple alternatives, all no more than a few lines, to estimate these latent distribution arrays, enabling the user to reduce computation time by only computing what they need to compute, as seen in the examples below.

Throughout these examples, you’ll see a variable called *confident\_joint*. The confident joint is an m x m matrix (m is the number of classes) that counts, for every observed, noisy class, the number of examples that confidently belong to every latent, hidden class. It counts the number of examples that we are confident are labeled correctly or incorrectly for every pair of observed and unobserved classes. The confident joint is an unnormalized estimate of the complete-information latent joint distribution, *Ps,y*.

The label flipping rates are denoted *P(s | y)*, the inverse rates are *P(y | s)*, and the latent prior of the unobserved, true labels, *p(y)*.

Most of the methods in the **cleanlab** package start by first estimating the *confident\_joint*. You can learn more about this in the [confident learning paper](https://arxiv.org/abs/1911.00068).

### Option 1: Compute the confident joint and predicted probs first. Stop if that’s all you need.

``` python
from cleanlab.count import estimate_latent
from cleanlab.count import estimate_confident_joint_and_cv_pred_proba

# Compute the confident joint and the n x m predicted probabilities matrix (pred_probs),
# for n examples, m classes. Stop here if all you need is the confident joint.
confident_joint, pred_probs = estimate_confident_joint_and_cv_pred_proba(
    X=X_train,
    labels=train_labels_with_errors,
    clf=logreg(), # default, you can use any classifier
)

# Estimate latent distributions: p(y) as est_py, P(s|y) as est_nm, and P(y|s) as est_inv
est_py, est_nm, est_inv = estimate_latent(
    confident_joint,
    labels=train_labels_with_errors,
)
```

### Option 2: Estimate the latent distribution matrices in a single line of code.

``` python
from cleanlab.count import estimate_py_noise_matrices_and_cv_pred_proba
est_py, est_nm, est_inv, confident_joint, pred_probs = estimate_py_noise_matrices_and_cv_pred_proba(
    X=X_train,
    labels=train_labels_with_errors,
)
```

### Option 3: Skip computing the predicted probabilities if you already have them.

``` python
# Already have pred_probs? (n x m matrix of predicted probabilities)
# For example, you might get them from a pre-trained model (like resnet on ImageNet)
# With the cleanlab package, you estimate directly with pred_probs.
from cleanlab.count import estimate_py_and_noise_matrices_from_probabilities
est_py, est_nm, est_inv, confident_joint = estimate_py_and_noise_matrices_from_probabilities(
    labels=train_labels_with_errors,
    pred_probs=pred_probs,
)
```

## Completely characterize label noise in a dataset:

The joint probability distribution of noisy and true labels, *P(s,y)*, completely characterizes label noise with a class-conditional *m x m* matrix.

``` python
from cleanlab.count import estimate_joint
joint = estimate_joint(
    labels=noisy_labels,
    pred_probs=probabilities,
    confident_joint=None,  # Provide if you have it already
)
```

<br/>
</details>

<details><summary>
Positive-Unlabeled learning with cleanlab
(<b>click to learn more</b>)
</summary>
<br/>

Positive-Unlabeled (PU) learning (in which your data only contains a few positively labeled examples with the rest unlabeled) is just a special case of [CleanLearning](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/classification.py#L141) when one of the classes has no error. `P` stands for the positive class and **is assumed to have zero label errors** and `U` stands for unlabeled data, but in practice, we just assume the `U` class is a noisy negative class that actually contains some positive examples. Thus, the goal of PU learning is to (1) estimate the proportion of negatively labeled examples that actually belong to the positive class (see`fraction\_noise\_in\_unlabeled\_class` in the last example), (2) find the errors (see last example), and (3) train on clean data (see first example below). cleanlab does all three, taking into account that there are no label errors in whichever class you specify as positive.

There are two ways to use cleanlab for PU learning. We'll look at each here.

Method 1. If you are using the cleanlab classifier [CleanLearning()](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/classification.py#L141), and your dataset has exactly two classes (positive = 1, and negative = 0), PU
learning is supported directly in cleanlab. You can perform PU learning like this:

``` python
from cleanlab.classification import CleanLearning
from sklearn.linear_model import LogisticRegression
# Wrap around any classifier. Yup, you can use sklearn/pyTorch/TensorFlow/FastText/etc.
pu_class = 0 # Should be 0 or 1. Label of class with NO ERRORS. (e.g., P class in PU)
cl = CleanLearning(clf=LogisticRegression(), pulearning=pu_class)
cl.fit(X=X_train_data, labels=train_noisy_labels)
# Estimate the predictions you would have gotten by training with *no* label errors.
predicted_test_labels = cl.predict(X_test)
```

Method 2. However, you might be using a more complicated classifier that doesn't work well with [CleanLearning](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/classification.py#L141) (see this example for CIFAR-10). Or you might have 3 or more classes. Here's how to use cleanlab for PU learning in this situation. To let cleanlab know which class has no error (in standard PU learning, this is the P class), you need to set the threshold for that class to 1 (1 means the probability that the labels of that class are correct is 1, i.e. that class has no
error). Here's the code:

``` python
import numpy as np
# K is the number of classes in your dataset
# pred_probs are the cross-validated predicted probabilities.
# s is the array/list/iterable of noisy labels
# pu_class is a 0-based integer for the class that has no label errors.
thresholds = np.asarray([np.mean(pred_probs[:, k][s == k]) for k in range(K)])
thresholds[pu_class] = 1.0
```

Now you can use cleanlab however you were before. Just be sure to pass in `thresholds` as a parameter wherever it applies. For example:

``` python
# Uncertainty quantification (characterize the label noise
# by estimating the joint distribution of noisy and true labels)
cj = compute_confident_joint(s, pred_probs, thresholds=thresholds, )
# Now the noise (cj) has been estimated taking into account that some class(es) have no error.
# We can use cj to find label errors like this:
indices_of_label_issues = find_label_issues(s, pred_probs, confident_joint=cj, )

# In addition to label issues, cleanlab can find the fraction of noise in the unlabeled class.
# First we need the inv_noise_matrix which contains P(y|s) (proportion of mislabeling).
_, _, inv_noise_matrix = estimate_latent(confident_joint=cj, labels=s, )
# Because inv_noise_matrix contains P(y|s), p (y = anything | labels = pu_class) should be 0
# because the prob(true label is something else | example is in pu_class) is 0.
# What's more interesting is p(y = anything | s is not put_class), or in the binary case
# this translates to p(y = pu_class | s = 1 - pu_class) because pu_class is 0 or 1.
# So, to find the fraction_noise_in_unlabeled_class, for binary, you just compute:
fraction_noise_in_unlabeled_class = inv_noise_matrix[pu_class][1 - pu_class]
```

Now that you have `indices_of_label_errors`, you can remove those label issues and train on clean data (or only remove some of the label issues and iteratively use confident learning / cleanlab to improve results).

<br/>
</details>

## Citation and related publications

cleanlab is based on peer-reviewed research. Here are the relevant papers to cite if you use this package:

<details><summary><a href="https://arxiv.org/abs/1911.00068">Confident Learning (JAIR '21)</a> (<b>click to show bibtex</b>) </summary>

    @article{northcutt2021confidentlearning,
        title={Confident Learning: Estimating Uncertainty in Dataset Labels},
        author={Curtis G. Northcutt and Lu Jiang and Isaac L. Chuang},
        journal={Journal of Artificial Intelligence Research (JAIR)},
        volume={70},
        pages={1373--1411},
        year={2021}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/1705.01936">Rank Pruning (UAI '17)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{northcutt2017rankpruning,
        author={Northcutt, Curtis G. and Wu, Tailin and Chuang, Isaac L.},
        title={Learning with Confident Examples: Rank Pruning for Robust Classification with Noisy Labels},
        booktitle = {Proceedings of the Thirty-Third Conference on Uncertainty in Artificial Intelligence},
        series = {UAI'17},
        year = {2017},
        location = {Sydney, Australia},
        numpages = {10},
        url = {http://auai.org/uai2017/proceedings/papers/35.pdf},
        publisher = {AUAI Press},
    }

</details>

## Other resources

- [Blog post: Introduction to Confident Learning](https://l7.curtisnorthcutt.com/confident-learning)

- [NeurIPS 2021 paper: Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks](https://arxiv.org/abs/2103.14749)

- [Cleanlab Blog](https://cleanlab.ai/blog/)

## Join our community

* The best place to learn is [our Slack community](https://join.slack.com/t/cleanlab-community/shared_invite/zt-17lszn4hv-gg2FhZPXYfljq_l01uo92g).

* Have ideas for the future of cleanlab? How are you using cleanlab? [Join the discussion](https://github.com/cleanlab/cleanlab/discussions).

* Have code improvements for cleanlab? See the [development guide](DEVELOPMENT.md) and [submit a pull request](CONTRIBUTING.md).

* Have an issue with cleanlab? [Search existing issues](https://github.com/cleanlab/cleanlab/issues?q=is%3Aissue) or [submit a new issue](https://github.com/cleanlab/cleanlab/issues/new).

* Need professional help with cleanlab? 
Join our [\#help Slack channel](https://join.slack.com/t/cleanlab-community/shared_invite/zt-17lszn4hv-gg2FhZPXYfljq_l01uo92g) and message one of our core developers, Jonas Mueller, or schedule a meeting via email: team@cleanlab.ai

## License

Copyright (c) 2017-2022 Cleanlab Inc.

cleanlab is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

cleanlab is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

See [GNU Affero General Public LICENSE](https://github.com/cleanlab/cleanlab/blob/master/LICENSE) for details.
