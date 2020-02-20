.. figure:: https://raw.githubusercontent.com/cgnorthcutt/cleanlab/master/img/cleanlab_logo.png
   :target: https://github.com/cgnorthcutt/cleanlab/
   :align: center
   :alt: cleanlab 

|  

``cleanlab`` is a machine learning python package for **learning with noisy labels** and **finding label errors in datasets**. ``cleanlab`` CLEANs LABels. It is powered by the theory of **confident learning**, published in this `paper <https://arxiv.org/abs/1911.00068>`__ |  `blog <https://l7.curtisnorthcutt.com/confident-learning>`__. 

****

* **News! (Jan 2020)** `cleanlab` achieves state-of-the-art on CIFAR-10 for learning with noisy labels. Code to reproduce is here:  `examples/cifar10 <https://github.com/cgnorthcutt/cleanlab/tree/master/examples/cifar10>`__. This is a great place for newcomers to see how to use cleanlab on real datasets. Data needed is available in the `confidentlearning-reproduce <https://github.com/cgnorthcutt/confidentlearning-reproduce>`__ repo, ``cleanlab`` v0.1.0 reproduces results in `the CL paper <https://arxiv.org/abs/1911.00068>`__.
* **News! (Feb 2020)**  `cleanlab` now natively supports Mac, Linux, and Windows.
* **News! (Feb 2020)**  `cleanlab` now supports `Co-Teaching <https://github.com/cgnorthcutt/cleanlab/blob/master/cleanlab/coteaching.py>`__  `(Han et al., 2018) <https://arxiv.org/abs/1804.06872>`__.


|pypi| |os| |py_versions| |build_status| |coverage|

.. |pypi| image:: https://img.shields.io/pypi/v/cleanlab.svg
    :target: https://pypi.org/pypi/cleanlab/
.. |os| image:: https://img.shields.io/badge/platform-windows%20%7C%20macos%20%7C%20linux-lightgrey
    :target: https://pypi.org/pypi/cleanlab/
.. |py_versions| image:: https://img.shields.io/pypi/pyversions/cleanlab.svg
    :target: https://pypi.org/pypi/cleanlab/
.. |build_status| image:: https://travis-ci.com/cgnorthcutt/cleanlab.svg?branch=master
    :target: https://travis-ci.com/cgnorthcutt/cleanlab
.. |coverage| image:: https://codecov.io/gh/cgnorthcutt/cleanlab/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/cgnorthcutt/cleanlab

``cleanlab`` **documentation** is available in `this blog post <https://l7.curtisnorthcutt.com/cleanlab-python-package>`__.

Past release notes and **future features planned**  is available `here <https://github.com/cgnorthcutt/cleanlab/blob/master/cleanlab/version.py>`__.

So fresh, so ``cleanlab`` 
=========================

``cleanlab`` finds and cleans label errors in any dataset using `state-of-the-art algorithms <https://arxiv.org/abs/1911.00068>`__ to find label errors, characterize noise, and learn in spite of it. ``cleanlab`` is fast: its built on optimized algorithms and parallelized across CPU threads automatically. ``cleanlab`` is powered by `provable guarantees <https://arxiv.org/abs/1911.00068>`__ of exact noise estimation and label error finding in realistic cases when model output probabilities are erroneous. ``cleanlab`` supports multi-label, multiclass, sparse matrices, etc. By default, ``cleanlab`` requires no hyper-parameters.

``cleanlab`` finds and cleans label errors in any dataset using state-of-the-art algorithms for learning with noisy labels by characterizing label noise. ``cleanlab`` is fast: its built on optimized algorithms and parallelized across CPU threads automatically. ``cleanlab`` implements the family of theory and algorithms called `confident learning <https://arxiv.org/abs/1911.00068>`__ with provable guarantees of exact noise estimation and label error finding (even when model output probabilities are noisy/imperfect). 

**How does confident learning work?** See:  `TUTORIAL: confident learning with just numpy and for-loops <https://github.com/cgnorthcutt/cleanlab/blob/master/examples/simplifying_confident_learning_tutorial.ipynb>`__.

``cleanlab`` supports most weak supervision tasks: multi-label, multiclass, sparse matrices, etc. 

``cleanlab`` is:

1. backed-by-theory - Provable perfect label error finding in realistic conditions.
2. fast - Non-iterative, parallelized algorithms (e.g. < 1 second to find label errors in ImageNet)
3. general - Works with any ML or deep learning framework: PyTorch, Tensorflow, MxNet, Caffe2, scikit-learn, etc.
4. unique - The only package for weak supervion with any dataset / classifier.


Find label errors with PyTorch, Tensorflow, MXNet, etc. in 1 line of code.
==========================================================================

.. code:: python

   # Compute psx (n x m matrix of predicted probabilities) on your own, with any classifier.
   # Here is an example that shows in detail how to compute psx on CIFAR-10:
   #    https://github.com/cgnorthcutt/cleanlab/tree/master/examples/cifar10
   # Be sure you compute probs in a holdout/out-of-sample manner (e.g. cross-validation)
   # Now getting label errors is trivial with cleanlab... its one line of code.
   # Label errors are ordered by likelihood of being an error. First index is most likely error.
   from cleanlab.pruning import get_noise_indices

   ordered_label_errors = get_noise_indices(
       s=numpy_array_of_noisy_labels,
       psx=numpy_array_of_predicted_probabilities,
       sorted_index_method='normalized_margin', # Orders label errors
    )

Pre-computed out-of-sample predicted probabilities for CIFAR-10 train set are available here: [`[LINK] <https://github.com/cgnorthcutt/cleanlab/tree/master/examples/cifar10>`__].
   
Learning with noisy labels in 3 lines of code!
==============================================
   
.. code:: python
   
   from cleanlab.classification import LearningWithNoisyLabels
   from sklearn.linear_model import LogisticRegression

   # Wrap around any classifier. Yup, you can use sklearn/pyTorch/Tensorflow/FastText/etc.
   lnl = LearningWithNoisyLabels(clf=LogisticRegression()) 
   lnl.fit(X=X_train_data, s=train_noisy_labels) 
   # Estimate the predictions you would have gotten by training with *no* label errors.
   predicted_test_labels = lnl.predict(X_test)


Check out these `examples <https://github.com/cgnorthcutt/cleanlab/tree/master/examples>`__ and `tests <https://github.com/cgnorthcutt/cleanlab/tree/master/tests>`__ (includes how to use pyTorch, FastText, etc.).


Installation
============

Python 2.7, 3.4, 3.5, 3.6, and 3.7 are supported. Linix, MacOS, and Windows are supported.

Stable release:

.. code-block:: bash

   $ pip install cleanlab

Developer (unstable) release:

.. code-block:: bash

   $ pip install git+https://github.com/cgnorthcutt/cleanlab.git

To install the codebase (enabling you to make modifications):

.. code-block:: bash

   $ conda update pip # if you use conda
   $ git clone https://github.com/cgnorthcutt/cleanlab.git
   $ cd cleanlab
   $ pip install -e .


Citations and Related Publications
==================================

If you use this package in your work, please cite the `confident learning paper <https://arxiv.org/abs/1911.00068>`__:

::

   @misc{northcutt2019confidentlearning,
     title={Confident Learning: Estimating Uncertainty in Dataset Labels},
     author={Curtis G. Northcutt and Lu Jiang and Isaac L. Chuang},
     year={2019},
     eprint={1911.00068},
     archivePrefix={arXiv},
     primaryClass={stat.ML}
 }



If used for binary classification, cleanlab also implements `this paper <https://arxiv.org/abs/1705.01936>`__:

::

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

Reproducing Results in  `confident learning paper <https://arxiv.org/abs/1911.00068>`__ 
=======================================================================================

See `cleanlab/examples/cifar10 <https://github.com/cgnorthcutt/cleanlab/tree/master/examples/cifar10>`__ and  `cleanlab/examples/imagenet <https://github.com/cgnorthcutt/cleanlab/tree/master/examples/imagenet>`__. You'll need to ``git clone`` `confidentlearning-reproduce <https://github.com/cgnorthcutt/confidentlearning-reproduce>`__  which contains the data and files needed to reproduce the CIFAR-10 results.


``cleanlab``: State of the Art Learning with Noisy Labels in CIFAR
------------------------------------------------------------------


A [`step-by-step guide <https://github.com/cgnorthcutt/cleanlab/tree/master/examples/cifar10>`__] to reproduce these results is available [`here <https://github.com/cgnorthcutt/cleanlab/tree/master/examples/cifar10>`__]. This guide is also helpful as a tutorial to use cleanlab on any large-scale dataset.

.. figure:: https://raw.githubusercontent.com/cgnorthcutt/cleanlab/master/img/cifar10_benchmarks.png
   :align: center
   :alt: Image depicting CIFAR10 benchmarks 

Comparison of confident learning (CL) and `cleanlab` versus seven recent methods for learning with noisy labels in CIFAR-10. Highlighted cells show CL robustness to sparsity. The five CL methods estimate label errors, remove them, then train on the cleaned data using `Co-Teaching <https://github.com/cgnorthcutt/cleanlab/blob/master/cleanlab/coteaching.py>`__.

Observe how cleanlab (CL methods) are robust to large sparsity in label noise whereas prior art tends to reduce in performance for increased sparsity, as shown by the red highlighted regions. This is important because real-world label noise is often sparse, e.g. a tiger is likely to be mislabeled as a lion, but not as most other classes like airplane, bathtub, and microwave.

``cleanlab``: Find Label Errors in ImageNet
-------------------------------------------

Use ``cleanlab`` to identify ~100,000 label errors in the 2012 ImageNet training dataset. 

.. figure:: https://raw.githubusercontent.com/cgnorthcutt/cleanlab/master/img/imagenet_train_label_errors_32.jpg
   :align: center
   :alt: Image depicting label errors in ImageNet train set 

Top label issues in the 2012 ILSVRC ImageNet train set identified using ``cleanlab``. Label Errors are boxed in red. Ontological issues in green. Multi-label images in blue.

``cleanlab``: Find Label Errors in MNIST
----------------------------------------

Use ``cleanlab`` to identify ~50 label errors in the MNIST dataset. 

.. figure:: https://raw.githubusercontent.com/cgnorthcutt/cleanlab/master/img/mnist_training_label_errors24_prune_by_noise_rate.png
   :align: center
   :alt: Image depicting label errors in MNIST train set 

Label errors of the original MNIST **train** dataset identified algorithmically using cleanlab. Depicts the 24 least confident labels, ordered left-right, top-down by increasing self-confidence (probability of belonging to the given label), denoted conf in teal. The label with the largest predicted probability is in green. Overt errors are in red.

 
``cleanlab`` Generality: View performance across 4 distributions and 9 classifiers.
-----------------------------------------------------------------------------------

Use ``cleanlab`` to learn with noisy labels regardless of dataset distribution or classifier. 

.. figure:: https://raw.githubusercontent.com/cgnorthcutt/cleanlab/master/img/demo_cleanlab_across_datasets_and_classifiers.png
   :align: center
   :alt: Image depicting generality of cleanlab across datasets and classifiers 

Each sub-figure in the figure above depicts the decision boundary learned using ``cleanlab.classification.LearningWithNoisyLabels`` in the presence of extreme (\~35%) label errors. Label errors are circled in green. Label noise is class-conditional (not simply uniformly random). Columns are organized by the classifier used, except the left-most column which depicts the ground-truth dataset distribution. Rows are organized by dataset used.

The code to reproduce this figure is available `here <https://github.com/cgnorthcutt/cleanlab/blob/master/examples/classifier_comparison.ipynb>`__.

Each figure depicts accuracy scores on a test set as decimal values: 

1. LEFT (in black): The classifier test accuracy trained with perfect labels (no label errors). 
2. MIDDLE (in blue): The classifier test accuracy trained with noisy labels using ``cleanlab``. 
3. RIGHT (in white): The baseline classifier test accuracy trained with noisy labels.

As an example, this is the noise matrix (noisy channel) *P(s \| y)* characterizing the label noise for the first dataset row in the figure. *s* represents the observed noisy labels and *y* represents the latent, true labels. The trace of this matrix is 2.6. A trace of 4 implies no label noise. A cell in this matrix is read like, "A random 38% of '3' labels were flipped to '2' labels."

======  ====  ====  ====  ==== 
p(s|y)   y=0   y=1   y=2   y=3
======  ====  ====  ====  ==== 
s=0     0.55  0.01  0.07  0.06
s=1     0.22  0.87  0.24  0.02
s=2     0.12  0.04  0.64  0.38
s=3     0.11  0.08  0.05  0.54
======  ====  ====  ====  ====


Get started with easy, quick examples.
======================================

New to **cleanlab**? Start with:

1. `Visualizing confident
   learning <https://github.com/cgnorthcutt/cleanlab/blob/master/examples/visualizing_confident_learning.ipynb>`__
2. `A simple example of learning with noisy labels on the multiclass
   Iris dataset <https://github.com/cgnorthcutt/cleanlab/blob/master/examples/iris_simple_example.ipynb>`__.

These examples show how easy it is to characterize label noise in
datasets, learn with noisy labels, identify label errors, estimate
latent priors and noisy channels, and more.

.. ..

   <!---

   

   ![Image depicting label errors in MNIST test set.](https://raw.githubusercontent.com/cgnorthcutt/cleanlab/master/img/mnist_test_label_errors8.png)
    Selected label errors in the MNIST **test** dataset ordered by increasing self-confidence (in teal).

   ## Automatically identify ~5k (of 50k) validation set label errors in ImageNet. [[link]](examples/finding_ImageNet_label_errors).
   ![Image depicting label errors in ImageNet validation set.](https://raw.githubusercontent.com/cgnorthcutt/cleanlab/master/img/imagenet_validation_label_errors_96_prune_by_noise_rate.jpg)
   Label errors in the 2012 ImageNet validation dataset identified automatically with cleanlab using a pre-trained resnet18. Displayed are the 96 least confident labels. We see that ImageNet contains numerous multi-label images, although it is used widely by the machine learning and vision communities as a single-label benchmark dataset.

   --->

Use ``cleanlab`` with any model (Tensorflow, caffe2, PyTorch, etc.)
-------------------------------------------------------------------

All of the features of the ``cleanlab`` package work with **any model**.
Yes, any model. Feel free to use PyTorch, Tensorflow, caffe2,
scikit-learn, mxnet, etc. If you use a scikit-learn classifier, all
``cleanlab`` methods will work out-of-the-box. It’s also easy to use
your favorite model from a non-scikit-learn package, just wrap your
model into a Python class that inherits the
``sklearn.base.BaseEstimator``:

.. code:: python

   from sklearn.base import BaseEstimator
   class YourFavoriteModel(BaseEstimator): # Inherits sklearn base classifier
       def __init__(self, ):
           pass
       def fit(self, X, y, sample_weight=None):
           pass
       def predict(self, X):
           pass
       def predict_proba(self, X):
           pass
       def score(self, X, y, sample_weight=None):
           pass
           
   # Now you can use your model with `cleanlab`. Here's one example:
   from cleanlab.classification import LearningWithNoisyLabels
   lnl = LearningWithNoisyLabels(clf=YourFavoriteModel())
   lnl.fit(train_data, train_labels_with_errors)

Want to see a working example? `Here’s a compliant PyTorch MNIST CNN class <https://github.com/cgnorthcutt/cleanlab/blob/master/cleanlab/models/mnist_pytorch.py#L28>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As you can see
`here <https://github.com/cgnorthcutt/cleanlab/blob/master/cleanlab/models/mnist_pytorch.py#L28>`__,
technically you don’t actually need to inherit from
``sklearn.base.BaseEstimator``, as you can just create a class that
defines .fit(), .predict(), and .predict_proba(), but inheriting makes
downstream scikit-learn applications like hyper-parameter optimization
work seamlessly. For example, the `LearningWithNoisyLabels()
model <https://github.com/cgnorthcutt/cleanlab/blob/master/cleanlab/classification.py#L48>`__
is fully compliant.

Note, some libraries exists to do this for you. For pyTorch, check out
the ``skorch`` Python library which will wrap your ``pytorch`` model
into a ``scikit-learn`` compliant model.


Documentation by Example
========================

``cleanlab`` Core Package Components
------------------------------------

1. **cleanlab/classification.py** - The LearningWithNoisyLabels() class for learning with noisy labels.
2. **cleanlab/latent_algebra.py** -	Equalities when noise information is known.
3. **cleanlab/latent_estimation.py** -	Estimates and fully characterizes all variants of label noise.
4. **cleanlab/noise_generation.py** - Generate mathematically valid synthetic noise matrices.
5. **cleanlab/polyplex.py** -	Characterizes joint distribution of label noise EXACTLY from noisy channel.
6. **cleanlab/pruning.py** - Finds the indices of the examples with label errors in a dataset.

Many of these methods have default parameters that won’t be covered
here. Check out the method docstrings for full documentation.


Methods to Standardize Research with Noisy Labels
-------------------------------------------------

``cleanlab`` supports a number of functions to generate noise for benchmarking and standardization in research. This next example shows how to generate valid, class-conditional, unformly random noisy channel matrices:

.. code:: python

    # Generate a valid (necessary conditions for learnability are met) noise matrix for any trace > 1
    from cleanlab.noise_generation import generate_noise_matrix_from_trace
    noise_matrix=generate_noise_matrix_from_trace(
        K=number_of_classes, 
        trace=float_value_greater_than_1_and_leq_K,
        py=prior_of_y_actual_labels_which_is_just_an_array_of_length_K,
        frac_zero_noise_rates=float_from_0_to_1_controlling_sparsity,
    )

    # Check if a noise matrix is valid (necessary conditions for learnability are met)
    from cleanlab.noise_generation import noise_matrix_is_valid
    is_valid=noise_matrix_is_valid(noise_matrix, prior_of_y_which_is_just_an_array_of_length_K)

For a given noise matrix, this example shows how to generate noisy labels. Methods can be seeded for reproducibility.

.. code:: python

    # Generate noisy labels using the noise_marix. Guarantees exact amount of noise in labels.
    from cleanlab.noise_generation import generate_noisy_labels
    s_noisy_labels = generate_noisy_labels(y_hidden_actual_labels, noise_matrix)
   
    # This package is a full of other useful methods for learning with noisy labels.
    # The tutorial stops here, but you don't have to. Inspect method docstrings for full docs.


Estimate the confident joint, the latent noisy channel matrix, *P(s \| y)* and inverse, *P(y \| s)*, the latent prior of the unobserved, actual true labels, *p(y)*, and the predicted probabilities.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*s* denotes a random variable that represents the observed, noisy
label and *y* denotes a random variable representing the hidden, actual
labels. Both *s* and *y* take any of the m classes as values. The
``cleanlab`` package supports different levels of granularity for
computation depending on the needs of the user. Because of this, we
support multiple alternatives, all no more than a few lines, to estimate
these latent distribution arrays, enabling the user to reduce
computation time by only computing what they need to compute, as seen in
the examples below.

Throughout these examples, you’ll see a variable called
*confident_joint*. The confident joint is an m x m matrix (m is the
number of classes) that counts, for every observed, noisy class, the
number of examples that confidently belong to every latent, hidden
class. It counts the number of examples that we are confident are
labeled correctly or incorrectly for every pair of obseved and
unobserved classes. The confident joint is an unnormalized estimate of
the complete-information latent joint distribution, *Ps,y*. Most of the
methods in the **cleanlab** package start by first estimating the
*confident_joint*. You can learn more about this in the `confident learning paper <https://arxiv.org/abs/1911.00068>`__.

Option 1: Compute the confident joint and predicted probs first. Stop if that’s all you need.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   from cleanlab.latent_estimation import estimate_latent
   from cleanlab.latent_estimation import estimate_confident_joint_and_cv_pred_proba

   # Compute the confident joint and the n x m predicted probabilities matrix (psx),
   # for n examples, m classes. Stop here if all you need is the confident joint.
   confident_joint, psx = estimate_confident_joint_and_cv_pred_proba(
       X=X_train, 
       s=train_labels_with_errors,
       clf=logreg(), # default, you can use any classifier
   )

   # Estimate latent distributions: p(y) as est_py, P(s|y) as est_nm, and P(y|s) as est_inv
   est_py, est_nm, est_inv = estimate_latent(confident_joint, s=train_labels_with_errors)

Option 2: Estimate the latent distribution matrices in a single line of code.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   from cleanlab.latent_estimation import estimate_py_noise_matrices_and_cv_pred_proba
   est_py, est_nm, est_inv, confident_joint, psx = estimate_py_noise_matrices_and_cv_pred_proba(
       X=X_train,
       s=train_labels_with_errors,
   )

Option 3: Skip computing the predicted probabilities if you already have them.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # Already have psx? (n x m matrix of predicted probabilities)
   # For example, you might get them from a pre-trained model (like resnet on ImageNet)
   # With the cleanlab package, you estimate directly with psx.
   from cleanlab.latent_estimation import estimate_py_and_noise_matrices_from_probabilities
   est_py, est_nm, est_inv, confident_joint = estimate_py_and_noise_matrices_from_probabilities(
       s=train_labels_with_errors, 
       psx=psx,
   )


Completely characterize label noise in a dataset:
-------------------------------------------------

The joint probability distribution of noisy and true labels, *P(s,y)*, completely characterizes label noise with a class-conditional *m x m* matrix. 

.. code:: python

    from cleanlab.latent_estimation import estimate_joint
    joint = compute_confident_joint(
        s=noisy_labels,
        psx=probabilities,
        confident_joint=None,  # Provide if you have it already
    )


The Polyplex
------------

The key to learning in the presence of label errors is estimating the joint distribution between the actual, hidden labels ‘*y*’ and the observed, noisy labels ‘*s*’. Using ``cleanlab`` and the theory of confident learning, we can completely characterize the trace of the latent joint distribution, *trace(P(s,y))*, given *p(y)*, for any fraction of label errors, i.e. for any trace of the noisy channel, *trace(P(s|y))*.

You can check out how to do this yourself here: 1. `Drawing
Polyplices <https://github.com/cgnorthcutt/cleanlab/blob/master/examples/drawing_polyplices.ipynb>`__ 2. `Computing
Polyplices <https://github.com/cgnorthcutt/cleanlab/blob/master/cleanlab/polyplex.py>`__

License
-------

Copyright (c) 2017-2020 Curtis Northcutt. Released under the MIT License. See `LICENSE <https://github.com/cgnorthcutt/cleanlab/blob/master/LICENSE>`__ for details.
