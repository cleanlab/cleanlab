.. Cleanlab documentation master file, created by
   sphinx-quickstart on Mon Jan 10 07:17:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

**Cleanlab automatically finds and fixes errors in your ML datasets.**

| This reduces manual work needed to fix data issues and helps train reliable ML models on partially mislabeled datasets. Cleanlab has already found thousands of `label errors <https://labelerrors.com>`_ in ImageNet, MNIST, and other popular ML benchmarking datasets, so let's get started with yours!

.. note::
   Cleanlab currently only supports classification tasks.


Quickstart
==========

1. Install Cleanlab.
--------------------

.. tabs::

   .. code-tab:: py pip
      
      pip install cleanlab

   .. code-tab:: py conda

      conda install -c conda-forge cleanlab

2. Find label errors with ``get_noise_indices``.
------------------------------------------------

Cleanlab's ``get_noise_indices`` function tells you which examples in your dataset are likely mislabeled. At a minimum, it expects two inputs - your data's labels, ``y``, and its out-of-sample predicted probabilities, ``pyx``, computed with cross-validation. 

Setting ``sorted_index_method='prob_given_label'`` instructs it to return the positional indices of potential mislabeled examples starting with the most likely one first.

.. code-block:: python

   from cleanlab.pruning import get_noise_indices

   ordered_label_errors = get_noise_indices(
      s=y, 
      psx=pyx,
      sorted_index_method='prob_given_label')

.. important::
   ``pyx`` is the out-of-sample predicted probabilities computed with cross validation.

..
   todo - include the url for tf and torch beginner tutorials

1. Train robust models with noisy labels using ``LearningWithNoisyLabels``.
---------------------------------------------------------------------------

Cleanlab's ``LearningWithNoisyLabels`` adapts any classification model, ``clf``, to a more reliable one by allowing it to train directly on partially mislabeled datasets. 

When the ``.fit()`` method is called, it automatically identifies and removes any examples that are deemed "noisy" in the provided dataset before returning a final trained model.

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from cleanlab.classification import LearningWithNoisyLabels

   clf = LogisticRegression()
   lnl = LearningWithNoisyLabels(clf=clf)
   lnl.fit(X=X, s=y)



.. toctree::
   :hidden:
   :caption: Get Started

   Quickstart <self>


.. toctree::
   :hidden:
   :caption: API Reference

   cleanlab