Introduction
============

`cleanlab <https://github.com/cleanlab/cleanlab>`_ **automatically finds and fixes errors in your ML datasets.**

| This reduces manual work needed to fix data issues and helps train reliable ML models on partially mislabeled datasets. ``cleanlab`` has already found thousands of `label errors <https://labelerrors.com>`_ in ImageNet, MNIST, and other popular ML benchmarking datasets, so let's get started with yours!

Quickstart
==========

1. Install ``cleanlab``.
------------------------

.. tabs::

   .. tab:: pip

      .. code-block:: python

         pip install cleanlab

   .. tab:: conda

      .. code-block:: python

         conda install -c cleanlab cleanlab

   .. tab:: source

      .. code-block:: python

         pip install git+https://github.com/cleanlab/cleanlab.git


2. Find label errors with ``get_noise_indices``.
------------------------------------------------

``cleanlab``'s ``get_noise_indices`` function tells you which examples in your dataset are likely mislabeled. At a minimum, it expects two inputs - your data's given labels, ``y``, and predicted probabilities, ``pyx``, from some trained model (Note: these must be out-of-sample predictions where the data points were held out from the model during training, which can be obtained via cross-validation). 

Setting ``sorted_index_method`` instructs ``cleanlab`` to return the indices of potential mislabeled examples, ordered by the likelihood of label error estimate via ``prob_given_label`` scores (predicted probability of given label according to the model).

.. code-block:: python

   from cleanlab.pruning import get_noise_indices

   ordered_label_errors = get_noise_indices(
      s=y, 
      psx=pyx,
      sorted_index_method='prob_given_label')

.. important::
   The predicted probabilities, ``pyx``, from your model **must be out-of-sample**! You should never provide predictions on the same data points used to train the model - this would reflect predictions of an overfitted model, making it unsuitable for finding label errors. To compute the out-of-sample predicted probabilities of the entire dataset, you can use cross-validation.

..
   todo - include the url for tf and torch beginner tutorials

3. Train robust models with noisy labels using ``LearningWithNoisyLabels``.
---------------------------------------------------------------------------

``cleanlab``'s ``LearningWithNoisyLabels`` adapts any classification model, ``clf``, to a more reliable one by allowing it to train directly on partially mislabeled datasets. 

When the ``.fit()`` method is called, it automatically identifies and removes any examples that are deemed "noisy" in the provided dataset before returning a final trained model.

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from cleanlab.classification import LearningWithNoisyLabels

   clf = LogisticRegression() # Here we've used sklearn's Logistic Regression model, but this can be any classifier that implements sklearn's API.
   lnl = LearningWithNoisyLabels(clf=clf)
   lnl.fit(X=X, s=y)

.. toctree::
   :hidden:
   :caption: Get Started

   Quickstart <self>
   notebooks/Image_Tut
   
.. toctree::
   :caption: API Reference
   :hidden:

   cleanlab

.. toctree::
   :caption: Links
   :hidden:

   Website <https://cleanlab.ai>
   GitHub <https://github.com/cleanlab/cleanlab>
   PyPI <https://pypi.org/project/cleanlab/>
   Conda <https://anaconda.org/Cleanlab/cleanlab>
