cleanlab documentation
======================

`cleanlab <https://github.com/cleanlab/cleanlab>`_ **automatically finds and fixes label issues in your ML datasets.**

| This reduces manual work needed to fix data errors and helps train reliable ML models on noisy real-world datasets. ``cleanlab`` has already found thousands of `label errors <https://labelerrors.com>`_ in ImageNet, MNIST, and other popular ML benchmarking datasets, so let's get started with yours!

Quickstart
==========

1. Install ``cleanlab``
-----------------------

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


2. Find label errors in your data
---------------------------------

``cleanlab``'s ``find_label_issues`` function tells you which examples in your dataset are likely mislabeled. At a minimum, it expects two inputs - your data's given labels, ``y``, and predicted probabilities, ``pred_probs``, from some trained classification model (Note: these must be out-of-sample predictions where the data points were held out from the model during training, which can be obtained via cross-validation).

Setting ``return_indices_ranked_by`` in this function instructs ``cleanlab`` to return the indices of potential mislabeled examples, ordered by how likely their given label is incorrect. This is estimated via a **label quality score**, which for example can be specified as the ``self_confidence`` (predicted probability of given label according to the model).

.. code-block:: python

   from cleanlab.pruning import find_label_issues

   ordered_label_issues = find_label_issues(
      labels=y,
      pred_probs=pred_probs,
      return_indices_ranked_by='self_confidence')

.. important::
   The predicted probabilities, ``pred_probs``, from your model **must be out-of-sample**! You should never provide predictions on the same data points used to train the model as these predictions are overfit and  unsuitable for finding label errors. To compute out-of-sample predicted probabilities for your entire dataset, you can use cross-validation.

..
   TODO - include the url for tf and torch beginner tutorials

3. Train robust models with noisy labels
----------------------------------------

``cleanlab``'s ``LearningWithNoisyLabels`` class adapts any existing (scikit-learn compatible) classification model, ``clf``, to a more reliable one by allowing it to train directly on partially mislabeled datasets.

When the ``.fit()`` method is called, it automatically removes any examples identified as "noisy" in the provided dataset and returns a model trained only on the clean data.

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from cleanlab.classification import LearningWithNoisyLabels

   clf = LogisticRegression() # Here we've used sklearn's Logistic Regression model, but this can be any classifier that implements sklearn's API.
   lnl = LearningWithNoisyLabels(clf=clf)
   lnl.fit(X=X, labels=y)

.. toctree::
   :hidden:

   Quickstart <self>

.. toctree::
   :hidden:
   :caption: Tutorials

   notebooks/Image_Tut
   notebooks/tabular_tutorial

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 3

   cleanlab/classification
   cleanlab/count
   cleanlab/rank
   cleanlab/filter
   cleanlab/noise_generation
   cleanlab/coteaching
   cleanlab/example_models/index
   cleanlab/internal/index

.. toctree::
   :caption: Links
   :hidden:

   Website <https://cleanlab.ai>
   GitHub <https://github.com/cleanlab/cleanlab>
   PyPI <https://pypi.org/project/cleanlab/>
   Conda <https://anaconda.org/Cleanlab/cleanlab>
