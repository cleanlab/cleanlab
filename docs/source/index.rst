cleanlab documentation
======================

`cleanlab <https://github.com/cleanlab/cleanlab>`_ **automatically detects data and label issues in your ML datasets.**

| This helps you improve your data and train reliable ML models on noisy real-world datasets. cleanlab has already found thousands of `label errors <https://labelerrors.com>`_ in ImageNet, MNIST, and other popular ML benchmarking datasets. Beyond handling label errors, this is a comprehensive open-source library implementing many data-centric AI capabilities. Start using automation to improve your data in 5 minutes!

Quickstart
==========

1. Install ``cleanlab``
-----------------------

.. tabs::

   .. tab:: pip

      .. code-block:: bash

         pip install cleanlab

      To install the package with all optional dependencies:

      .. code-block:: bash

         pip install "cleanlab[all]"

   .. tab:: conda

      .. code-block:: bash

         conda install -c cleanlab cleanlab

   .. tab:: source

      .. code-block:: bash

         pip install git+https://github.com/cleanlab/cleanlab.git

      To install the package with all optional dependencies:

      .. code-block:: bash

         pip install "git+https://github.com/cleanlab/cleanlab.git#egg=cleanlab[all]"


2. Find common issues in your data
----------------------------------

cleanlab automatically detects various issues in *any dataset that a classifier can be trained on*. The cleanlab package *works with any ML model* by operating on model outputs (predicted class probabilities or feature embeddings) -- it doesn't require that a particular model created those outputs. For any classification dataset, use your trained model to produce `pred_probs` (predicted class probabilities) and/or `feature_embeddings` (numeric vector representations of each datapoint). Then, these few lines of code can detect common real-world issues in your dataset like label errors, outliers, near duplicates, etc:

.. code-block:: python

    from cleanlab import Datalab

    lab = Datalab(data=your_dataset, label_name="column_name_of_labels")
    lab.find_issues(features=feature_embeddings, pred_probs=pred_probs)
    lab.report()  # summarize issues in dataset, how severe they are, ...


3. Handle label errors and train robust models with noisy labels
----------------------------------------------------------------

Mislabeled data is a particularly concerning issue plaguing real-world datasets. To use a scikit-learn-compatible model for classification with noisy labels, you don't need to train a model to find label issues -- you can pass the untrained model object, data, and labels into :py:meth:`CleanLearning.find_label_issues <cleanlab.classification.CleanLearning.find_label_issues>` and cleanlab will handle model training for you.

.. code-block:: python

    from cleanlab.classification import CleanLearning

    # This works with any sklearn-compatible model - just input data + labels and cleanlab will detect label issues ツ
    label_issues_info = CleanLearning(clf=sklearn_compatible_model).find_label_issues(data, labels)

:py:class:`CleanLearning <cleanlab.classification.CleanLearning>` also works with models from most standard ML frameworks by wrapping the model for scikit-learn compliance, e.g. `tensorflow/keras <tutorials/text.ipynb>`_ (using our KerasWrapperModel), `pytorch <tutorials/image.ipynb>`_ (using skorch package), etc.

:py:meth:`find_label_issues <cleanlab.classification.CleanLearning.find_label_issues>` returns a boolean mask flagging which examples have label issues and a numeric label quality score for each example quantifying our confidence that its label is correct.

Beyond standard classification tasks, cleanlab can also detect mislabeled examples in: `multi-label data <tutorials/multilabel_classification.ipynb>`_ (e.g. image/document tagging), `sequence prediction <tutorials/token_classification.ipynb>`_ (e.g. entity recognition), and `data labeled by multiple annotators <tutorials/multiannotator.ipynb>`_ (e.g. crowdsourcing).

.. important::
   Cleanlab performs better if the ``pred_probs`` from your model are **out-of-sample**. Details on how to compute out-of-sample predicted probabilities for your entire dataset are :ref:`here <pred_probs_cross_val>`.

cleanlab's :py:class:`CleanLearning <cleanlab.classification.CleanLearning>` class trains a more robust version of any existing (`scikit-learn <https://scikit-learn.org/>`_ `compatible <https://scikit-learn.org/stable/developers/develop.html>`_) classification model, `clf`, by fitting it to an automatically filtered version of your dataset with low-quality data removed. It returns a model trained only on the clean data, from which you can get predictions in the same way as your existing classifier.

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from cleanlab.classification import CleanLearning

   cl = CleanLearning(clf=LogisticRegression())  # any sklearn-compatible classifier
   cl.fit(train_data, labels)

   # Estimate the predictions you would have gotten if you trained without mislabeled data
   predictions = cl.predict(test_data)


4. Dataset curation: fix dataset-level issues
---------------------------------------------

cleanlab's `dataset <tutorials/dataset_health.ipynb>`_ module helps you deal with dataset-level issues -- :py:meth:`find overlapping classes <cleanlab.dataset.find_overlapping_classes>` (classes to merge), :py:meth:`rank class-level label quality <cleanlab.dataset.rank_classes_by_label_quality>` (classes to keep/delete), and :py:meth:`measure overall dataset health <cleanlab.dataset.overall_label_health_score>` (to track dataset quality as you make adjustments).

View all dataset-level issues in one line of code with :py:meth:`dataset.health_summary() <cleanlab.dataset.health_summary>`.

.. code-block:: python

   from cleanlab.dataset import health_summary

   health_summary(labels, pred_probs, class_names=class_names)


5. Improve your data via many other techniques
----------------------------------------------

Beyond handling label errors, cleanlab supports other data-centric AI capabilities including:

- Detecting outliers and out-of-distribution examples in both training and future test data `(tutorial) <tutorials/outliers.ipynb>`_
- Analyzing data labeled by multiple annotators to estimate consensus labels and their quality `(tutorial) <tutorials/multiannotator.ipynb>`_
- Active learning with multiple annotators to identify which data is most informative to label or re-label next  `(tutorial) <https://github.com/cleanlab/examples/blob/master/active_learning_multiannotator/active_learning.ipynb>`_


If you have questions, check out our `FAQ <tutorials/faq.ipynb>`_ and feel free to ask in `Slack <https://cleanlab.ai/slack>`_!

Contributing
------------

As cleanlab is an open-source project, we welcome contributions from the community.

Please see our `contributing guidelines <https://github.com/cleanlab/cleanlab/blob/master/CONTRIBUTING.md>`_ for more information.


.. toctree::
   :hidden:

   Quickstart <self>

.. toctree::
   :hidden:
   :caption: Tutorials

   Datalab Tutorials <tutorials/datalab/index>
   Workflows of Data-Centric AI <tutorials/indepth_overview>
   Image Classification (pytorch) <tutorials/image>
   Text Classification (transformers) <tutorials/text>
   Tabular Classification (sklearn) <tutorials/tabular>
   Audio Classification (speechbrain) <tutorials/audio>
   Object Detection (detectron2) <tutorials/object_detection>
   Find Dataset-level Issues <tutorials/dataset_health>
   Regression <tutorials/regression>
   Identifying Outliers (pytorch) <tutorials/outliers>
   Improving Consensus Labels for Multiannotator Data <tutorials/multiannotator>
   Multi-Label Classification <tutorials/multilabel_classification>
   Token Classification (text) <tutorials/token_classification>
   Semantic Segmentation (pytorch) <tutorials/segmentation>
   Object Detection (detectron2) <tutorials/object_detection>
   Predicted Probabilities via Cross Validation <tutorials/pred_probs_cross_val>
   FAQ <tutorials/faq>

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 3

   cleanlab/classification
   cleanlab/filter
   cleanlab/rank
   cleanlab/count
   cleanlab/dataset
   cleanlab/outlier
   cleanlab/multiannotator
   cleanlab/multilabel_classification/index
   cleanlab/regression/index
   cleanlab/token_classification/index
   cleanlab/segmentation/index
   cleanlab/object_detection/index
   cleanlab/datalab/index
   cleanlab/benchmarking/index
   cleanlab/models/index
   cleanlab/experimental/index
   cleanlab/internal/index

.. toctree::
   :caption: Guides
   :hidden:

   How to contribute <https://github.com/cleanlab/cleanlab/blob/master/CONTRIBUTING.md>
   Migrating to v2.x <migrating/migrate_v2>

.. toctree::
   :caption: Links
   :hidden:

   Website <https://cleanlab.ai>
   GitHub <https://github.com/cleanlab/cleanlab>
   PyPI <https://pypi.org/project/cleanlab/>
   Conda <https://anaconda.org/Cleanlab/cleanlab>
   Cleanlab Studio <https://cleanlab.ai/studio/?utm_source=github&utm_medium=docs&utm_campaign=clostostudio>
