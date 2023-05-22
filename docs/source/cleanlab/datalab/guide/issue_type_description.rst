Datalab Issue Types
*******************


Types of issues Datalab can detect
===================================

This page describes the various types of issues that Datalab can detect in a dataset.
For each type of issue, we explain: what it says about your data if detected, why this matters, and what parameters you can optionally specify to control the detection of this issue.

Estimates for Each Issue Type
------------------------------

Datalab produces three estimates for **each** type of issue (called say `<ISSUE_NAME>` here):


1. A numeric quality score `<ISSUE_NAME>_score` (between 0 and 1) estimating how severe this issue is exhibited in each example from a dataset. Examples with higher scores are less likely to suffer from this issue. Access these via: the :py:attr:`Datalab.issues <cleanlab.datalab.datalab.Datalab.issues>` attribute or the method :py:meth:`Datalab.get_issues(\<ISSUE_NAME\>) <cleanlab.datalab.datalab.Datalab.get_issues>`.
2. A Boolean `is_<ISSUE_NAME>_issue` flag for each example from a dataset. Examples where this has value  `True` are those estimated to exhibit this issue. Access these via: the :py:attr:`Datalab.issues <cleanlab.datalab.datalab.Datalab.issues>` attribute or the method :py:meth:`Datalab.get_issues(\<ISSUE_NAME\>) <cleanlab.datalab.datalab.Datalab.get_issues>`.
3. An overall dataset quality score (between 0 and 1), quantifying how severe this issue is overall across the entire dataset. Datasets with higher scores do not exhibit this issue as badly overall. Access these via: the :py:attr:`Datalab.issue_summary <cleanlab.datalab.datalab.Datalab.issue_summary>` attribute.

**Example (for the outlier issue type)**

.. code-block:: python

    issue_name = "outlier"  # how to reference the outlier issue type in code
    issue_score = "outlier_score"  # name of column with quality scores for the outlier issue type, atypical datapoints receive lower scores 
    is_issue = "is_outlier_issue"  # name of Boolean column flagging which datapoints are considered outliers in the dataset

Datalab estimates various issues based on the four inputs below.
Each input is optional, if you do not provide it, Datalab will skip checks for those types of issues that require this input.

1. ``label_name`` - a field in the dataset that the stores the annotated class for each example in a multi-class classification dataset.
2. ``pred_probs`` - predicted class probabilities output by your trained model for each example in the dataset (these should be out-of-sample, eg. produced via cross-validation).
3. ``features`` - numeric vector representations of the features for each example in the dataset. These may be embeddings from a (pre)trained model, or just a numerically-transformed version of the original data features.
4. ``knn_graph`` - K nearest neighbor graph represented as a sparse matrix of dissimilarity values between examples in the dataset. If both `knn_graph` and `features` are provided, the `knn_graph` takes precedence, and if only `features` is provided, then a `knn_graph` is internally constructed based on the (either euclidean or cosine) distance between different examples’ features.

Label Issue
-----------

Examples whose given label is estimated to be potentially incorrect (e.g. due to annotation error) are flagged as having label issues.
Datalab estimates which examples appear mislabeled as well as a numeric label quality score for each, which quantifies the likelihood that an example is correctly labeled.

For now, Datalab can only detect label issues in a multi-class classification dataset.
The cleanlab library has alternative methods you can us to detect label issues in other types of datasets (multi-label, multi-annotator, token classification, etc.). 

Label issues are calculated based on provided `pred_probs` from a trained model. If you do not provide this argument, this type of issue will not be considered.
For the most accurate results, provide out-of-sample `pred_probs` which can be obtained for a dataset via `cross-validation <https://docs.cleanlab.ai/stable/tutorials/pred_probs_cross_val.html>`_. 
 
Having mislabeled examples in your dataset may hamper the performance of supervised learning models you train on this data.
For evaluating models or performing other types of data analytics, mislabeled examples may lead you to draw incorrect conclusions.
To handle mislabeled examples, you can either filter out the data with label issues or try to correct their labels.



Outlier Issue
-------------

Examples that are very different from the rest of the dataset (i.e. potentially out-of-distribution or rare/anomalous instances).

Outlier issues are calculated based on provided `features` , `knn_graph` , or `pred_probs`.
If you do not provide one of these arguments, this type of issue will not be considered.
This article describes how outlier issues are detected in a dataset: `https://cleanlab.ai/blog/outlier-detection/ <https://cleanlab.ai/blog/outlier-detection/>`_.

When based on `features` or `knn_graph`, the outlier quality of each example is scored inversely proportional to its distance to its K nearest neighbors in the dataset.

When based on `pred_probs`, the outlier quality of each example is scored inversely proportional to the uncertainty in its prediction.

Modeling data with outliers may have unexpected consequences.
Closely inspect them and consider removing some outliers that may be negatively affecting your models.

(Near) Duplicate Issue
----------------------

A (near) duplicate issue refers to two or more examples in a dataset that are extremely similar to each other, relative to the rest of the dataset.
The examples flagged with this issue may be exactly duplicated, or lie atypically close together when represented as vectors (i.e. feature embeddings).
Near duplicated examples may record the same information with different:

- Abbreviations, misspellings, typos, formatting, etc. in text data.
- Compression formats, resolutions, or sampling rates in image, video, and audio data.
- Minor variations which naturally occur in many types of data (e.g. translated versions of an image).

Near Duplicate issues are calculated based on provided `features` or `knn_graph`.
If you do not provide one of these arguments, this type of issue will not be considered. 

Datalab defines near duplicates as those examples whose distance to their nearest neighbor (in the space of provided `features`) in the dataset is less than `c * D`, where `0 < c < 1` is a fractional constant parameter, and `D` is the median (over the full dataset) of such distances between each example and its nearest neighbor.
Scoring the numeric quality of an example in terms of the near duplicate issue type is done proportionally to its distance to its nearest neighbor.

Including near-duplicate examples in a dataset may negatively impact a ML model's generalization performance and lead to overfitting.
In particular, it is questionable to include examples in a test dataset which are (nearly) duplicated in the corresponding training dataset.
More generally, examples which happen to be duplicated can affect the final modeling results much more than other examples — so you should at least be aware of their presence.

Non-IID Issue
-------------

Whether the dataset exhibits statistically significant violations of the IID assumption like:  changepoints or shift, drift, autocorrelation, etc. The specific form of violation considered is whether the examples are ordered such that almost adjacent examples tend to have more similar feature values. If you care about this check, do **not** first shuffle your dataset -- this check is entirely based on the sequential order of your data.

The Non-IID issue is detected based on provided `features` or `knn_graph`. If you do not provide one of these arguments, this type of issue will not be considered.

Mathematically, the **overall** Non-IID score for the dataset is defined as the p-value of a statistical test for whether the distribution of *index-gap* values differs between group A vs. group B defined as follows. For a pair of examples in the dataset `x1, x2`, we define their *index-gap* as the distance between the indices of these examples in the ordering of the data (e.g. if `x1` is the 10th example and `x2` is the 100th example in the dataset, their index-gap is 90). We construct group A from pairs of examples which are amongst the K nearest neighbors of each other, where neighbors are defined based on the provided `knn_graph` or via distances in the space of the provided vector `features` . Group B is constructed from random pairs of examples in the dataset. 

The Non-IID quality score for each example `x` is defined via a similarly computed p-value but with Group A constructed from the K nearest neighbors of `x` and Group B constructed from  random examples from the dataset paired with `x`.

The assumption that examples in a dataset are Independent and Identically Distributed (IID) is  fundamental to most proper modeling.  Detecting all possible violations of the IID assumption is statistically impossible. This issue type only considers specific forms of violation where examples that tend to be closer together in the dataset ordering also tend to have more similar feature values. This includes scenarios where:

- The underlying distribution from which examples stem is evolving over time (not identically distributed).
- An example can influence the values of future examples in the dataset (not independent).

For datasets with low non-IID score, you should consider why your data are not IID and act accordingly. For example, if the data distribution is drifting over time, consider employing a time-based train/test split instead of a random partition.  Note that shuffling the data ahead of time will ensure a good non-IID score, but this is not always a fix to the underlying problem (e.g. future deployment data may stem from a different distribution, or you may overlook the fact that examples influence each other). We thus recommend **not** shuffling your data to be able to diagnose this issue if it exists.

Optional Issue Parameters
=========================

Here is the dict of possible (**optional**) parameter values that can be specified via the argument `issue_types` to :py:meth:`Datalab.find_issues <cleanlab.datalab.datalab.Datalab.find_issues>`.
Optionally specify these to exert greater control over how issues are detected in your dataset.
Appropriate defaults are used for any parameters you do not specify, so no need to specify all of these!

.. code-block:: python

    possible_issue_types = {
        "label": label_kwargs, "outlier": outlier_kwargs, 
        "near_duplicate": near_duplicate_kwargs, "non_iid": non_iid_kwargs
    }


where the possible `kwargs` dicts for each key are described in the sections below.

Label Issue Parameters
----------------------

.. code-block:: python

    label_kwargs = {
        "health_summary_parameters": # dict of potential keyword arguments to method `dataset.health_summary()`,
        "clean_learning_kwargs": # dict of keyword arguments to constructor `CleanLearning()` including keys like: "find_label_issues_kwargs" or "label_quality_scores_kwargs",
        "thresholds": # `thresholds` argument to `CleanLearning.find_label_issues()`,
        "noise_matrix": # `noise_matrix` argument to `CleanLearning.find_label_issues()`,
        "inverse_noise_matrix": # `inverse_noise_matrix` argument to `CleanLearning.find_label_issues()`,
        "save_space": # `save_space` argument to `CleanLearning.find_label_issues()`,
        "clf_kwargs": # `clf_kwargs` argument to `CleanLearning.find_label_issues()`. Currently has no effect.,
        "validation_func": # `validation_func` argument to `CleanLearning.fit()`. Currently has no effect.,
    }

.. attention::

    ``health_summary_parameters`` and ``health_summary_kwargs`` can work in tandem to determine the arguments to be used in the call to :py:meth:`dataset.health_summary <cleanlab.dataset.health_summary>`.

.. note::

    For more information, view the source code of:  :py:class:`datalab.issue_manager.label.LabelIssueManager <cleanlab.datalab.issue_manager.label.LabelIssueManager>`.

Outlier Issue Parameters
------------------------

.. code-block:: python

    outlier_kwargs = {
        "threshold": # floating value between 0 and 1 that sets the sensitivity of the outlier detection algorithms, based on either features or pred_probs..
    	"ood_kwargs": # dict of keyword arguments to constructor `OutOfDistribution()`{
    		"params": {
    			# NOTE: Each of the following keyword arguments can also be provided outside "ood_kwargs"

    			"knn": # `knn` argument to constructor `OutOfDistribution()`. Used with features,
    			"k": # `k` argument to constructor `OutOfDistribution()`. Used with features,
    			"t": # `t` argument to constructor `OutOfDistribution()`. Used with features,
    			"adjust_pred_probs": # `adjust_pred_probs` argument to constructor `OutOfDistribution()`. Used with pred_probs,
    			"method": # `method` argument to constructor `OutOfDistribution()`. Used with pred_probs,
    			"confident_thresholds": # `confident_thresholds` argument to constructor `OutOfDistribution()`. Used with pred_probs,
    		},
    	},
    }

.. note::

    For more information, view the source code of:  :py:class:`datalab.issue_manager.outlier.OutlierIssueManager <cleanlab.datalab.issue_manager.outlier.OutlierIssueManager>`.  

Duplicate Issue Parameters
--------------------------

.. code-block:: python

    near_duplicate_kwargs = {
    	"metric": # string representing the distance metric used in nearest neighbors search (passed as argument to `NearestNeighbors`), if necessary,
    	"k": # integer representing the number of nearest neighbors for nearest neighbors search (passed as argument to `NearestNeighbors`), if necessary,
    	"threshold": # `threshold` argument to constructor of `NearDuplicateIssueManager()`. Non-negative floating value that determines the maximum distance between two examples to be considered outliers, relative to the median distance to the nearest neighbors,
    }

.. attention::

    `k` does not affect the results of the (near) duplicate search algorithm. It only affects the construction of the knn graph, if necessary.

.. note::

    For more information, view the source code of:  :py:class:`datalab.issue_manager.duplicate.NearDuplicateIssueManager <cleanlab.datalab.issue_manager.duplicate.NearDuplicateIssueManager>`. 


Non-IID Issue Parameters
------------------------

.. code-block:: python

    non_iid_kwargs = {
    	"metric": # `metric` argument to constructor of `NonIIDIssueManager`. String for the distance metric used for nearest neighbors search if necessary. `metric` argument to constructor of `sklearn.neighbors.NearestNeighbors`,
    	"k": # `k` argument to constructor of `NonIIDIssueManager`. Integer representing the number of nearest neighbors for nearest neighbors search if necessary. `n_neighbors` argument to constructor of `sklearn.neighbors.NearestNeighbors`,
        "num_permutations": # `num_permutations` argument to constructor of `NonIIDIssueManager`,
        "seed": # seed for numpy's random number generator (used for permutation tests),
        "significance_threshold": # `significance_threshold` argument to constructor of `NonIIDIssueManager`. Floating value between 0 and 1 that determines the overall signicance of non-IID issues found in the dataset.
    }

.. note::

    For more information, view the source code of:  :py:class:`datalab.issue_manager.noniid.NonIIDIssueManager <cleanlab.datalab.issue_manager.noniid.NonIIDIssueManager>`.
