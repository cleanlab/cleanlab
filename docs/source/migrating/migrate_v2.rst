How to migrate to 2.0.0 from versions pre 1.0.1
===============================================

If you previously used older versions of cleanlab,
this migration guide can help you update your existing code to work with v2.0.0 in no time!
Below we outline the major updates and code substitutions to be aware of.
The full change-log is listed in the `v2.0.0. Release Notes <https://github.com/cleanlab/cleanlab/releases/tag/v2.0.0>`_.


Function and class name changes
-------------------------------

This section covers the most commonly-used functionality from Cleanlab 1.0.

| **Old:** ``pruning.get_noise_indices(s, psx, prune_method, sorted_index_method, ...)``
| -->
| **New:** :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` ``(labels, pred_probs, filter_by, return_indices_ranked_by, ...)``

Note: ``inverse_noise_matrix`` is no longer a supported input argument, but ``confident_joint`` remains (you can easily convert between these two).

----

| **Old:** ``pruning.order_label_errors(label_errors_bool, psx, labels, sorted_index_method)``
| -->
| **New:** :py:func:`rank.order_label_issues <cleanlab.rank.order_label_issues>` ``(label_issues_mask, labels, pred_probs, rank_by, ...)``

Note: You can now alternatively use :py:func:`rank.get_label_quality_score() <cleanlab.rank.get_label_quality_score>` to numerically score the labels instead of ranking them.

----

| **Old:** ``latent_estimation.num_label_errors(labels, psx, ...)``
| -->
| **New:** :py:func:`count.num_label_issues <cleanlab.count.num_label_issues>` ``(labels, pred_probs, ...)``

Note: This is the most accurate way to estimate the raw *number* of label errors in a dataset.

----

| **Old:** ``classification.LearningWithNoisyLabels(..., prune_method)``
| -->
| **New:** :py:class:`classification.CleanLearning <cleanlab.classification.CleanLearning>` ``(..., find_label_issues_kwargs)``

Note: :py:class:`CleanLearning <cleanlab.classification.CleanLearning>` can now find label errors for you, neatly organizing them in a ``pandas.DataFrame`` as well as computing the required out-of-sample predicted probabilities. You just specify which classifier, we handle the cross-validation!


Module name changes
-------------------

Reorganized modules:

- ``cleanlab.pruning`` --> :py:mod:`cleanlab.filter`
- ``cleanlab.latent_estimation`` --> :py:mod:`cleanlab.count`
- ``cleanlab.noise_generation`` --> :py:mod:`cleanlab.benchmarking.noise_generation`
- ``cleanlab.baseline_methods`` --> incorporated into :py:mod:`cleanlab.filter`

Internal and experimental functionality, marked as such and not guaranteed to be stable between releases:

- ``cleanlab.models`` --> :py:mod:`cleanlab.experimental`
- ``cleanlab.coteaching`` --> :py:mod:`cleanlab.experimental.coteaching`
- ``cleanlab.latent_algebra`` --> :py:mod:`cleanlab.internal.latent_algebra`
- ``cleanlab.util`` --> :py:mod:`cleanlab.internal.util`


New modules
-----------

- :py:mod:`cleanlab.dataset` : New methods to print summaries of overall types of label issues most common in a dataset.
- :py:mod:`cleanlab.rank` : Moved all ranking and ordering functions from ``cleanlab.pruning`` to here. This module contains methods to score the label quality of each example and rank your data by the quality of their labels.
- :py:mod:`cleanlab.internal` and :py:mod:`cleanlab.experimental`: Moved all advanced code and utility methods to this module, including the old ``cleanlab.latent_algebra`` module. Researchers may find useful functions in here.


Removed modules
---------------

- ``cleanlab.polyplex``


Common argument and variable name changes
-----------------------------------------

Here are some common name and terminology changes in Cleanlab 2.0:

- ``s`` --> ``labels``  (the given labels in the data, which are potentially noisy)
- ``psx`` --> ``pred_probs``  (predicted probabilities output by trained classifier)
- ``label_error`` --> ``label_issue``  (a label that is likely to be wrong)

See the documentation for individual functions for details on how argument names changed.
