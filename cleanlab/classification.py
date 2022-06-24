# Copyright (C) 2017-2022  Cleanlab Inc.
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with cleanlab.  If not, see <https://www.gnu.org/licenses/>.

"""
cleanlab can be used for multiclass (or multi-label) learning with noisy labels for any dataset and model.

The :py:class:`CleanLearning <cleanlab.classification.CleanLearning>` class wraps an instance of an
sklearn classifier. The wrapped classifier must adhere to the `sklearn estimator API
<https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator>`_,
meaning it must define four functions:

* ``clf.fit(X, y, sample_weight=None)``
* ``clf.predict_proba(X)``
* ``clf.predict(X)``
* ``clf.score(X, y, sample_weight=None)``

where `X` contains data, `y` contains labels (with elements in 0, 1, ..., K-1,
where K is the number of classes), and `sample_weight` re-weights examples in
the loss function while training.

Furthermore, the estimator should be correctly clonable via
`sklearn.base.clone <https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html>`_:
cleanlab internally creates multiple instances of the
estimator, and if you e.g. manually wrap a PyTorch model, you must ensure that
every call to the estimator's ``__init__()`` creates an independent instance of
the model.

Note
----
There are two new notions of confidence in this package:

1. Confident *examples* --- examples we are confident are labeled correctly.
We prune everything else. Mathematically, this means keeping the examples
with high probability of belong to their provided label class.

2. Confident *errors* --- examples we are confident are labeled erroneously.
We prune these. Mathematically, this means pruning the examples with
high probability of belong to a different class.

Examples
--------
>>> from cleanlab.classification import CleanLearning
>>> from sklearn.linear_model import LogisticRegression as LogReg
>>> cl = CleanLearning(clf=LogReg()) # Pass in any classifier.
>>> cl.fit(X_train, labels_maybe_with_errors)
>>> # Estimate the predictions as if you had trained without label issues.
>>> pred = cl.predict(X_test)

If the model is not sklearn-compatible by default, it might be the case that
standard packages can adapt the model. For example, you can adapt PyTorch
models using `skorch <https://skorch.readthedocs.io/>`_ and adapt Keras models
using `SciKeras <https://www.adriangb.com/scikeras/>`_.

If an open-source adapter doesn't already exist, you can manually wrap the
model to be sklearn-compatible. This is made easy by inheriting from
`sklearn.base.BaseEstimator
<https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>`_:

.. code:: python

    from sklearn.base import BaseEstimator

    class YourModel(BaseEstimator):
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

Note
----

* `labels` refers to the given labels in the original dataset, which may have errors
* labels must be integers in 0, 1, ..., K-1, where K is the total number of classes

Note
----

Confident learning is the state-of-the-art (`Northcutt et al., 2021 <https://jair.org/index.php/jair/article/view/12125>`_) for
weak supervision, finding label issues in datasets, learning with noisy
labels, uncertainty estimation, and more. It works with *any* classifier,
including deep neural networks. See the `clf` parameter.

Confident learning is a subfield of theory and algorithms of machine learning with noisy labels.
Cleanlab achieves state-of-the-art performance of any open-sourced implementation of confident
learning across a variety of tasks like multi-class classification, multi-label classification,
and PU learning.

Given any classifier having the `predict_proba` method, an input feature
matrix `X`, and a discrete vector of noisy labels `labels`, confident learning estimates the
classifications that would be obtained if the *true labels* had instead been provided
to the classifier during training. `labels` denotes the noisy labels instead of
the :math:`\\tilde{y}` used in confident learning paper.
"""

from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import inspect
import warnings

from cleanlab.rank import get_label_quality_scores
from cleanlab import filter
from cleanlab.internal.util import (
    value_counts,
    compress_int_array,
    subset_X_y,
    get_num_classes,
)
from cleanlab.count import (
    estimate_py_noise_matrices_and_cv_pred_proba,
    estimate_py_and_noise_matrices_from_probabilities,
    estimate_cv_predicted_probabilities,
    estimate_latent,
    compute_confident_joint,
)
from cleanlab.internal.latent_algebra import (
    compute_py_inv_noise_matrix,
    compute_noise_matrix_from_inverse,
)
from cleanlab.internal.validation import (
    assert_valid_inputs,
    labels_to_array,
)


class CleanLearning(BaseEstimator):  # Inherits sklearn classifier
    """
    CleanLearning = Machine Learning with cleaned data (even when training on messy, error-ridden data).

    Automated and robust learning with noisy labels using any dataset and any model. This class
    trains a model `clf` with error-prone, noisy labels as if the model had been instead trained
    on a dataset with perfect labels. It achieves this by cleaning out the error and providing
    cleaned data while training.

    Parameters
    ----------
    clf : estimator instance, optional
      A classifier implementing the `sklearn estimator API
      <https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator>`_,
      defining the following functions:

      * ``clf.fit(X, y, sample_weight=None)``
      * ``clf.predict_proba(X)``
      * ``clf.predict(X)``
      * ``clf.score(X, y, sample_weight=None)``

      See :py:mod:`cleanlab.experimental` for examples of sklearn wrappers,
      e.g. around PyTorch and FastText.

      If the model is not sklearn-compatible by default, it might be the case that
      standard packages can adapt the model. For example, you can adapt PyTorch
      models using `skorch <https://skorch.readthedocs.io/>`_ and adapt Keras models
      using `SciKeras <https://www.adriangb.com/scikeras/>`_.

      Stores the classifier used in Confident Learning.
      Default classifier used is `sklearn.linear_model.LogisticRegression
      <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.

    seed : int, optional
      Set the default state of the random number generator used to split
      the cross-validated folds. By default, uses `np.random` current random state.

    cv_n_folds : int, default=5
      This class needs holdout predicted probabilities for every data example
      and if not provided, uses cross-validation to compute them.
      `cv_n_folds` sets the number of cross-validation folds used to compute
      out-of-sample probabilities for each example in `X`.

    converge_latent_estimates : bool, optional
      If true, forces numerical consistency of latent estimates. Each is
      estimated independently, but they are related mathematically with closed
      form equivalences. This will iteratively enforce consistency.

    pulearning : {None, 0, 1}, default=None
      Only works for 2 class datasets. Set to the integer of the class that is
      perfectly labeled (you are certain that there are no errors in that class).

    find_label_issues_kwargs : dict, optional
      Keyword arguments to pass into :py:func:`filter.find_label_issues
      <cleanlab.filter.find_label_issues>`. Options that may especially impact
      accuracy include: `filter_by`, `frac_noise`, `min_examples_per_class`.

    label_quality_scores_kwargs : dict, optional
      Keyword arguments to pass into :py:func:`rank.get_label_quality_scores
      <cleanlab.rank.get_label_quality_scores>`. Options include: `method`, `adjust_pred_probs`.

    verbose : bool, default=True
      Controls how much output is printed. Set to ``False`` to suppress print
      statements.
    """

    def __init__(
        self,
        clf=None,
        *,
        seed=None,
        # Hyper-parameters (used by .fit() function)
        cv_n_folds=5,
        converge_latent_estimates=False,
        pulearning=None,
        find_label_issues_kwargs={},
        label_quality_scores_kwargs={},
        verbose=False,
    ):

        if clf is None:
            # Use logistic regression if no classifier is provided.
            clf = LogReg(multi_class="auto", solver="lbfgs")

        # Make sure the given classifier has the appropriate methods defined.
        if not hasattr(clf, "fit"):
            raise ValueError("The classifier (clf) must define a .fit() method.")
        if not hasattr(clf, "predict_proba"):
            raise ValueError("The classifier (clf) must define a .predict_proba() method.")
        if not hasattr(clf, "predict"):
            raise ValueError("The classifier (clf) must define a .predict() method.")

        if seed is not None:
            np.random.seed(seed=seed)

        self.clf = clf
        self.seed = seed
        self.cv_n_folds = cv_n_folds
        self.converge_latent_estimates = converge_latent_estimates
        self.pulearning = pulearning
        self.find_label_issues_kwargs = find_label_issues_kwargs
        self.label_quality_scores_kwargs = label_quality_scores_kwargs
        self.verbose = verbose
        self.label_issues_df = None
        self.label_issues_mask = None
        self.sample_weight = None
        self.confident_joint = None
        self.py = None
        self.ps = None
        self.num_classes = None
        self.noise_matrix = None
        self.inverse_noise_matrix = None
        self.clf_kwargs = None
        self.clf_final_kwargs = None

    def fit(
        self,
        X,
        labels,
        *,
        pred_probs=None,
        thresholds=None,
        noise_matrix=None,
        inverse_noise_matrix=None,
        label_issues=None,
        sample_weight=None,
        clf_kwargs={},
        clf_final_kwargs={},
    ):
        """
        Train the model `clf` with error-prone, noisy labels as if
        the model had been instead trained on a dataset with the correct labels.
        `fit` achieves this by first training `clf` via cross-validation on the noisy data,
        using the resulting predicted probabilities to identify label issues,
        pruning the data with label issues, and finally training `clf` on the remaining clean data.

        Parameters
        ----------
        X : np.array or pd.DataFrame
          Data features (i.e. training inputs for ML), typically an array of shape ``(N, ...)``,
          where N is the number of examples. Sparse matrices are supported.
          If not an array or DataFrame, then ``X`` must support list-based indexing:
          ``X[index_list]`` to select a subset of training examples.
          The classifier that this instance was initialized with,
          ``clf``, must be able to fit() and predict() data with this format.

        labels : np.array or pd.Series
          An array of shape ``(N,)`` of noisy labels, i.e. some labels may be erroneous.
          Elements must be in the set 0, 1, ..., K-1, where K is the number of classes.

        pred_probs : np.array, optional
          An array of shape ``(N, K)`` of model-predicted probabilities,
          ``P(label=k|x)``. Each row of this matrix corresponds
          to an example `x` and contains the model-predicted probabilities that
          `x` belongs to each possible class, for each of the K classes. The
          columns must be ordered such that these probabilities correspond to
          class 0, 1, ..., K-1. `pred_probs` should have been computed using 3 (or
          higher) fold cross-validation.

          Note
          ----
          If you are not sure, leave ``pred_probs=None`` (the default) and it
          will be computed for you using cross-validation with the model.

        thresholds : array_like, optional
          An array of shape ``(K, 1)`` or ``(K,)`` of per-class threshold
          probabilities, used to determine the cutoff probability necessary to
          consider an example as a given class label (see `Northcutt et al.,
          2021 <https://jair.org/index.php/jair/article/view/12125>`_, Section
          3.1, Equation 2).

          This is for advanced users only. If not specified, these are computed
          for you automatically. If an example has a predicted probability
          greater than this threshold, it is counted as having true_label =
          k. This is not used for pruning/filtering, only for estimating the
          noise rates using confident counts.

        noise_matrix : np.array, optional
          An array of shape ``(K, K)`` representing the conditional probability
          matrix ``P(label=k_s | true label=k_y)``, the
          fraction of examples in every class, labeled as every other class.
          Assumes columns of `noise_matrix` sum to 1.

        inverse_noise_matrix : np.array, optional
          An array of shape ``(K, K)`` representing the conditional probability
          matrix ``P(true label=k_y | label=k_s)``,
          the estimated fraction observed examples in each class ``k_s``
          that are mislabeled examples from every other class ``k_y``,
          Assumes columns of `inverse_noise_matrix` sum to 1.

        label_issues : pd.DataFrame or np.array, optional
          Specifies the label issues for each example in dataset.
          If ``pd.DataFrame``, must be formatted as the one returned by:
          :py:meth:`CleanLearning.find_label_issues
          <cleanlab.classification.CleanLearning.find_label_issues>` or
          :py:meth:`CleanLearning.get_label_issues
          <cleanlab.classification.CleanLearning.get_label_issues>`.
          If ``np.array``, must contain either boolean `label_issues_mask` as output by:
          default :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`,
          or integer indices as output by
          :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`
          with its `return_indices_ranked_by` argument specified.
          Providing this argument significantly reduces the time this method takes to run by
          skipping the slow cross-validation step necessary to find label issues.
          Examples identified to have label issues will be
          pruned from the data before training the final `clf` model.

          Caution: If you provide `label_issues` without having previously called
          :py:meth:`self.find_label_issues<cleanlab.classification.CleanLearning.find_label_issues>`,
          e.g. as a ``np.array``, then some functionality like training with sample weights may be disabled.

        sample_weight : array-like of shape (n_samples,), optional
          Array of weights that are assigned to individual samples.
          If not provided, samples may still be weighted by the estimated noise in the class they are labeled as.

        clf_kwargs : dict, optional
          Optional keyword arguments to pass into `clf`'s ``fit()`` method.

        clf_final_kwargs : dict, optional
          Optional extra keyword arguments to pass into the final `clf` ``fit()`` on the cleaned data
          but not the `clf` ``fit()`` in each fold of cross-validation on the noisy data.
          The final ``fit()`` will also receive `clf_kwargs`,
          but these may be overwritten by values in `clf_final_kwargs`.
          This can be useful for training differently in the final ``fit()``
          than during cross-validation.

        Returns
        -------
        CleanLearning
          ``self`` - Fitted estimator that has all the same methods as any sklearn estimator.


          After calling ``self.fit()``, this estimator also stores a few extra useful attributes, in particular
          `self.label_issues_df`: a ``pd.DataFrame`` accessible via
          :py:meth:`get_label_issues
          <cleanlab.classification.CleanLearning.get_label_issues>`
          of similar format as the one returned by: :py:meth:`CleanLearning.find_label_issues<cleanlab.classification.CleanLearning.find_label_issues>`.
          See documentation of :py:meth:`CleanLearning.find_label_issues<cleanlab.classification.CleanLearning.find_label_issues>`
          for column descriptions.
          After calling ``self.fit()``, `self.label_issues_df` may also contain an extra column:

          * *sample_weight*: Numeric values that were used to weight examples during
            the final training of `clf` in ``CleanLearning.fit()``.
            `sample_weight` column will only be present if automatic sample weights were actually used.
            These automatic weights are assigned to each example based on the class it belongs to,
            i.e. there are only num_classes unique sample_weight values.
            The sample weight for an example belonging to class k is computed as ``1 / p(given_label = k | true_label = k)``.
            This sample_weight normalizes the loss to effectively trick `clf` into learning with the distribution
            of the true labels by accounting for the noisy data pruned out prior to training on cleaned data.
            In other words, examples with label issues were removed, so this weights the data proportionally
            so that the classifier trains as if it had all the true labels,
            not just the subset of cleaned data left after pruning out the label issues.
        """

        self.clf_final_kwargs = {**clf_kwargs, **clf_final_kwargs}

        if "sample_weight" in clf_kwargs:
            raise ValueError(
                "sample_weight should be provided directly in fit() or in clf_final_kwargs rather than in clf_kwargs"
            )

        if sample_weight is not None:
            if "sample_weight" not in inspect.getfullargspec(self.clf.fit).args:
                raise ValueError(
                    "sample_weight must be a supported fit() argument for your model in order to be specified here"
                )

        if label_issues is None:
            if self.label_issues_df is not None and self.verbose:
                print(
                    "If you already ran self.find_label_issues() and don't want to recompute, you "
                    "should pass the label_issues in as a parameter to this function next time."
                )
            label_issues = self.find_label_issues(
                X,
                labels,
                pred_probs=pred_probs,
                thresholds=thresholds,
                noise_matrix=noise_matrix,
                inverse_noise_matrix=inverse_noise_matrix,
                clf_kwargs=clf_kwargs,
            )

        else:  # set args that may not have been set if `self.find_label_issues()` wasn't called yet
            assert_valid_inputs(X, labels, pred_probs)
            if self.num_classes is None:
                if noise_matrix is not None:
                    label_matrix = noise_matrix
                else:
                    label_matrix = inverse_noise_matrix
                self.num_classes = get_num_classes(labels, pred_probs, label_matrix)
            if self.verbose:
                print("Using provided label_issues instead of finding label issues.")
                if self.label_issues_df is not None:
                    print(
                        "These will overwrite self.label_issues_df and will be returned by "
                        "`self.get_label_issues()`. "
                    )

        # label_issues always overwrites self.label_issues_df. Ensure it is properly formatted:
        self.label_issues_df = self._process_label_issues_arg(label_issues, labels)

        if "label_quality" not in self.label_issues_df.columns and pred_probs is not None:
            if self.verbose:
                print("Computing label quality scores based on given pred_probs ...")
            self.label_issues_df["label_quality"] = get_label_quality_scores(
                labels, pred_probs, **self.label_quality_scores_kwargs
            )

        self.label_issues_mask = self.label_issues_df["is_label_issue"].values
        x_mask = ~self.label_issues_mask
        x_cleaned, labels_cleaned = subset_X_y(X, labels, x_mask)
        if self.verbose:
            print(f"Pruning {np.sum(self.label_issues_mask)} examples with label issues ...")
            print(f"Remaining clean data has {len(labels_cleaned)} examples.")

        if sample_weight is None:
            # Check if sample_weight in args of clf.fit()
            if (
                "sample_weight" in inspect.getfullargspec(self.clf.fit).args
                and "sample_weight" not in self.clf_final_kwargs
                and self.noise_matrix is not None
            ):
                # Re-weight examples in the loss function for the final fitting
                # such that the "apparent" original number of examples in each class
                # is preserved, even though the pruned sets may differ.
                if self.verbose:
                    print(
                        "Assigning sample weights for final training based on estimated label quality."
                    )
                sample_weight_auto = np.ones(np.shape(labels_cleaned))
                for k in range(self.num_classes):
                    sample_weight_k = 1.0 / max(
                        self.noise_matrix[k][k], 1e-3
                    )  # clip sample weights
                    sample_weight_auto[labels_cleaned == k] = sample_weight_k

                sample_weight_expanded = np.zeros(
                    len(labels)
                )  # pad pruned examples with zeros, length of original dataset
                sample_weight_expanded[x_mask] = sample_weight_auto
                # Store the sample weight for every example in the original, unfiltered dataset
                self.label_issues_df["sample_weight"] = sample_weight_expanded
                self.sample_weight = self.label_issues_df[
                    "sample_weight"
                ]  # pointer to here to avoid duplication
                self.clf_final_kwargs["sample_weight"] = sample_weight_auto
                if self.verbose:
                    print("Fitting final model on the clean data ...")
            else:
                if (
                    "sample_weight" in inspect.getfullargspec(self.clf.fit).args
                    and "sample_weight" not in self.clf_final_kwargs
                    and self.noise_matrix is None
                ):
                    print(
                        "Cannot utilize sample weights for final training. To utilize must either specify noise_matrix "
                        "or have previously called self.find_label_issues() instead of filter.find_label_issues()"
                    )

                if self.verbose:  # pragma: no cover
                    if "sample_weight" in self.clf_final_kwargs:
                        print("Fitting final model on the clean data with custom sample_weight...")
                    else:
                        print("Fitting final model on the clean data ...")

        elif sample_weight is not None and "sample_weight" not in self.clf_final_kwargs:
            self.clf_final_kwargs["sample_weight"] = sample_weight[x_mask]
            if self.verbose:
                print("Fitting final model on the clean data with custom sample_weight...")

        else:  # pragma: no cover
            if self.verbose:
                if "sample_weight" in self.clf_final_kwargs:
                    print("Fitting final model on the clean data with custom sample_weight...")
                else:
                    print("Fitting final model on the clean data ...")

        self.clf.fit(x_cleaned, labels_cleaned, **self.clf_final_kwargs)

        if self.verbose:
            print(
                "Label issues stored in label_issues_df DataFrame accessible via: self.get_label_issues(). "
                "Call self.save_space() to delete this potentially large DataFrame attribute."
            )
        return self

    def predict(self, *args, **kwargs):
        """Returns a vector of predictions.

        Parameters
        ----------
        X : np.array
          An array of shape ``(N, ...)`` of test data."""

        return self.clf.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        """Returns a vector of predicted probabilities for each example in X,
        ``P(true label=k)``.

        Parameters
        ----------
        X : np.array
          An array of shape ``(N, ...)`` of test data."""

        return self.clf.predict_proba(*args, **kwargs)

    def score(self, X, y, sample_weight=None):
        """Returns the `clf`'s score on a test set `X` with labels `y`.
        Uses the model's default scoring function.

        Parameters
        ----------
        X : np.array
          An array of shape ``(N, ...)`` of test data.

        y : np.array
          An array of shape ``(N,)`` or ``(N, 1)`` of test labels.

        sample_weight : np.array, optional
          An array of shape ``(N,)`` or ``(N, 1)`` used to weight each example when computing the score."""

        if hasattr(self.clf, "score"):

            # Check if sample_weight in clf.score()
            if "sample_weight" in inspect.getfullargspec(self.clf.score).args:
                return self.clf.score(X, y, sample_weight=sample_weight)
            else:
                return self.clf.score(X, y)
        else:
            return accuracy_score(
                y,
                self.clf.predict(X),
                sample_weight=sample_weight,
            )

    def find_label_issues(
        self,
        X=None,
        labels=None,
        *,
        pred_probs=None,
        thresholds=None,
        noise_matrix=None,
        inverse_noise_matrix=None,
        save_space=False,
        clf_kwargs={},
    ):
        """
        Identifies potential label issues in the dataset using confident learning.

        Runs cross-validation to get out-of-sample pred_probs from `clf`
        and then calls :py:func:`filter.find_label_issues
        <cleanlab.filter.find_label_issues>` to find label issues.
        These label issues are cached internally and returned in a pandas DataFrame.
        Kwargs for :py:func:`filter.find_label_issues
        <cleanlab.filter.find_label_issues>` must have already been specified
        in the initialization of this class, not here.

        Unlike :py:func:`filter.find_label_issues
        <cleanlab.filter.find_label_issues>`, which requires `pred_probs`,
        this method only requires a classifier and it can do the cross-validation for you.
        Both methods return the same boolean mask that identifies which examples have label issues.
        This is the preferred method to use if you plan to subsequently invoke:
        :py:meth:`CleanLearning.fit()
        <cleanlab.classification.CleanLearning.fit>`.

        Note: this method computes the label issues from scratch. To access
        previously-computed label issues from this :py:class:`CleanLearning
        <cleanlab.classification.CleanLearning>` instance, use the
        :py:meth:`get_label_issues
        <cleanlab.classification.CleanLearning.get_label_issues>` method.

        This is the method called to find label issues inside
        :py:meth:`CleanLearning.fit()
        <cleanlab.classification.CleanLearning.fit>`
        and they share mostly the same parameters.

        Parameters
        ----------
        save_space : bool, optional
          If True, then returned `label_issues_df` will not be stored as attribute.
          This means some other methods like `self.get_label_issues()` will no longer work.


        For info about the **other parameters**, see the docstring of :py:meth:`CleanLearning.fit()
        <cleanlab.classification.CleanLearning.fit>`.

        Returns
        -------
        pd.DataFrame
          pandas DataFrame of label issues for each example.
          Unless `save_space` argument is specified, same DataFrame is also stored as
          `self.label_issues_df` attribute accessible via
          :py:meth:`get_label_issues<cleanlab.classification.CleanLearning.get_label_issues>`.
          Each row represents an example from our dataset and
          the DataFrame may contain the following columns:

          * *is_label_issue*: boolean mask for the entire dataset where ``True`` represents a label issue and ``False`` represents an example that is accurately labeled with high confidence. This column is equivalent to `label_issues_mask` output from :py:func:`filter.find_label_issues<cleanlab.filter.find_label_issues>`.
          * *label_quality*: Numeric score that measures the quality of each label (how likely it is to be correct, with lower scores indicating potentially erroneous labels).
          * *given_label*: Integer indices corresponding to the class label originally given for this example (same as `labels` input). Included here for ease of comparison against `clf` predictions, only present if "predicted_label" column is present.
          * *predicted_label*: Integer indices corresponding to the class predicted by trained `clf` model. Only present if ``pred_probs`` were provided as input or computed during label-issue-finding.
          * *sample_weight*: Numeric values used to weight examples during the final training of `clf` in :py:meth:`CleanLearning.fit()<cleanlab.classification.CleanLearning.fit>`. This column not be present after `self.find_label_issues()` but may be added after call to :py:meth:`CleanLearning.fit()<cleanlab.classification.CleanLearning.fit>`. For more precise definition of sample weights, see documentation of :py:meth:`CleanLearning.fit()<cleanlab.classification.CleanLearning.fit>`
        """

        # Check inputs
        assert_valid_inputs(X, labels, pred_probs)
        if noise_matrix is not None and np.trace(noise_matrix) <= 1:
            t = np.round(np.trace(noise_matrix), 2)
            raise ValueError("Trace(noise_matrix) is {}, but must exceed 1.".format(t))
        if inverse_noise_matrix is not None and (np.trace(inverse_noise_matrix) <= 1):
            t = np.round(np.trace(inverse_noise_matrix), 2)
            raise ValueError("Trace(inverse_noise_matrix) is {}. Must exceed 1.".format(t))

        if noise_matrix is not None:
            label_matrix = noise_matrix
        else:
            label_matrix = inverse_noise_matrix
        self.num_classes = get_num_classes(labels, pred_probs, label_matrix)
        if len(labels) / self.num_classes < self.cv_n_folds:
            raise ValueError(
                "Need more data from each class for cross-validation. "
                "Try decreasing cv_n_folds (eg. to 2 or 3) in CleanLearning()"
            )
        # 'ps' is p(labels=k)
        self.ps = value_counts(labels) / float(len(labels))

        self.clf_kwargs = clf_kwargs
        self._process_label_issues_kwargs(self.find_label_issues_kwargs)
        # self._process_label_issues_kwargs might set self.confident_joint. If so, we should use it.
        if self.confident_joint is not None:
            self.py, noise_matrix, inv_noise_matrix = estimate_latent(
                confident_joint=self.confident_joint,
                labels=labels,
            )

        # If needed, compute noise rates (probability of class-conditional mislabeling).
        if noise_matrix is not None:
            self.noise_matrix = noise_matrix
            if inverse_noise_matrix is None:
                if self.verbose:
                    print("Computing label noise estimates from provided noise matrix ...")
                self.py, self.inverse_noise_matrix = compute_py_inv_noise_matrix(
                    ps=self.ps,
                    noise_matrix=self.noise_matrix,
                )
        if inverse_noise_matrix is not None:
            self.inverse_noise_matrix = inverse_noise_matrix
            if noise_matrix is None:
                if self.verbose:
                    print("Computing label noise estimates from provided inverse noise matrix ...")
                self.noise_matrix = compute_noise_matrix_from_inverse(
                    ps=self.ps,
                    inverse_noise_matrix=self.inverse_noise_matrix,
                )

        if noise_matrix is None and inverse_noise_matrix is None:
            if pred_probs is None:
                if self.verbose:
                    print(
                        "Computing out of sample predicted probabilities via "
                        f"{self.cv_n_folds}-fold cross validation. May take a while ..."
                    )
                (
                    self.py,
                    self.noise_matrix,
                    self.inverse_noise_matrix,
                    self.confident_joint,
                    pred_probs,
                ) = estimate_py_noise_matrices_and_cv_pred_proba(
                    X=X,
                    labels=labels,
                    clf=self.clf,
                    cv_n_folds=self.cv_n_folds,
                    thresholds=thresholds,
                    converge_latent_estimates=self.converge_latent_estimates,
                    seed=self.seed,
                    clf_kwargs=self.clf_kwargs,
                )
            else:  # pred_probs is provided by user (assumed holdout probabilities)
                if self.verbose:
                    print("Computing label noise estimates from provided pred_probs ...")
                (
                    self.py,
                    self.noise_matrix,
                    self.inverse_noise_matrix,
                    self.confident_joint,
                ) = estimate_py_and_noise_matrices_from_probabilities(
                    labels=labels,
                    pred_probs=pred_probs,
                    thresholds=thresholds,
                    converge_latent_estimates=self.converge_latent_estimates,
                )
        # If needed, compute P(label=k|x), denoted pred_probs (the predicted probabilities)
        if pred_probs is None:
            if self.verbose:
                print(
                    "Computing out of sample predicted probabilities via "
                    f"{self.cv_n_folds}-fold cross validation. May take a while ..."
                )

            pred_probs = estimate_cv_predicted_probabilities(
                X=X,
                labels=labels,
                clf=self.clf,
                cv_n_folds=self.cv_n_folds,
                seed=self.seed,
                clf_kwargs=self.clf_kwargs,
            )
        # If needed, compute the confident_joint (e.g. occurs if noise_matrix was given)
        if self.confident_joint is None:
            self.confident_joint = compute_confident_joint(
                labels=labels,
                pred_probs=pred_probs,
                thresholds=thresholds,
            )
        # if pulearning == the integer specifying the class without noise.
        if self.num_classes == 2 and self.pulearning is not None:  # pragma: no cover
            # pulearning = 1 (no error in 1 class) implies p(label=1|true_label=0) = 0
            self.noise_matrix[self.pulearning][1 - self.pulearning] = 0
            self.noise_matrix[1 - self.pulearning][1 - self.pulearning] = 1
            # pulearning = 1 (no error in 1 class) implies p(true_label=0|label=1) = 0
            self.inverse_noise_matrix[1 - self.pulearning][self.pulearning] = 0
            self.inverse_noise_matrix[self.pulearning][self.pulearning] = 1
            # pulearning = 1 (no error in 1 class) implies p(label=1,true_label=0) = 0
            self.confident_joint[self.pulearning][1 - self.pulearning] = 0
            self.confident_joint[1 - self.pulearning][1 - self.pulearning] = 1

        labels = labels_to_array(labels)
        if self.verbose:
            print("Using predicted probabilities to identify label issues ...")
        label_issues_mask = filter.find_label_issues(
            labels,
            pred_probs,
            **self.find_label_issues_kwargs,
        )
        label_quality_scores = get_label_quality_scores(
            labels, pred_probs, **self.label_quality_scores_kwargs
        )
        label_issues_df = pd.DataFrame(
            {"is_label_issue": label_issues_mask, "label_quality": label_quality_scores}
        )
        if self.verbose:
            print(f"Identified {np.sum(label_issues_mask)} examples with label issues.")

        predicted_labels = pred_probs.argmax(axis=1)
        label_issues_df["given_label"] = compress_int_array(labels, self.num_classes)
        label_issues_df["predicted_label"] = compress_int_array(predicted_labels, self.num_classes)

        if not save_space:
            if self.label_issues_df is not None and self.verbose:
                print(
                    "Overwriting previously identified label issues stored at self.label_issues_df. "
                    "self.get_label_issues() will now return the newly identified label issues. "
                )
            self.label_issues_df = label_issues_df
            self.label_issues_mask = label_issues_df[
                "is_label_issue"
            ]  # pointer to here to avoid duplication
        elif self.verbose:
            print(  # pragma: no cover
                "Not storing label_issues as attributes since save_space was specified."
            )

        return label_issues_df

    def get_label_issues(self):
        """
        Accessor. Returns `label_issues_df` attribute if previously already computed.
        This ``pd.DataFrame`` describes the label issues identified for each example
        (each row corresponds to an example).
        For column definitions, see the documentation of
        :py:meth:`CleanLearning.find_label_issues<cleanlab.classification.CleanLearning.find_label_issues>`.

        Returns
        -------
        pd.DataFrame
        """

        if self.label_issues_df is None:
            warnings.warn(
                "Label issues have not yet been computed. Run `self.find_label_issues()` or `self.fit()` first."
            )
        return self.label_issues_df

    def save_space(self):
        """
        Clears non-sklearn attributes of this estimator to save space (in-place).
        This includes the DataFrame attribute that stored label issues which may be large for big datasets.
        You may want to call this method before deploying this model (i.e. if you just care about producing predictions).
        After calling this method, certain non-prediction-related attributes/functionality will no longer be available
        (e.g. you cannot call ``self.fit()`` anymore).
        """

        if self.label_issues_df is None and self.verbose:
            print("self.label_issues_df is already empty")  # pragma: no cover
        self.label_issues_df = None
        self.sample_weight = None
        self.label_issues_mask = None
        self.find_label_issues_kwargs = None
        self.label_quality_scores_kwargs = None
        self.label_issues_df = None
        self.label_issues_mask = None
        self.sample_weight = None
        self.confident_joint = None
        self.py = None
        self.ps = None
        self.num_classes = None
        self.noise_matrix = None
        self.inverse_noise_matrix = None
        self.clf_kwargs = None
        self.clf_final_kwargs = None
        if self.verbose:
            print("Deleted non-sklearn attributes such as label_issues_df to save space.")

    def _process_label_issues_kwargs(self, find_label_issues_kwargs):
        """
        Private helper function that is used to modify the arguments to passed to
        filter.find_label_issues via the CleanLearning.find_label_issues class. Because
        this is a classification task, some default parameters change and some errors should
        be throne if certain unsupported (for classification) arguments are passed in. This method
        handles those parameters inside of find_label_issues_kwargs and throws an error if you pass
        in a kwargs argument to filter.find_label_issues that is not supported by the
        CleanLearning.find_label_issues() function.
        """

        # Defaults for CleanLearning.find_label_issues() vs filter.find_label_issues()
        DEFAULT_FIND_LABEL_ISSUES_KWARGS = {"min_examples_per_class": 10}
        find_label_issues_kwargs = {**DEFAULT_FIND_LABEL_ISSUES_KWARGS, **find_label_issues_kwargs}
        # Todo: support multi_label classification in the future and remove multi_label from list
        unsupported_kwargs = ["return_indices_ranked_by", "multi_label"]
        for unsupported_kwarg in unsupported_kwargs:
            if unsupported_kwarg in find_label_issues_kwargs:
                raise ValueError(
                    "These kwargs of `find_label_issues()` are not supported "
                    f"for `CleanLearning`: {unsupported_kwargs}"
                )
        # CleanLearning will use this to compute the noise_matrix and inverse_noise_matrix
        if "confident_joint" in find_label_issues_kwargs:
            self.confident_joint = find_label_issues_kwargs["confident_joint"]
        self.find_label_issues_kwargs = find_label_issues_kwargs

    def _process_label_issues_arg(self, label_issues, labels):
        """
        Helper method to get the label_issues input arg into a formatted DataFrame.
        """

        labels = labels_to_array(labels)
        if isinstance(label_issues, pd.DataFrame):
            if "is_label_issue" not in label_issues.columns:
                raise ValueError(
                    "DataFrame label_issues must contain column: 'is_label_issue'. "
                    "See CleanLearning.fit() documentation for label_issues column descriptions."
                )
            if len(label_issues) != len(labels):
                raise ValueError("label_issues and labels must have same length")
            if "given_label" in label_issues.columns and np.any(
                label_issues["given_label"].values != labels
            ):
                raise ValueError("labels must match label_issues['given_label']")
            return label_issues
        elif isinstance(label_issues, np.ndarray):
            if not label_issues.dtype in [np.dtype("bool"), np.dtype("int")]:
                raise ValueError("If label_issues is numpy.array, dtype must be 'bool' or 'int'.")
            if label_issues.dtype is np.dtype("bool") and label_issues.shape != labels.shape:
                raise ValueError(
                    "If label_issues is boolean numpy.array, must have same shape as labels"
                )
            if label_issues.dtype is np.dtype("int"):  # convert to boolean mask
                if len(np.unique(label_issues)) != len(label_issues):
                    raise ValueError(
                        "If label_issues.dtype is 'int', must contain unique integer indices "
                        "corresponding to examples with label issues such as output by: "
                        "filter.find_label_issues(..., return_indices_ranked_by=...)"
                    )
                issue_indices = label_issues
                label_issues = np.full(len(labels), False, dtype=bool)
                if len(issue_indices) > 0:
                    label_issues[issue_indices] = True
            return pd.DataFrame({"is_label_issue": label_issues})
        else:
            raise ValueError("label_issues must be either pandas.DataFrame or numpy.array")
