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
cleanlab package for multiclass, multi-label learning with noisy labels for any dataset and model.

The CleanLearning class wraps around an instance of a
classifier class. Your classifier must adhere to the sklearn template,
meaning it must define four functions:

* ``clf.fit(X, y, sample_weight = None)``
* ``clf.predict_proba(X)``
* ``clf.predict(X)``
* ``clf.score(X, y, sample_weight = None)``

where ``X`` (of length *n*) contains the data/examples, ``y`` (of length *n*)
contains the contains targets formatted as ``0, 1, 2, ..., num_classes-1``, and
``sample_weight`` (of length *n*) re-weights examples in the loss function while
training.

Furthermore, your estimator should be correctly clonable via
`sklearn.base.clone`: cleanlab internally creates multiple instances of your
estimator, and if you e.g. manually wrap a PyTorch model, you must ensure that
every call to your estimator's `__init__()` creates an independent instance of
the model.

Note
----
There are two new notions of confidence in this package:

1. Confident **examples** -- examples we are confident are labeled correctly
We prune everything else. Comptuationally, this means keeping the examples
with high probability of belong to their provided label class.

2. Confident **errors** -- examples we are confident are labeled erroneously.
We prune these. Comptuationally, this means pruning the examples with
high probability of belong to a different class.

Examples
--------
>>> from cleanlab.classification import CleanLearning
>>> from sklearn.linear_model import LogisticRegression as LogReg
>>> rp = CleanLearning(clf=LogReg()) # Pass in any classifier.
>>> rp.fit(X_train, labels_maybe_with_errors)
>>> # Estimate the predictions as if you had trained without label issues.
>>> pred = rp.predict(X_test)

The easiest way to use any model (Tensorflow, caffe2, PyTorch, etc.)
with ``cleanlab`` is to wrap it in a class that inherits
the ``sklearn.base.BaseEstimator``:

.. code:: python

    from sklearn.base import BaseEstimator
    class YourModel(BaseEstimator): # Inherits sklearn base classifier
        def __init__(self, ):
            pass
        def fit(self, X, y, sample_weight = None):
            pass
        def predict(self, X):
            pass
        def predict_proba(self, X):
            pass
        def score(self, X, y, sample_weight = None):
            pass

Note
----

* `labels` - The given (maybe noisy) labels in the original dataset, which may have errors.
* Class labels must be formatted as natural numbers: 0, 1, ..., num_classes-1

Note
----

Confident Learning is the state-of-the-art (Northcutt et al., 2021) for
weak supervision, finding label issues in datasets, learning with noisy
labels, uncertainty estimation, and more. It works with ANY classifier,
including deep neural networks. See clf parameter.

Confident learning is a subfield of theory and algorithms of machine learning with noisy labels.
Cleanlab achieves state-of-the-art performance of any open-sourced implementation of confident
learning across a variety of tasks like multi-class classification, multi-label classification,
and PU learning.

Given any classifier having the predict_proba() method, an input feature
matrix, `X`, and a discrete vector of noisy labels, `labels`, Confident Learning estimates the
classifications that would be obtained if the `true_labels` had instead been provided
to the classifier during training. `labels` denotes the noisy label instead of
\\tilde(y) (used in confident learning paper), for ASCII encoding reasons.
"""

from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
import numpy as np
import inspect
import warnings
from cleanlab.internal.util import (
    assert_inputs_are_valid,
    value_counts,
)
from cleanlab.count import (
    estimate_py_noise_matrices_and_cv_pred_proba,
    estimate_py_and_noise_matrices_from_probabilities,
    estimate_cv_predicted_probabilities,
    estimate_latent,
)
from cleanlab.internal.latent_algebra import (
    compute_py_inv_noise_matrix,
    compute_noise_matrix_from_inverse,
)
from cleanlab import filter


class CleanLearning(BaseEstimator):  # Inherits sklearn classifier
    """CleanLearning = Machine Learning with cleaned data (even if with messy, error-prone data).

    Automated and robust learning with noisy labels using any dataset and any model. This class
    trains a model `clf` with error-prone, noisy labels as if the model had been instead trained
    on a dataset with perfect labels. It achieves this by cleaning out the error and providing
    cleaned data while training.

    Parameters
    ----------
    clf : :obj:`sklearn.classifier` compliant class (e.g. skorch wraps around PyTorch)
      See cleanlab.example_models for examples of sklearn wrappers around, e.g. PyTorch.
      The clf object must have the following three functions defined:
      1. clf.predict_proba(X) # Predicted probabilities
      2. clf.predict(X) # Predict labels
      3. clf.fit(X, y, sample_weight) # Train classifier
      Stores the classifier used in Confident Learning.
      Default classifier used is logistic regression.

    seed : :obj:`int`, default: None
      Set the default state of the random number generator used to split
      the cross-validated folds. If None, uses np.random current random state.

    cv_n_folds : :obj:`int`
      This class needs holdout predicted probabilities for every data example
      and if not provided, uses cross-validation to compute them.
      cv_n_folds sets the number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.

    converge_latent_estimates : :obj:`bool` (Default: False)
      If true, forces numerical consistency of latent estimates. Each is
      estimated independently, but they are related mathematically with closed
      form equivalences. This will iteratively enforce consistency.

    pulearning : :obj:`int` (0 or 1, default: None)
      Only works for 2 class datasets. Set to the integer of the class that is
      perfectly labeled (certain no errors in that class).

    find_label_issues_kwargs : dict, default = {}
      Optional keyword arguments to pass into `filter.find_label_issues()`.
      Options that may especially impact accuracy include:
      `filter_by`, `frac_noise`, `min_examples_per_class`

    verbose : :obj:`int` (0 or 1, default: 1)
      Controls how much output is printed. Set = 0 to suppress print statements.
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
        verbose=1,
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
        self.verbose = verbose
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
        label_issues_mask=None,
        clf_kwargs={},
        clf_final_kwargs={},
    ):
        """This method trains the model `self.clf` with error-prone, noisy labels as if
        the model had been instead trained on a dataset with the correct labels.
        It achieves this by: first training `clf` via cross-validation on the noisy data,
        using the resulting predicted probabilities to identify label issues,
        pruning the data with label issues, and finally training `clf` on the remaining clean data.

        Parameters
        ----------
        X : :obj:`np.array`
          Input feature matrix (N, D), 2D numpy array

        labels : :obj:`np.array`
          A discrete vector of noisy labels, i.e. some labels may be erroneous.
          *Format requirements*: the labels must be in the set {0, 1, ..., num_classes-1}.

        pred_probs : :obj:`np.array` (shape (N, num_classes))
          P(label=k|x) is a matrix with num_classes model-predicted probabilities.
          Each row of this matrix corresponds to an example `x` and contains the model-predicted
          probabilities that `x` belongs to each possible class.
          The columns must be ordered such that these probabilities correspond to class 0,1,2,...
          `pred_probs` should have been computed using 3 (or higher) fold cross-validation.

          Note
          ----
          If you are not sure, leave `pred_probs = None` (default) and it will be computed
          for you using cross-validation with your model.

        thresholds : :obj:`iterable` (list or np.array) of shape (num_classes, 1)  or (num_classes,)
          P(label^=k|label=k). List of probabilities used to determine the cutoff
          predicted probability necessary to consider an example as a given
          class label.
          Default is ``None``. These are computed for you automatically.
          If an example has a predicted probability "greater" than
          this threshold, it is counted as having true_label = k. This is
          not used for pruning/filtering, only for estimating the noise rates using
          confident counts. Values in list should be between 0 and 1.

        noise_matrix : :obj:`np.array` of shape (num_classes, num_classes)
          A conditional probability matrix of the form P(label=k_s|true_label=k_y) containing
          the fraction of examples in every class, labeled as every other class.
          Assumes columns of noise_matrix sum to 1.

        inverse_noise_matrix : :obj:`np.array` of shape (num_classes, num_classes)
          A conditional probability matrix of the form P(true_label=k_y|label=k_s). Contains
          the estimated fraction observed examples in each class k_s, that are
          mislabeled examples from every other class k_y. If None, the
          inverse_noise_matrix will be computed from pred_probs and labels.
          Assumes columns of inverse_noise_matrix sum to 1.

        label_issues_mask : np.array<bool>, default = None
          A boolean mask for the entire dataset such as those returned by:
          `CleanLearning.find_label_issues()` or `filter.find_label_issues()`.
          If specified, examples corresponding to False entries will be pruned from the data before
          training the `clf` model.
          Providing this argument significantly reduces the time this method takes to run by
          skipping the slow cross validation training step necessary to pre-compute the boolean mask
          of label issues

        clf_kwargs : dict, default = {}
          Optional keyword arguments to pass into `clf` fit() method.

        clf_final_kwargs : dict, default = {}
          Optional extra keyword arguments to pass into the final `clf` fit() on the cleaned data
          but not the `clf` fit() in each fold of cross-validation on the noisy data.
          The final fit() will also receive `clf_kwargs`,
          but these may be overwritten by values in `clf_final_kwargs`.
          This can be useful for training differently in the final fit()
          than during cross-validation.

        Returns
        -------
        tuple
          (label_issues_mask, sample_weight)"""

        clf_final_kwargs = {**clf_kwargs, **clf_final_kwargs}
        self.clf_final_kwargs = clf_final_kwargs

        if label_issues_mask is None:
            label_issues_mask = self.find_label_issues(
                X,
                labels,
                pred_probs=pred_probs,
                thresholds=thresholds,
                noise_matrix=noise_matrix,
                inverse_noise_matrix=inverse_noise_matrix,
                clf_kwargs=clf_kwargs,
            )
        elif self.verbose:
            print("Using provided label_issues_mask instead of finding label issues.")

        # Always overwrites `self.label_issues_mask` and ensures same length as `X` and `labels`.
        self.label_issues_mask = label_issues_mask
        x_mask = ~self.label_issues_mask
        x_cleaned = X[x_mask]
        labels_cleaned = labels[x_mask]
        if self.verbose:
            print(f"Pruning {np.sum(self.label_issues_mask)} examples with label issues ...")
            print(f"Remaining clean data has {len(labels_cleaned)} examples.")

        # Check if sample_weight in clf.fit()
        if (
            "sample_weight" in inspect.getfullargspec(self.clf.fit).args
            and "sample_weight" not in self.clf_kwargs
        ):
            # Re-weight examples in the loss function for the final fitting
            # labels.t. the "apparent" original number of examples in each class
            # is preserved, even though the pruned sets may differ.
            if self.verbose:
                print(
                    "Assigning sample weights for final training based on estimated label quality."
                )

            self.sample_weight = np.ones(np.shape(labels_cleaned))
            for k in range(self.num_classes):
                sample_weight_k = 1.0 / max(self.noise_matrix[k][k], 1e-3)  # clip sample weights
                self.sample_weight[labels_cleaned == k] = sample_weight_k

            if self.verbose:
                print("Fitting final model on the clean data ...")
            self.clf.fit(
                x_cleaned,
                labels_cleaned,
                sample_weight=self.sample_weight,
                **self.clf_final_kwargs,
            )
        else:
            if self.verbose:
                print("Fitting final model on the clean data ...")
            # This is less accurate, but best we can do if no sample_weight.
            self.clf.fit(x_cleaned, labels_cleaned, **self.clf_final_kwargs)

        return self.clf

    def predict(self, *args, **kwargs):
        """Returns a vector of predictions.

        Parameters
        ----------
        X : :obj:`np.array` of shape (n, m)
          The test data as a feature matrix."""

        return self.clf.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        """Returns a vector of probabilities P(true_label=k)
        for each example in X.

        Parameters
        ----------
        X : :obj:`np.array` of shape (n, m)
          The test data as a feature matrix."""

        return self.clf.predict_proba(*args, **kwargs)

    def score(self, X, y, sample_weight=None):
        """Returns the clf's score on a test set X with labels y.
        Uses the model/clf's default scoring function.

        Parameters
        ----------
        X : :obj:`np.array` of shape (n, m)
          The test data as a feature matrix.

        y : :obj:`np.array<int>` of shape (n,) or (n, 1)
          The test classification labels as an array.

        sample_weight : :obj:`np.array<float>` of shape (n,) or (n, 1)
          Weights each example when computing the score / accuracy."""

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
        clf_kwargs={},
    ):
        """Runs cross-validation to get out-of-sample pred_probs from `clf`
        and then calls `filter.find_label_issues(labels, pred_probs)` to find label issues.
        The resulting label_issues_mask is saved in self.label_issues_mask and returned.
        Kwargs for `filter.find_label_issues` must have already been specified
        in the initialization of CleanLearning, not here.

        Unlike `filter.find_label_issues`, which requires `pred_probs`,
        this method only requires a classifier and it can do the cross-validation training for you.
        Both methods return the same boolean mask that identifies which examples have label issues.

        Note: This method computes the label issues from scratch, to access previously-computed
        label issues from this CleanLearning instance, use `self.get_label_issues()`.

        This is the method called to find label issues inside `CleanLearning.fit()`.
        For descriptions of the arguments, see `CleanLearning.fit()` documentation.

        Returns
        -------
        label_issues_mask : np.array<bool>
          This method returns a boolean mask for the entire dataset where True represents
          a label issue and False represents an example that is accurately labeled with high confidence.
        """

        if self.label_issues_mask is not None and self.verbose:
            print(
                "Overwriting previously identified label issues stored at self.label_issues_mask. "
                "`self.get_label_issues()` will now return the newly identified label issues. "
                "If you already ran `self.find_label_issues()` and don't want to recompute, you"
                "should pass the `label_issues_mask` in as a parameter to this function."
            )

        # Check inputs
        allow_empty_X = False if pred_probs is None else True
        assert_inputs_are_valid(X, labels, pred_probs, allow_empty_X=allow_empty_X)
        if noise_matrix is not None and np.trace(noise_matrix) <= 1:
            t = np.round(np.trace(noise_matrix), 2)
            raise ValueError("Trace(noise_matrix) is {}, but must exceed 1.".format(t))
        if inverse_noise_matrix is not None and (np.trace(inverse_noise_matrix) <= 1):
            t = np.round(np.trace(inverse_noise_matrix), 2)
            raise ValueError("Trace(inverse_noise_matrix) is {}. Must exceed 1.".format(t))

        # Number of classes
        self.num_classes = len(np.unique(labels))
        if len(labels) / self.num_classes < self.cv_n_folds:
            raise ValueError(
                "Need more data from each class for cross-validation. "
                "Try decreasing `cv_n_folds` (eg. to 2,3) in CleanLearning()"
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
                        "Computing out of sample predicted probabilites via "
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
                    "Computing out of sample predicted probabilites via "
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

        if self.verbose:
            print("Using predicted probabilities to identify label issues ...")
        # Get the boolean mask of the label issues
        self.label_issues_mask = filter.find_label_issues(
            labels,
            pred_probs,
            **self.find_label_issues_kwargs,
        )
        if self.verbose:
            print(f"Identified {np.sum(self.label_issues_mask)} examples with label issues.")

        return self.label_issues_mask

    def get_label_issues(self):
        """Accessor. Returns `self.label_issues_mask` if previously already computed."""

        if self.label_issues_mask is None:
            warnings.warn(
                "The label issues have not yet been computed. "
                "Run self.find_label_issues() or self.fit() first."
            )
        return self.label_issues_mask

    def _process_label_issues_kwargs(self, find_label_issues_kwargs):
        """Private helper function that is used to modify the arguments to passed to
        filter.find_label_issues via the CleanLearning.find_label_issues class. Because
        this is a classification task, some default parameters change and some errors should
        be throne if certain unsupported (for classification) arguments are passed in. This method
        handles those parameters inside of find_label_issues_kwargs and throws an error if you pass
        in a kwargs argument to filter.find_label_issues that is not supported by the
        CleanLearning.find_label_issues() function."""

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
