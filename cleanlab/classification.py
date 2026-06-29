"""
cleanlab can be used for learning with noisy labels for any dataset and model.

For regular (multi-class) classification tasks,
the `~cleanlab.classification.CleanLearning` class wraps an instance of an
sklearn classifier. The wrapped classifier must adhere to the `sklearn estimator API
<https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator>`_,
meaning it must define four functions:

* ``clf.fit(X, y, sample_weight=None)``
* ``clf.predict_proba(X)``
* ``clf.predict(X)``
* ``clf.score(X, y, sample_weight=None)``

where `X` contains data (i.e. features), `y` contains labels (with elements in 0, 1, ..., K-1,
where K is the number of classes). The first index of `X` and of `y` should correspond to the different examples in the dataset,
such that ``len(X) = len(y) = N`` (sample-size). Here `sample_weight` re-weights examples in
the loss function while training (supporting `sample_weight` in your classifier is recommended but optional).

Furthermore, your estimator should be correctly clonable via
`sklearn.base.clone <https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html>`_:
cleanlab internally creates multiple instances of the
estimator, and if you e.g. manually wrap a PyTorch model, you must ensure that
every call to the estimator's ``__init__()`` creates an independent instance of
the model (for sklearn compatibility, the weights of neural network models should typically be initialized inside of ``clf.fit()``).

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
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from typing_extensions import Self


@dataclass
class _CleanLearningConfig:
    clf: object
    seed: Optional[int]
    cv_n_folds: int
    converge_latent_estimates: bool
    pulearning: Optional[int]
    find_label_issues_kwargs: dict = field(default_factory=dict)
    label_quality_scores_kwargs: dict = field(default_factory=dict)
    verbose: bool = False
    low_memory: bool = False


@dataclass
class _CleanLearningState:
    label_issues_df: Optional[pd.DataFrame] = None
    label_issues_mask: Optional[np.ndarray] = None
    sample_weight: Optional[np.ndarray] = None
    confident_joint: Optional[np.ndarray] = None
    py: Optional[np.ndarray] = None
    ps: Optional[np.ndarray] = None
    num_classes: Optional[int] = None
    noise_matrix: Optional[np.ndarray] = None
    inverse_noise_matrix: Optional[np.ndarray] = None
    clf_kwargs: Optional[dict] = None
    clf_final_kwargs: Optional[dict] = None

from cleanlab.rank import get_label_quality_scores
from cleanlab import filter
from cleanlab.internal.util import (
    value_counts,
    compress_int_array,
    subset_X_y,
    get_num_classes,
    force_two_dimensions,
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
from cleanlab.experimental.label_issues_batched import find_label_issues_batched


class CleanLearning(BaseEstimator):  # Inherits sklearn classifier
    """
    CleanLearning = Machine Learning with cleaned data (even when training on messy, error-ridden data).

    Automated and robust learning with noisy labels using any dataset and any model. This class
    trains a model `clf` with error-prone, noisy labels as if the model had been instead trained
    on a dataset with perfect labels. It achieves this by cleaning out the error and providing
    cleaned data while training. This class is currently intended for standard (multi-class) classification tasks.

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

      See :py:mod:`cleanlab.models`, the tutorials, and examples/ repo
      for examples of sklearn wrappers, e.g. around PyTorch, Keras, or FastText.

      If the model is not sklearn-compatible by default, it might be the case that
      standard packages can adapt the model. For example, you can adapt PyTorch
      models using `skorch <https://skorch.readthedocs.io/>`_ and adapt Keras models
      using `SciKeras <https://www.adriangb.com/scikeras/>`_.

      Stores the classifier used in Confident Learning.
      Default classifier used is `sklearn.linear_model.LogisticRegression
      <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.
      Default classifier assumes that indexing along the first dimension of the dataset corresponds to
      selecting different training examples.

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
      <cleanlab.filter.find_label_issues>`. Particularly useful options include:
      `filter_by`, `frac_noise`, `min_examples_per_class` (which all impact ML accuracy),
      `n_jobs` (set this to 1 to disable multi-processing if it's causing issues).

    label_quality_scores_kwargs : dict, optional
      Keyword arguments to pass into :py:func:`rank.get_label_quality_scores
      <cleanlab.rank.get_label_quality_scores>`. Options include: `method`, `adjust_pred_probs`.

    verbose : bool, default=False
      Controls how much output is printed. Set to ``False`` to suppress print
      statements.

    low_memory: bool, default=False
      Set as ``True`` if you have a big dataset with limited memory.
      Uses :py:func:`experimental.label_issues_batched.find_label_issues_batched <cleanlab.experimental.label_issues_batched>`
      to find label issues.
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
        low_memory=False,
    ):
        self._default_clf = False
        if clf is None:
            # Use logistic regression if no classifier is provided.
            clf = LogReg(solver="lbfgs")
            self._default_clf = True

        # Make sure the given classifier has the appropriate methods defined.
        if not hasattr(clf, "fit"):
            raise ValueError("The classifier (clf) must define a .fit() method.")
        if not hasattr(clf, "predict_proba"):
            raise ValueError("The classifier (clf) must define a .predict_proba() method.")
        if not hasattr(clf, "predict"):
            raise ValueError("The classifier (clf) must define a .predict() method.")

        if seed is not None:
            np.random.seed(seed=seed)

        self._config = _CleanLearningConfig(
            clf=clf,
            seed=seed,
            cv_n_folds=cv_n_folds,
            converge_latent_estimates=converge_latent_estimates,
            pulearning=pulearning,
            find_label_issues_kwargs=find_label_issues_kwargs,
            label_quality_scores_kwargs=label_quality_scores_kwargs,
            verbose=verbose,
            low_memory=low_memory,
        )
        self._state = _CleanLearningState()

    @property
    def clf(self):
        return self._config.clf

    @clf.setter
    def clf(self, value):
        self._config.clf = value

    @property
    def seed(self):
        return self._config.seed

    @seed.setter
    def seed(self, value):
        self._config.seed = value

    @property
    def cv_n_folds(self):
        return self._config.cv_n_folds

    @cv_n_folds.setter
    def cv_n_folds(self, value):
        self._config.cv_n_folds = value

    @property
    def converge_latent_estimates(self):
        return self._config.converge_latent_estimates

    @converge_latent_estimates.setter
    def converge_latent_estimates(self, value):
        self._config.converge_latent_estimates = value

    @property
    def pulearning(self):
        return self._config.pulearning

    @pulearning.setter
    def pulearning(self, value):
        self._config.pulearning = value

    @property
    def find_label_issues_kwargs(self):
        return self._config.find_label_issues_kwargs

    @find_label_issues_kwargs.setter
    def find_label_issues_kwargs(self, value):
        self._config.find_label_issues_kwargs = value

    @property
    def label_quality_scores_kwargs(self):
        return self._config.label_quality_scores_kwargs

    @label_quality_scores_kwargs.setter
    def label_quality_scores_kwargs(self, value):
        self._config.label_quality_scores_kwargs = value

    @property
    def verbose(self):
        return self._config.verbose

    @verbose.setter
    def verbose(self, value):
        self._config.verbose = value

    @property
    def low_memory(self):
        return self._config.low_memory

    @low_memory.setter
    def low_memory(self, value):
        self._config.low_memory = value

    @property
    def label_issues_df(self):
        return self._state.label_issues_df

    @label_issues_df.setter
    def label_issues_df(self, value):
        self._state.label_issues_df = value

    @property
    def label_issues_mask(self):
        return self._state.label_issues_mask

    @label_issues_mask.setter
    def label_issues_mask(self, value):
        self._state.label_issues_mask = value

    @property
    def sample_weight(self):
        return self._state.sample_weight

    @sample_weight.setter
    def sample_weight(self, value):
        self._state.sample_weight = value

    @property
    def confident_joint(self):
        return self._state.confident_joint

    @confident_joint.setter
    def confident_joint(self, value):
        self._state.confident_joint = value

    @property
    def py(self):
        return self._state.py

    @py.setter
    def py(self, value):
        self._state.py = value

    @property
    def ps(self):
        return self._state.ps

    @ps.setter
    def ps(self, value):
        self._state.ps = value

    @property
    def num_classes(self):
        return self._state.num_classes

    @num_classes.setter
    def num_classes(self, value):
        self._state.num_classes = value

    @property
    def noise_matrix(self):
        return self._state.noise_matrix

    @noise_matrix.setter
    def noise_matrix(self, value):
        self._state.noise_matrix = value

    @property
    def inverse_noise_matrix(self):
        return self._state.inverse_noise_matrix

    @inverse_noise_matrix.setter
    def inverse_noise_matrix(self, value):
        self._state.inverse_noise_matrix = value

    @property
    def clf_kwargs(self):
        return self._state.clf_kwargs

    @clf_kwargs.setter
    def clf_kwargs(self, value):
        self._state.clf_kwargs = value

    @property
    def clf_final_kwargs(self):
        return self._state.clf_final_kwargs

    @clf_final_kwargs.setter
    def clf_final_kwargs(self, value):
        self._state.clf_final_kwargs = value

    def _validate_fit_args(self, X, labels, y, sample_weight, clf_kwargs, clf_final_kwargs):
        """Validate and normalize core inputs for ``fit()``."""

        if labels is not None and y is not None:
            raise ValueError("You must specify either `labels` or `y`, but not both.")
        if y is not None:
            labels = y
        if labels is None:
            raise ValueError("You must specify `labels`.")
        if self._default_clf:
            X = force_two_dimensions(X)

        self._state.clf_final_kwargs = {**clf_kwargs, **clf_final_kwargs}

        if "sample_weight" in clf_kwargs:
            raise ValueError(
                "sample_weight should be provided directly in fit() or in clf_final_kwargs rather than in clf_kwargs"
            )

        if sample_weight is not None and "sample_weight" not in inspect.signature(
            self.clf.fit
        ).parameters:
            raise ValueError(
                "sample_weight must be a supported fit() argument for your model in order to be specified here"
            )

        return X, labels

    def _get_label_issues_for_fit(
        self,
        X,
        labels,
        *,
        pred_probs,
        thresholds,
        noise_matrix,
        inverse_noise_matrix,
        label_issues,
        clf_kwargs,
        validation_func,
    ):
        """Return label issues for ``fit()``, computing them when needed."""

        if label_issues is None:
            if self.label_issues_df is not None and self.verbose:
                print(
                    "If you already ran self.find_label_issues() and don't want to recompute, you "
                    "should pass the label_issues in as a parameter to this function next time."
                )
            return self.find_label_issues(
                X,
                labels,
                pred_probs=pred_probs,
                thresholds=thresholds,
                noise_matrix=noise_matrix,
                inverse_noise_matrix=inverse_noise_matrix,
                clf_kwargs=clf_kwargs,
                validation_func=validation_func,
            )

        assert_valid_inputs(X, labels, pred_probs)
        if self.num_classes is None:
            if noise_matrix is not None:
                label_matrix = noise_matrix
            else:
                label_matrix = inverse_noise_matrix
            self._state.num_classes = get_num_classes(labels, pred_probs, label_matrix)
        if self.verbose:
            print("Using provided label_issues instead of finding label issues.")
            if self.label_issues_df is not None:
                print(
                    "These will overwrite self.label_issues_df and will be returned by "
                    "`self.get_label_issues()`. "
                )
        return label_issues

    def _final_fit_message(self):
        if "sample_weight" in self._state.clf_final_kwargs:
            return "Fitting final model on the clean data with custom sample_weight ..."
        return "Fitting final model on the clean data ..."

    def _prepare_final_model_sample_weight(self, labels, x_mask, labels_cleaned, sample_weight):
        if sample_weight is None:
            can_use_auto_weights = (
                "sample_weight" in inspect.signature(self.clf.fit).parameters
                and "sample_weight" not in self._state.clf_final_kwargs
                and self._state.noise_matrix is not None
            )
            if can_use_auto_weights:
                if self.verbose:
                    print(
                        "Assigning sample weights for final training based on estimated label quality."
                    )
                sample_weight_auto = np.ones(np.shape(labels_cleaned))
                for k in range(self._state.num_classes):
                    sample_weight_k = 1.0 / max(self._state.noise_matrix[k][k], 1e-3)
                    sample_weight_auto[labels_cleaned == k] = sample_weight_k

                sample_weight_expanded = np.zeros(len(labels))
                sample_weight_expanded[x_mask] = sample_weight_auto
                self._state.label_issues_df["sample_weight"] = sample_weight_expanded
                self._state.sample_weight = self._state.label_issues_df["sample_weight"]
                self._state.clf_final_kwargs["sample_weight"] = sample_weight_auto
                return

            if self.verbose and (
                "sample_weight" in inspect.signature(self.clf.fit).parameters
                and self._state.noise_matrix is None
            ):
                print(
                    "Cannot utilize sample weights for final training! "
                    "Why this matters: during final training, sample weights help account for the amount of removed data in each class. "
                    "This helps ensure the correct class prior for the learned model. "
                    "To use sample weights, you need to either provide the noise_matrix or have previously called self.find_label_issues() instead of filter.find_label_issues() which computes them for you."
                )
            return

        if "sample_weight" not in self._state.clf_final_kwargs:
            self._state.clf_final_kwargs["sample_weight"] = sample_weight[x_mask]

    def _fit_final_model(self, X, labels, label_issues, sample_weight, pred_probs):
        """Prune label issues, prepare weights, and fit the wrapped classifier."""

        self._state.label_issues_df = self._process_label_issues_arg(label_issues, labels)

        if "label_quality" not in self._state.label_issues_df.columns and pred_probs is not None:
            if self.verbose:
                print("Computing label quality scores based on given pred_probs ...")
            self._state.label_issues_df["label_quality"] = get_label_quality_scores(
                labels, pred_probs, **self.label_quality_scores_kwargs
            )

        self._state.label_issues_mask = self._state.label_issues_df["is_label_issue"].to_numpy()
        x_mask = np.invert(self._state.label_issues_mask)
        x_cleaned, labels_cleaned = subset_X_y(X, labels, x_mask)
        if self.verbose:
            print(f"Pruning {np.sum(self._state.label_issues_mask)} examples with label issues ...")
            print(f"Remaining clean data has {len(labels_cleaned)} examples.")

        self._prepare_final_model_sample_weight(labels, x_mask, labels_cleaned, sample_weight)

        if self.verbose:
            print(self._final_fit_message())

        self._config.clf.fit(x_cleaned, labels_cleaned, **self._state.clf_final_kwargs)

    def fit(
        self,
        X,
        labels=None,
        *,
        pred_probs=None,
        thresholds=None,
        noise_matrix=None,
        inverse_noise_matrix=None,
        label_issues=None,
        sample_weight=None,
        clf_kwargs={},
        clf_final_kwargs={},
        validation_func=None,
        y=None,
    ) -> "Self":
        """
        Train the model `clf` with error-prone, noisy labels as if
        the model had been instead trained on a dataset with the correct labels.
        `fit` achieves this by first training `clf` via cross-validation on the noisy data,
        using the resulting predicted probabilities to identify label issues,
        pruning the data with label issues, and finally training `clf` on the remaining clean data.

        Parameters
        ----------
        X : np.ndarray or DatasetLike
          Data features (i.e. training inputs for ML), typically an array of shape ``(N, ...)``,
          where N is the number of examples.
          Supported `DatasetLike` types beyond ``np.ndarray`` include:
          ``pd.DataFrame``, ``scipy.sparse.csr_matrix``, ``torch.utils.data.Dataset``,
          or any dataset object ``X`` that supports list-based indexing:
          ``X[index_list]`` to select a subset of training examples.
          Your classifier that this instance was initialized with,
          ``clf``, must be able to ``fit()`` and ``predict()`` data of this format.


        labels : array_like
          An array of shape ``(N,)`` of noisy classification labels, where some labels may be erroneous.
          Elements must be integers in the set 0, 1, ..., K-1, where K is the number of classes.
          Supported `array_like` types include: ``np.ndarray``, ``pd.Series``, or ``list``.

        pred_probs : np.ndarray, optional
          An array of shape ``(N, K)`` of model-predicted probabilities,
          ``P(label=k|x)``. Each row of this matrix corresponds
          to an example `x` and contains the model-predicted probabilities that
          `x` belongs to each possible class, for each of the K classes. The
          columns must be ordered such that these probabilities correspond to class 0, 1, ..., K-1.
          `pred_probs` should be :ref:`out-of-sample, eg. computed via cross-validation <pred_probs_cross_val>`.
          If provided, `pred_probs` will be used to find label issues rather than the ``clf`` classifier.

          Note
          ----
          If you are not sure, leave ``pred_probs=None`` (the default) and it
          will be computed for you using cross-validation with the provided model.

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

        noise_matrix : np.ndarray, optional
          An array of shape ``(K, K)`` representing the conditional probability
          matrix ``P(label=k_s | true label=k_y)``, the
          fraction of examples in every class, labeled as every other class.
          Assumes columns of `noise_matrix` sum to 1.

        inverse_noise_matrix : np.ndarray, optional
          An array of shape ``(K, K)`` representing the conditional probability
          matrix ``P(true label=k_y | label=k_s)``,
          the estimated fraction observed examples in each class ``k_s``
          that are mislabeled examples from every other class ``k_y``,
          Assumes columns of `inverse_noise_matrix` sum to 1.

        label_issues : pd.DataFrame or np.ndarray, optional
          Specifies the label issues for each example in dataset.
          If ``pd.DataFrame``, must be formatted as the one returned by:
          :py:meth:`CleanLearning.find_label_issues
          <cleanlab.classification.CleanLearning.find_label_issues>` or
          `~cleanlab.classification.CleanLearning.get_label_issues`.
          If ``np.ndarray``, must contain either boolean `label_issues_mask` as output by:
          default :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`,
          or integer indices as output by
          :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`
          with its `return_indices_ranked_by` argument specified.
          Providing this argument significantly reduces the time this method takes to run by
          skipping the slow cross-validation step necessary to find label issues.
          Examples identified to have label issues will be
          pruned from the data before training the final `clf` model.

          Caution: If you provide `label_issues` without having previously called
          `~cleanlab.classification.CleanLearning.find_label_issues`
          e.g. as a ``np.ndarray``, then some functionality like training with sample weights may be disabled.

        sample_weight : array_like, optional
          Array of weights with shape ``(N,)`` that are assigned to individual samples,
          assuming total number of examples in dataset is `N`.
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

        validation_func : callable, optional
          Optional callable function that takes two arguments, `X_val`, `y_val`, and returns a dict
          of keyword arguments passed into to ``clf.fit()`` which may be functions of the validation
          data in each cross-validation fold. Specifies how to map the validation data split in each
          cross-validation fold into the appropriate format to pass into `clf`'s ``fit()`` method, assuming
          ``clf.fit()`` can utilize validation data if it is appropriately passed in (eg. for early-stopping).
          Eg. if your model's ``fit()`` method is called using ``clf.fit(X, y, X_validation, y_validation)``,
          then you could set ``validation_func = f`` where
          ``def f(X_val, y_val): return {"X_validation": X_val, "y_validation": y_val}``

          Note that `validation_func` will be ignored in the final call to `clf.fit()` on the
          cleaned subset of the data. This argument is only for allowing `clf` to access the
          validation data in each cross-validation fold (eg. for early-stopping or hyperparameter-selection
          purposes). If you want to pass in validation data even in the final training call to ``clf.fit()``
          on the cleaned data subset, you should explicitly pass in that data yourself
          (eg. via `clf_final_kwargs` or `clf_kwargs`).

        y: array_like, optional
          Alternative argument that can be specified instead of `labels`.
          Specifying `y` has the same effect as specifying `labels`,
          and is offered as an alternative for compatibility with sklearn.

        Returns
        -------
        self : CleanLearning
          Fitted estimator that has all the same methods as any sklearn estimator.


          After calling ``self.fit()``, this estimator also stores extra attributes such as:

          * *self.label_issues_df*: a ``pd.DataFrame`` accessible via
          `~cleanlab.classification.CleanLearning.get_label_issues`
          of similar format as the one returned by: `~cleanlab.classification.CleanLearning.find_label_issues`.
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

        Note
        ----
        If ``CleanLearning.fit()`` does not work for your data/model, you can run the same procedure yourself:
        * Utilize :ref:`cross-validation <pred_probs_cross_val>` to get out-of-sample `pred_probs` for each example.
        * Call :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` with `pred_probs`.
        * Filter the examples with detected issues and train your model on the remaining data.
        """
        X, labels = self._validate_fit_args(
            X, labels, y, sample_weight, clf_kwargs, clf_final_kwargs
        )
        label_issues = self._get_label_issues_for_fit(
            X,
            labels,
            pred_probs=pred_probs,
            thresholds=thresholds,
            noise_matrix=noise_matrix,
            inverse_noise_matrix=inverse_noise_matrix,
            label_issues=label_issues,
            clf_kwargs=clf_kwargs,
            validation_func=validation_func,
        )

        # label_issues always overwrites self.label_issues_df. Ensure it is properly formatted:
        self._fit_final_model(X, labels, label_issues, sample_weight, pred_probs)

        if self.verbose:
            print(
                "Label issues stored in label_issues_df DataFrame accessible via: self.get_label_issues(). "
                "Call self.save_space() to delete this potentially large DataFrame attribute."
            )
        return self

    def predict(self, *args, **kwargs) -> np.ndarray:
        """Predict class labels using your wrapped classifier `clf`.
        Works just like ``clf.predict()``.

        Parameters
        ----------
        X : np.ndarray or DatasetLike
          Test data in the same format expected by your wrapped classifier.

        Returns
        -------
        class_predictions : np.ndarray
          Vector of class predictions for the test examples.
        """
        if self._default_clf:
            if args:
                X = args[0]
            elif "X" in kwargs:
                X = kwargs["X"]
                del kwargs["X"]
            else:
                raise ValueError("No input provided to predict, please provide X.")
            X = force_two_dimensions(X)
            new_args = (X,) + args[1:]
            return self.clf.predict(*new_args, **kwargs)
        else:
            return self.clf.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs) -> np.ndarray:
        """Predict class probabilities ``P(true label=k)`` using your wrapped classifier `clf`.
        Works just like ``clf.predict_proba()``.

        Parameters
        ----------
        X : np.ndarray or DatasetLike
          Test data in the same format expected by your wrapped classifier.

        Returns
        -------
        pred_probs : np.ndarray
          ``(N x K)`` array of predicted class probabilities, one row for each test example.
        """
        if self._default_clf:
            if args:
                X = args[0]
            elif "X" in kwargs:
                X = kwargs["X"]
                del kwargs["X"]
            else:
                raise ValueError("No input provided to predict, please provide X.")
            X = force_two_dimensions(X)
            new_args = (X,) + args[1:]
            return self.clf.predict_proba(*new_args, **kwargs)
        else:
            return self.clf.predict_proba(*args, **kwargs)

    def score(self, X, y, sample_weight=None) -> float:
        """Evaluates your wrapped classifier `clf`'s score on a test set `X` with labels `y`.
        Uses your model's default scoring function, or simply accuracy if your model as no ``"score"`` attribute.

        Parameters
        ----------
        X : np.ndarray or DatasetLike
          Test data in the same format expected by your wrapped classifier.

        y : array_like
          Test labels in the same format as labels previously used in ``fit()``.

        sample_weight : np.ndarray, optional
          An array of shape ``(N,)`` or ``(N, 1)`` used to weight each test example when computing the score.

        Returns
        -------
        score: float
          Number quantifying the performance of this classifier on the test data.
        """
        if self._default_clf:
            X = force_two_dimensions(X)
        if hasattr(self.clf, "score"):
            # Check if sample_weight in clf.score()
            if "sample_weight" in inspect.signature(self.clf.score).parameters:
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
        validation_func=None,
    ) -> pd.DataFrame:
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
        `~cleanlab.classification.CleanLearning.fit`.

        Note: this method computes the label issues from scratch. To access
        previously-computed label issues from this `~cleanlab.classification.CleanLearning` instance, use the
        `~cleanlab.classification.CleanLearning.get_label_issues` method.

        This is the method called to find label issues inside
        `~cleanlab.classification.CleanLearning.fit`
        and they share mostly the same parameters.

        Parameters
        ----------
        save_space : bool, optional
          If True, then returned `label_issues_df` will not be stored as attribute.
          This means some other methods like `self.get_label_issues()` will no longer work.


        For info about the **other parameters**, see the docstring of `~cleanlab.classification.CleanLearning.fit`.

        Returns
        -------
        label_issues_df : pd.DataFrame
          DataFrame with info about label issues for each example.
          Unless `save_space` argument is specified, same DataFrame is also stored as
          `self.label_issues_df` attribute accessible via
          `~cleanlab.classification.CleanLearning.get_label_issues`.
          Each row represents an example from our dataset and
          the DataFrame may contain the following columns:

          * *is_label_issue*: boolean mask for the entire dataset where ``True`` represents a label issue and ``False`` represents an example that is accurately labeled with high confidence. This column is equivalent to `label_issues_mask` output from :py:func:`filter.find_label_issues<cleanlab.filter.find_label_issues>`.
          * *label_quality*: Numeric score that measures the quality of each label (how likely it is to be correct, with lower scores indicating potentially erroneous labels).
          * *given_label*: Integer indices corresponding to the class label originally given for this example (same as `labels` input). Included here for ease of comparison against `clf` predictions, only present if "predicted_label" column is present.
          * *predicted_label*: Integer indices corresponding to the class predicted by trained `clf` model. Only present if ``pred_probs`` were provided as input or computed during label-issue-finding.
          * *sample_weight*: Numeric values used to weight examples during the final training of `clf` in `~cleanlab.classification.CleanLearning.fit`. This column may not be present after `self.find_label_issues()` but may be added after call to `~cleanlab.classification.CleanLearning.fit`. For more precise definition of sample weights, see documentation of `~cleanlab.classification.CleanLearning.fit`
        """
        X, labels = self._prepare_find_label_issues_inputs(
            X, labels, pred_probs, noise_matrix, inverse_noise_matrix
        )
        self._state.clf_kwargs = clf_kwargs

        if self.low_memory:
            label_issues_mask, pred_probs = self._find_label_issues_low_memory(
                X,
                labels,
                pred_probs,
                thresholds,
                noise_matrix,
                inverse_noise_matrix,
                validation_func,
            )
        else:
            label_issues_mask, pred_probs = self._find_label_issues_standard(
                X,
                labels,
                pred_probs,
                thresholds,
                noise_matrix,
                inverse_noise_matrix,
                validation_func,
            )

        return self._finalize_label_issues_df(labels, pred_probs, label_issues_mask, save_space)

    def get_label_issues(self) -> Optional[pd.DataFrame]:
        """
        Accessor. Returns `label_issues_df` attribute if previously already computed.
        This ``pd.DataFrame`` describes the label issues identified for each example
        (each row corresponds to an example).
        For column definitions, see the documentation of
        `~cleanlab.classification.CleanLearning.find_label_issues`.

        Returns
        -------
        label_issues_df : pd.DataFrame
          DataFrame with (precomputed) info about label issues for each example.
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

        if self._state.label_issues_df is None and self.verbose:
            print("self.label_issues_df is already empty")  # pragma: no cover
        self._state.label_issues_df = None
        self._state.sample_weight = None
        self._state.label_issues_mask = None
        self._config.find_label_issues_kwargs = None
        self._config.label_quality_scores_kwargs = None
        self._state.confident_joint = None
        self._state.py = None
        self._state.ps = None
        self._state.num_classes = None
        self._state.noise_matrix = None
        self._state.inverse_noise_matrix = None
        self._state.clf_kwargs = None
        self._state.clf_final_kwargs = None
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
            self._state.confident_joint = find_label_issues_kwargs["confident_joint"]
        self._config.find_label_issues_kwargs = find_label_issues_kwargs

    def _set_noise_state_from_confident_joint(self, labels, noise_matrix, inverse_noise_matrix):
        if self.confident_joint is not None:
            self._state.py, noise_matrix, inverse_noise_matrix = estimate_latent(
                confident_joint=self.confident_joint,
                labels=labels,
            )
        return noise_matrix, inverse_noise_matrix

    def _set_noise_state_from_matrices(self, noise_matrix, inverse_noise_matrix):
        if noise_matrix is not None:
            self._state.noise_matrix = noise_matrix
            if inverse_noise_matrix is None:
                if self.verbose:
                    print("Computing label noise estimates from provided noise matrix ...")
                self._state.py, self._state.inverse_noise_matrix = compute_py_inv_noise_matrix(
                    ps=self.ps,
                    noise_matrix=self._state.noise_matrix,
                )
        if inverse_noise_matrix is not None:
            self._state.inverse_noise_matrix = inverse_noise_matrix
            if noise_matrix is None:
                if self.verbose:
                    print("Computing label noise estimates from provided inverse noise matrix ...")
                self._state.noise_matrix = compute_noise_matrix_from_inverse(
                    ps=self.ps,
                    inverse_noise_matrix=self._state.inverse_noise_matrix,
                )

    def _set_noise_state_and_pred_probs(
        self,
        X,
        labels,
        pred_probs,
        thresholds,
        validation_func,
    ):
        if pred_probs is None:
            if self.verbose:
                print(
                    "Computing out of sample predicted probabilities via "
                    f"{self.cv_n_folds}-fold cross validation. May take a while ..."
                )
            (
                self._state.py,
                self._state.noise_matrix,
                self._state.inverse_noise_matrix,
                self._state.confident_joint,
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
                validation_func=validation_func,
            )
        else:  # pred_probs is provided by user (assumed holdout probabilities)
            if self.verbose:
                print("Computing label noise estimates from provided pred_probs ...")
            (
                self._state.py,
                self._state.noise_matrix,
                self._state.inverse_noise_matrix,
                self._state.confident_joint,
            ) = estimate_py_and_noise_matrices_from_probabilities(
                labels=labels,
                pred_probs=pred_probs,
                thresholds=thresholds,
                converge_latent_estimates=self.converge_latent_estimates,
            )

        return pred_probs

    def _ensure_pred_probs(self, X, labels, pred_probs, validation_func):
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
                validation_func=validation_func,
            )
        return pred_probs

    def _ensure_confident_joint(self, labels, pred_probs, thresholds):
        if self.confident_joint is None:
            self._state.confident_joint = compute_confident_joint(
                labels=labels,
                pred_probs=pred_probs,
                thresholds=thresholds,
            )

    def _apply_pulearning_adjustments(self):
        if self.num_classes == 2 and self.pulearning is not None:  # pragma: no cover
            self._state.noise_matrix[self.pulearning][1 - self.pulearning] = 0
            self._state.noise_matrix[1 - self.pulearning][1 - self.pulearning] = 1
            self._state.inverse_noise_matrix[1 - self.pulearning][self.pulearning] = 0
            self._state.inverse_noise_matrix[self.pulearning][self.pulearning] = 1
            self._state.confident_joint[self.pulearning][1 - self.pulearning] = 0
            self._state.confident_joint[1 - self.pulearning][1 - self.pulearning] = 1

    def _inject_confident_joint_kwarg(self):
        if "confident_joint" not in self.find_label_issues_kwargs.keys():
            if self.find_label_issues_kwargs.get("filter_by") != "confident_learning":
                self.find_label_issues_kwargs["confident_joint"] = self.confident_joint

    def _prepare_find_label_issues_inputs(
        self, X, labels, pred_probs, noise_matrix, inverse_noise_matrix
    ):
        """Validate inputs and initialize shared state for label-issue finding."""

        assert_valid_inputs(X, labels, pred_probs)
        labels = labels_to_array(labels)
        if noise_matrix is not None and np.trace(noise_matrix) <= 1:
            t = np.round(np.trace(noise_matrix), 2)
            raise ValueError("Trace(noise_matrix) is {}, but must exceed 1.".format(t))
        if inverse_noise_matrix is not None and (np.trace(inverse_noise_matrix) <= 1):
            t = np.round(np.trace(inverse_noise_matrix), 2)
            raise ValueError("Trace(inverse_noise_matrix) is {}. Must exceed 1.".format(t))

        if self._default_clf:
            X = force_two_dimensions(X)
        label_matrix = noise_matrix if noise_matrix is not None else inverse_noise_matrix
        self._state.num_classes = get_num_classes(labels, pred_probs, label_matrix)
        if (pred_probs is None) and (len(labels) / self._state.num_classes < self.cv_n_folds):
            raise ValueError(
                "Need more data from each class for cross-validation. "
                "Try decreasing cv_n_folds (eg. to 2 or 3) in CleanLearning()"
            )
        self._state.ps = value_counts(labels) / float(len(labels))
        return X, labels

    def _find_label_issues_low_memory(
        self,
        X,
        labels,
        pred_probs,
        thresholds,
        noise_matrix,
        inverse_noise_matrix,
        validation_func,
    ):
        """Compute label issues using the low-memory path."""

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
                validation_func=validation_func,
            )

        if self.verbose:
            print("Using predicted probabilities to identify label issues ...")

        if self.find_label_issues_kwargs:
            warnings.warn(f"`find_label_issues_kwargs` is not used when `low_memory=True`.")
        arg_values = {
            "thresholds": thresholds,
            "noise_matrix": noise_matrix,
            "inverse_noise_matrix": inverse_noise_matrix,
        }
        for arg_name, arg_val in arg_values.items():
            if arg_val is not None:
                warnings.warn(f"`{arg_name}` is not used when `low_memory=True`.")
        return find_label_issues_batched(labels, pred_probs, return_mask=True), pred_probs

    def _find_label_issues_standard(
        self,
        X,
        labels,
        pred_probs,
        thresholds,
        noise_matrix,
        inverse_noise_matrix,
        validation_func,
    ):
        """Compute label issues using the standard confident-learning path."""

        self._process_label_issues_kwargs(self.find_label_issues_kwargs)
        noise_matrix, inverse_noise_matrix = self._set_noise_state_from_confident_joint(
            labels=labels,
            noise_matrix=noise_matrix,
            inverse_noise_matrix=inverse_noise_matrix,
        )

        if noise_matrix is None and inverse_noise_matrix is None:
            pred_probs = self._set_noise_state_and_pred_probs(
                X=X,
                labels=labels,
                pred_probs=pred_probs,
                thresholds=thresholds,
                validation_func=validation_func,
            )
        else:
            self._set_noise_state_from_matrices(noise_matrix, inverse_noise_matrix)

        pred_probs = self._ensure_pred_probs(
            X=X,
            labels=labels,
            pred_probs=pred_probs,
            validation_func=validation_func,
        )
        self._ensure_confident_joint(labels=labels, pred_probs=pred_probs, thresholds=thresholds)
        self._apply_pulearning_adjustments()
        self._inject_confident_joint_kwarg()

        if self.verbose:
            print("Using predicted probabilities to identify label issues ...")
        return filter.find_label_issues(
            labels,
            pred_probs,
            **self.find_label_issues_kwargs,
        ), pred_probs

    def _finalize_label_issues_df(self, labels, pred_probs, label_issues_mask, save_space):
        """Build the returned DataFrame and update cached attributes."""

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
            self._state.label_issues_df = label_issues_df
            self._state.label_issues_mask = label_issues_df["is_label_issue"]
        elif self.verbose:
            print(  # pragma: no cover
                "Not storing label_issues as attributes since save_space was specified."
            )

        return label_issues_df

    def _process_label_issues_arg(self, label_issues, labels) -> pd.DataFrame:
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
                label_issues["given_label"].to_numpy() != labels
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
