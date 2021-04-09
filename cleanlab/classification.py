# coding: utf-8

# Copyright (c) 2017-2050 Curtis G. Northcutt
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License

# This agreement applies to this version and all previous versions of cleanlab.


# The LearningWithNoisyLabels algorithm class for multiclass learning with
# noisy labels.
# The LearningWithNoisyLabels class wraps around an instantion of a
# classifier. Your classifier must adhere to the sklearn template,
# meaning it must define four functions:
# * clf.fit(X, y, sample_weight = None)
# * clf.predict_proba(X)
# * clf.predict(X)
# * clf.score(X, y, sample_weight = None)
# 
# where 'X' (of length n) contains your data, 'y' (of length n) contains your
# targets formatted as 0, 1, 2, ..., K-1, and sample_weight (of length n) that
# reweights examples in the loss function while training.
# 
# ## Confidence
# 
# There are two new notions of confidence in this package
# 1. Confident examples -- examples we are confident are labeled correctly.
#   We prune everything else. Comptuationally, this means keeping the examples
#   with `high probability of belong to their provided label class'.
# 2. Confident errors -- examples we are confident are labeled incorrectly.
#   We prune these. Comptuationally, this means pruning the examples with
#   `high probability of belong to a different class'.
# 
# ## Example
# 
# ```python
# from cleanlab.classification import LearningWithNoisyLabels
# from sklearn.linear_model import LogisticRegression as LogReg
# 
# rp = LearningWithNoisyLabels(clf=LogReg()) # Pass in any classifier.
# rp.fit(X_train, y_may_have_label_errors)
# Estimate the predictions you would have gotten
#   had you trained without label errors.
# pred = rp.predict(X_test)
# ```
# 
# ## Notes
# 
# * s - denotes *noisy labels*. This is just dataset labels, maybe with errors.
# * Class labels (K classes) must be formatted as natural numbers: 0, 1, .., K-1
#
# ### The easiest way to use any model (Tensorflow, caffe2, PyTorch, etc.)
#   with `cleanlab` is to wrap it in a class that inherets
#   the `sklearn.base.BaseEstimator`:
# ```python
# from sklearn.base import BaseEstimator
# class YourModel(BaseEstimator): # Inherits sklearn base classifier
#     def __init__(self, ):
#         pass
#     def fit(self, X, y, sample_weight = None):
#         pass
#     def predict(self, X):
#         pass
#     def predict_proba(self, X):
#         pass
#     def score(self, X, y, sample_weight = None):
#         pass
# ```

from __future__ import (
    print_function, absolute_import, division, unicode_literals, with_statement)

from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
import numpy as np
import inspect
import multiprocessing
from cleanlab.util import (
    assert_inputs_are_valid,
    value_counts,
)
from cleanlab.latent_estimation import (
    estimate_py_noise_matrices_and_cv_pred_proba,
    estimate_py_and_noise_matrices_from_probabilities,
    estimate_cv_predicted_probabilities,
)
from cleanlab.latent_algebra import (
    compute_py_inv_noise_matrix,
    compute_noise_matrix_from_inverse,
)
from cleanlab.pruning import get_noise_indices


class LearningWithNoisyLabels(BaseEstimator):  # Inherits sklearn classifier
    """Confident Learning is the state-of-the-art (Northcutt et al. 2019) for
      weak supervision, finding label errors in datasets, learning with noisy
      labels, uncertainty estimation, and omre. It works with ANY classifier,
      including deep neural networks. See clf parameter.
    This subfield of machine learning is referred to as Confident Learning.
    Confident Learning also achieves state-of-the-art performance for binary
      classification with noisy labels and positive-unlabeled
      learning (PU learning) where a subset of positive examples is given and
      all other examples are unlabeled and assumed to be negative examples.
    Confident Learning works by "learning from confident examples." Confident
      examples are identified as examples with high predicted probability
      for their training label.
    Given any classifier having the predict_proba() method, an input feature
      matrix, X, and a discrete vector of labels, s, which may contain
      mislabeling, Confident Learning estimates the classifications that would
      be obtained if the hidden, true labels, y, had instead been provided to
      the classifier during training. "s" denotes the noisy label instead of
      \tilde(y), for ASCII encoding reasons.

    Parameters
    ----------
    clf : sklearn.classifier compliant class (e.g. skorch wraps around PyTorch)
      See cleanlab.models for examples of sklearn wrappers around, e.g. PyTorch.
      The clf object must have the following three functions defined:
      1. clf.predict_proba(X) # Predicted probabilities
      2. clf.predict(X) # Predict labels
      3. clf.fit(X, y, sample_weight) # Train classifier
      Stores the classifier used inConfident Learning.
      Default classifier used is logistic regression.

    seed : int (default = None)
      Set the default state of the random number generator used to split
      the cross-validated folds. If None, uses np.random current random state.

    cv_n_folds : int
      This class needs holdout predicted probabilities for every data example
      and if not provided, uses cross-validation to compute them.
      cv_n_folds sets the number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.

    prune_method : str (default: 'prune_by_noise_rate')
      Available options: 'prune_by_class', 'prune_by_noise_rate', or 'both'.
      This str determines the method used for pruning.
      1. 'prune_by_noise_rate': works by removing examples with
        *high probability* of being mislabeled for every non-diagonal in the
        `prune_counts_matrix` (see pruning.py).
      2. 'prune_by_class': works by removing the examples with *smallest
        probability* of belonging to their given class label for every class.
      3. 'both': Finds the examples satisfying (1) AND (2) and removes their
        set conjunction.

    converge_latent_estimates : bool (Default: False)
      If true, forces numerical consistency of latent estimates. Each is
      estimated independently, but they are related mathematically with closed
      form equivalences. This will iteratively enforce consistency.

    pulearning : int (0 or 1, default: None)
      Only works for 2 class datasets. Set to the integer of the class that is
      perfectly labeled (certain no errors in that class).

    n_jobs : int (Windows users may see a speed-up with n_jobs = 1)
      Number of processing threads used by multiprocessing. Default None
      sets to the number of processing threads on your CPU.
      Set this to 1 to REMOVE parallel processing (if its causing issues)."""

    def __init__(
            self,
            clf=None,
            seed=None,
            # Hyper-parameters (used by .fit() function)
            cv_n_folds=5,
            prune_method='prune_by_noise_rate',
            converge_latent_estimates=False,
            pulearning=None,
            n_jobs=None,
    ):

        if clf is None:
            # Use logistic regression if no classifier is provided.
            clf = LogReg(multi_class='auto', solver='lbfgs')

        # Make sure the given classifier has the appropriate methods defined.
        if not hasattr(clf, "fit"):
            raise ValueError(
                'The classifier (clf) must define a .fit() method.')
        if not hasattr(clf, "predict_proba"):
            raise ValueError(
                'The classifier (clf) must define a .predict_proba() method.')
        if not hasattr(clf, "predict"):
            raise ValueError(
                'The classifier (clf) must define a .predict() method.')

        if seed is not None:
            np.random.seed(seed=seed)

        # Set-up number of multiprocessing threads used by get_noise_indices()
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()
        else:
            assert (n_jobs >= 1)

        self.clf = clf
        self.seed = seed
        self.cv_n_folds = cv_n_folds
        self.prune_method = prune_method
        self.converge_latent_estimates = converge_latent_estimates
        self.pulearning = pulearning
        self.n_jobs = n_jobs
        self.noise_mask = None
        self.sample_weight = None
        self.confident_joint = None
        self.py = None
        self.ps = None
        self.K = None
        self.noise_matrix = None
        self.inverse_noise_matrix = None

    def fit(
            self,
            X,
            s,
            psx=None,
            thresholds=None,
            noise_matrix=None,
            inverse_noise_matrix=None,
    ):
        """This method implements the confident learning. It counts examples
        that are likely labeled correctly and incorrectly and uses their ratio
        to create a predicted confusion matrix.
        This function fits the classifier (self.clf) to (X, s) accounting for
        the noise in both the positive and negative sets.

        Parameters
        ----------
        X : np.array
          Input feature matrix (N, D), 2D numpy array

        s : np.array
          A binary vector of labels, s, which may contain mislabeling.

        psx : np.array (shape (N, K))
          P(s=k|x) is a matrix with K (noisy) probabilities for each of the N
          examples x.
          This is the probability distribution over all K classes, for each
          example, regarding whether the example has label s==k P(s=k|x). psx
          should have been computed using 3 (or higher) fold cross-validation.
          If you are not sure, leave psx = None (default) and
          it will be computed for you using cross-validation.

        thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
          P(s^=k|s=k). List of probabilities used to determine the cutoff
          predicted probability necessary to consider an example as a given
          class label.
          Default is None. These are computed for you automatically.
          If an example has a predicted probability "greater" than
          this threshold, it is counted as having hidden label y = k. This is
          not used for pruning, only for estimating the noise rates using
          confident counts. Values in list should be between 0 and 1.

        noise_matrix : np.array of shape (K, K), K = number of classes
          A conditional probablity matrix of the form P(s=k_s|y=k_y) containing
          the fraction of examples in every class, labeled as every other class.
          Assumes columns of noise_matrix sum to 1.

        inverse_noise_matrix : np.array of shape (K, K), K = number of classes
          A conditional probablity matrix of the form P(y=k_y|s=k_s). Contains
          the estimated fraction observed examples in each class k_s, that are
          mislabeled examples from every other class k_y. If None, the
          inverse_noise_matrix will be computed from psx and s.
          Assumes columns of inverse_noise_matrix sum to 1.

        Output
        ------
          Returns (noise_mask, sample_weight)"""

        # Check inputs
        assert_inputs_are_valid(X, s, psx)
        if noise_matrix is not None and np.trace(noise_matrix) <= 1:
            t = np.round(np.trace(noise_matrix), 2)
            raise ValueError(
                "Trace(noise_matrix) is {}, but must exceed 1.".format(t))
        if inverse_noise_matrix is not None and (
                np.trace(inverse_noise_matrix) <= 1
        ):
            t = np.round(np.trace(inverse_noise_matrix), 2)
            raise ValueError(
                "Trace(inverse_noise_matrix) is {}. Must exceed 1.".format(t))

        # Number of classes
        self.K = len(np.unique(s))

        # 'ps' is p(s=k)
        self.ps = value_counts(s) / float(len(s))

        self.confident_joint = None
        # If needed, compute noise rates (mislabeling) for all classes. 
        # Also, if needed, compute P(s=k|x), denoted psx.

        # Set / re-set noise matrices / psx; estimate if not provided.
        if noise_matrix is not None:
            self.noise_matrix = noise_matrix
            if inverse_noise_matrix is None:
                self.py, self.inverse_noise_matrix = (
                    compute_py_inv_noise_matrix(self.ps, self.noise_matrix))
        if inverse_noise_matrix is not None:
            self.inverse_noise_matrix = inverse_noise_matrix
            if noise_matrix is None:
                self.noise_matrix = compute_noise_matrix_from_inverse(
                    self.ps,
                    self.inverse_noise_matrix,
                )
        if noise_matrix is None and inverse_noise_matrix is None:
            if psx is None:
                self.py, self.noise_matrix, self.inverse_noise_matrix, \
                self.confident_joint, psx = \
                    estimate_py_noise_matrices_and_cv_pred_proba(
                        X=X,
                        s=s,
                        clf=self.clf,
                        cv_n_folds=self.cv_n_folds,
                        thresholds=thresholds,
                        converge_latent_estimates=(
                            self.converge_latent_estimates),
                        seed=self.seed,
                    )
            else:  # psx is provided by user (assumed holdout probabilities)
                self.py, self.noise_matrix, self.inverse_noise_matrix, \
                self.confident_joint = \
                    estimate_py_and_noise_matrices_from_probabilities(
                        s=s,
                        psx=psx,
                        thresholds=thresholds,
                        converge_latent_estimates=(
                            self.converge_latent_estimates),
                    )

        if psx is None:
            psx = estimate_cv_predicted_probabilities(
                X=X,
                labels=s,
                clf=self.clf,
                cv_n_folds=self.cv_n_folds,
                seed=self.seed,
            )

        # if pulearning == the integer specifying the class without noise.
        if self.K == 2 and self.pulearning is not None:  # pragma: no cover
            # pulearning = 1 (no error in 1 class) implies p(s=1|y=0) = 0
            self.noise_matrix[self.pulearning][1 - self.pulearning] = 0
            self.noise_matrix[1 - self.pulearning][1 - self.pulearning] = 1
            # pulearning = 1 (no error in 1 class) implies p(y=0|s=1) = 0
            self.inverse_noise_matrix[1 - self.pulearning][self.pulearning] = 0
            self.inverse_noise_matrix[self.pulearning][self.pulearning] = 1
            # pulearning = 1 (no error in 1 class) implies p(s=1,y=0) = 0
            self.confident_joint[self.pulearning][1 - self.pulearning] = 0
            self.confident_joint[1 - self.pulearning][1 - self.pulearning] = 1

        # This is the actual work of this function.

        # Get the indices of the examples we wish to prune
        self.noise_mask = get_noise_indices(
            s,
            psx,
            inverse_noise_matrix=self.inverse_noise_matrix,
            confident_joint=self.confident_joint,
            prune_method=self.prune_method,
            n_jobs=self.n_jobs,
        )

        x_mask = ~self.noise_mask
        x_pruned = X[x_mask]
        s_pruned = s[x_mask]

        # Check if sample_weight in clf.fit(). Compatible with Python 2/3.
        if hasattr(inspect, 'getfullargspec') and \
                'sample_weight' in inspect.getfullargspec(self.clf.fit).args \
                or hasattr(inspect, 'getargspec') and \
                'sample_weight' in inspect.getargspec(self.clf.fit).args:
            # Re-weight examples in the loss function for the final fitting
            # s.t. the "apparent" original number of examples in each class
            # is preserved, even though the pruned sets may differ.
            self.sample_weight = np.ones(np.shape(s_pruned))
            for k in range(self.K):
                sample_weight_k = 1.0 / self.noise_matrix[k][k]
                self.sample_weight[s_pruned == k] = sample_weight_k

            self.clf.fit(x_pruned, s_pruned, sample_weight=self.sample_weight)
        else:
            # This is less accurate, but best we can do if no sample_weight.
            self.clf.fit(x_pruned, s_pruned)

        return self.clf

    def predict(self, *args, **kwargs):
        """Returns a binary vector of predictions.

        Typical Parameters
        ----------
        X : np.array of shape (n, m)
          The test data as a feature matrix."""

        return self.clf.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        """Returns a vector of probabilties P(y=k)
        for each example in X.

        Typical Parameters
        ----------
        X : np.array of shape (n, m)
          The test data as a feature matrix."""

        return self.clf.predict_proba(*args, **kwargs)

    def score(self, X, y, sample_weight=None):
        """Returns the clf's score on a test set X with labels y.
        Uses the models default scoring function.

        Parameters
        ----------
        X : np.array of shape (n, m)
          The test data as a feature matrix.

        y : np.array<int> of shape (n,) or (n, 1)
          The test classification labels as an array.

        sample_weight : np.array<float> of shape (n,) or (n, 1)
          Weights each example when computing the score / accuracy."""

        if hasattr(self.clf, 'score'):

            # Check if sample_weight in clf.score(). Compatible with Python 2/3.
            if hasattr(inspect, 'getfullargspec') and 'sample_weight' in \
                    inspect.getfullargspec(self.clf.score).args or \
                    hasattr(inspect, 'getargspec') and \
                    'sample_weight' in inspect.getargspec(self.clf.score).args:
                return self.clf.score(X, y, sample_weight=sample_weight)
            else:
                return self.clf.score(X, y)
        else:
            return accuracy_score(
                y, self.clf.predict(X), sample_weight=sample_weight, )
