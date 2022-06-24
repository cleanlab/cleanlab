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


# ## Latent Algebra
#
# #### Contains mathematical functions relating the latent terms,
# $p(labels), P_{labels \vert y}, P_{y \vert labels}, p(true_labels)$, etc. together.
# For every function here, if the inputs are exact, the output is guaranteed
# to be exact. Every function herein is the computational equivalent of a
# mathematical equation having a closed, exact form. If the inputs are
# inexact, the error will of course propagate.

import numpy as np

from cleanlab.internal.util import value_counts, clip_values, clip_noise_rates
import warnings


def compute_ps_py_inv_noise_matrix(labels, noise_matrix):
    """Compute ps := P(labels=k), py := P(true_labels=k), and the inverse noise matrix.

    Parameters
    ----------
    labels : np.array
          A discrete vector of noisy labels, i.e. some labels may be erroneous.
          *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    noise_matrix : np.array of shape (K, K), K = number of classes
        A conditional probability matrix of the form P(label=k_s|true_label=k_y) containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1."""

    # 'ps' is p(labels=k)
    ps = value_counts(labels) / float(len(labels))

    py, inverse_noise_matrix = compute_py_inv_noise_matrix(ps, noise_matrix)
    return ps, py, inverse_noise_matrix


def compute_py_inv_noise_matrix(ps, noise_matrix):
    """Compute py := P(true_label=k), and the inverse noise matrix.

    Parameters
    ----------
    ps : np.array (shape (K, ) or (1, K))
        The fraction (prior probability) of each observed, NOISY class P(labels = k).

    noise_matrix : np.array of shape (K, K), K = number of classes
        A conditional probability matrix of the form P(label=k_s|true_label=k_y) containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1."""

    # 'py' is p(true_labels=k) = noise_matrix^(-1) * p(labels=k)
    # because in *vector computation*: P(label=k|true_label=k) * p(true_label=k) = P(label=k)
    # The pseudo-inverse is used when noise_matrix is not invertible.
    py = np.linalg.inv(noise_matrix).dot(ps)

    # No class should have probability 0, so we use .000001
    # Make sure valid probabilities that sum to 1.0
    py = clip_values(py, low=1e-6, high=1.0, new_sum=1.0)

    # All the work is done in this function (below)
    return py, compute_inv_noise_matrix(py=py, noise_matrix=noise_matrix, ps=ps)


def compute_inv_noise_matrix(py, noise_matrix, *, ps=None):
    """Compute the inverse noise matrix if py := P(true_label=k) is given.

    Parameters
    ----------
    py : np.array (shape (K, 1))
        The fraction (prior probability) of each TRUE class label, P(true_label = k)

    noise_matrix : np.array of shape (K, K), K = number of classes
        A conditional probability matrix of the form P(label=k_s|true_label=k_y) containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1.

    ps : np.array (shape (K, 1))
        The fraction (prior probability) of each NOISY given label, P(labels = k).
        ps is easily computable from py and should only be provided if it has
        already been precomputed, to increase code efficiency.

    Examples
    --------
    For loop based implementation:

    .. code:: python

        # Number of classes
        K = len(py)

        # 'ps' is p(labels=k) = noise_matrix * p(true_labels=k)
        # because in *vector computation*: P(label=k|true_label=k) * p(true_label=k) = P(label=k)
        if ps is None:
            ps = noise_matrix.dot(py)

        # Estimate the (K, K) inverse noise matrix P(true_label = k_y | label = k_s)
        inverse_noise_matrix = np.empty(shape=(K,K))
        # k_s is the class value k of noisy label `label == k`
        for k_s in range(K):
            # k_y is the (guessed) class value k of true label y
            for k_y in range(K):
                # P(true_label|label) = P(label|y) * P(true_label) / P(labels)
                inverse_noise_matrix[k_y][k_s] = noise_matrix[k_s][k_y] * \
                                                 py[k_y] / ps[k_s]
    """

    joint = noise_matrix * py
    ps = joint.sum(axis=1) if ps is None else ps
    inverse_noise_matrix = joint.T / ps

    # Clip inverse noise rates P(true_label=k_s|true_label=k_y) into proper range [0,1)
    return clip_noise_rates(inverse_noise_matrix)


def compute_noise_matrix_from_inverse(ps, inverse_noise_matrix, *, py=None):
    """Compute the noise matrix P(label=k_s|true_label=k_y).

    Parameters
    ----------
    py : np.array (shape (K, 1))
        The fraction (prior probability) of each TRUE class label, P(true_label = k)

    inverse_noise_matrix : np.array of shape (K, K), K = number of classes
        A conditional probability matrix of the form P(true_label=k_y|label=k_s) representing
        the estimated fraction observed examples in each class k_s, that are
        mislabeled examples from every other class k_y. If None, the
        inverse_noise_matrix will be computed from pred_probs and labels.
        Assumes columns of inverse_noise_matrix sum to 1.

    ps : np.array (shape (K, 1))
        The fraction (prior probability) of each observed NOISY label, P(labels = k).
        ps is easily computable from py and should only be provided if it has
        already been precomputed, to increase code efficiency.

    Returns
    -------
    noise_matrix : np.array of shape (K, K), K = number of classes
        A conditional probability matrix of the form P(label=k_s|true_label=k_y) containing
        the fraction of examples in every class, labeled as every other class.
        Columns of noise_matrix sum to 1.

    Examples
    --------
    For loop based implementation:

    .. code:: python

        # Number of classes labels
        K = len(ps)

        # 'py' is p(true_label=k) = inverse_noise_matrix * p(true_label=k)
        # because in *vector computation*: P(true_label=k|label=k) * p(label=k) = P(true_label=k)
        if py is None:
            py = inverse_noise_matrix.dot(ps)

        # Estimate the (K, K) noise matrix P(labels = k_s | true_labels = k_y)
        noise_matrix = np.empty(shape=(K,K))
        # k_s is the class value k of noisy label `labels == k`
        for k_s in range(K):
            # k_y is the (guessed) class value k of true label y
            for k_y in range(K):
                # P(labels|y) = P(true_label|labels) * P(labels) / P(true_label)
                noise_matrix[k_s][k_y] = inverse_noise_matrix[k_y][k_s] * \
                                         ps[k_s] / py[k_y]

    """

    joint = (inverse_noise_matrix * ps).T
    py = joint.sum(axis=0) if py is None else py
    noise_matrix = joint / py

    # Clip inverse noise rates P(true_label=k_y|true_label=k_s) into proper range [0,1)
    return clip_noise_rates(noise_matrix)


def compute_py(
    ps, noise_matrix, inverse_noise_matrix, *, py_method="cnt", true_labels_class_counts=None
):
    """Compute py := P(true_labels=k) from ps := P(labels=k), noise_matrix, and the
    inverse noise matrix.

    This method is ** ROBUST ** when py_method = 'cnt'
    It may work well even when the noise matrices are estimated
    poorly by using the diagonals of the matrices
    instead of all the probabilities in the entire matrix.

    Parameters
    ----------
    ps : np.array (shape (K, ) or (1, K))
        The fraction (prior probability) of each observed, noisy label, P(labels = k)

    noise_matrix : np.array of shape (K, K), K = number of classes
        A conditional probability matrix of the form P(label=k_s|true_label=k_y) containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1.

    inverse_noise_matrix : np.array of shape (K, K), K = number of classes
        A conditional probability matrix of the form P(true_label=k_y|label=k_s) representing
        the estimated fraction observed examples in each class k_s, that are
        mislabeled examples from every other class k_y. If None, the
        inverse_noise_matrix will be computed from pred_probs and labels.
        Assumes columns of inverse_noise_matrix sum to 1.

    py_method : str (Options: ["cnt", "eqn", "marginal", "marginal_ps"])
        How to compute the latent prior p(true_label=k). Default is "cnt" as it often
        works well even when the noise matrices are estimated poorly by using
        the matrix diagonals instead of all the probabilities.

    true_labels_class_counts : np.array (shape (K, ) or (1, K))
        The marginal counts of the confident joint (like cj.sum(axis = 0))

    Returns
    -------
    py : np.array (shape (K, ) or (1, K))
        The fraction (prior probability) of each TRUE class label, P(true_label = k)."""

    if len(np.shape(ps)) > 2 or (len(np.shape(ps)) == 2 and np.shape(ps)[0] != 1):
        w = "Input parameter np.array ps has shape " + str(np.shape(ps))
        w += ", but shape should be (K, ) or (1, K)"
        warnings.warn(w)

    if py_method == "marginal" and true_labels_class_counts is None:
        msg = (
            'py_method == "marginal" requires true_labels_class_counts, '
            "but true_labels_class_counts is None. "
        )
        msg += " Provide parameter true_labels_class_counts."
        raise ValueError(msg)

    if py_method == "cnt":
        # Computing py this way avoids dividing by zero noise rates.
        # More robust bc error est_p(true_label|labels) / est_p(labels|y) ~ p(true_label|labels) / p(labels|y)
        py = inverse_noise_matrix.diagonal() / noise_matrix.diagonal() * ps
        # Equivalently,
        # py = (true_labels_class_counts / labels_class_counts) * ps
    elif py_method == "eqn":
        py = np.linalg.inv(noise_matrix).dot(ps)
    elif py_method == "marginal":
        py = true_labels_class_counts / float(sum(true_labels_class_counts))
    elif py_method == "marginal_ps":
        py = np.dot(inverse_noise_matrix, ps)
    else:
        err = "py_method {}".format(py_method)
        err += " should be in [cnt, eqn, marginal, marginal_ps]"
        raise ValueError(err)

    # Clip py (0,1), s.t. no class should have prob 0, hence 1e-5
    py = clip_values(py, low=1e-5, high=1.0, new_sum=1.0)
    return py


def compute_pyx(pred_probs, noise_matrix, inverse_noise_matrix):
    """Compute pyx := P(true_label=k|x) from pred_probs := P(label=k|x), and the noise_matrix and
    inverse noise matrix.

    This method is ROBUST - meaning it works well even when the
    noise matrices are estimated poorly by only using the diagonals of the
    matrices which tend to be easy to estimate correctly.

    Parameters
    ----------
    pred_probs : np.array (shape (N, K))
        P(label=k|x) is a matrix with K model-predicted probabilities.
        Each row of this matrix corresponds to an example `x` and contains the model-predicted
        probabilities that `x` belongs to each possible class.
        The columns must be ordered such that these probabilities correspond to class 0,1,2,...
        `pred_probs` should have been computed using 3 (or higher) fold cross-validation.

    noise_matrix : np.array of shape (K, K), K = number of classes
        A conditional probability matrix of the form P(label=k_s|true_label=k_y) containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1.

    inverse_noise_matrix : np.array of shape (K, K), K = number of classes
        A conditional probability matrix of the form P(true_label=k_y|label=k_s) representing
        the estimated fraction observed examples in each class k_s, that are
        mislabeled examples from every other class k_y. If None, the
        inverse_noise_matrix will be computed from pred_probs and labels.
        Assumes columns of inverse_noise_matrix sum to 1.

    Returns
    -------
    pyx : np.array (shape (N, K))
        P(true_label=k|x) is a matrix with K model-predicted probabilities.
        Each row of this matrix corresponds to an example `x` and contains the model-predicted
        probabilities that `x` belongs to each possible class.
        The columns must be ordered such that these probabilities correspond to class 0,1,2,...
        `pred_probs` should have been computed using 3 (or higher) fold cross-validation."""

    if len(np.shape(pred_probs)) != 2:
        raise ValueError(
            "Input parameter np.array 'pred_probs' has shape "
            + str(np.shape(pred_probs))
            + ", but shape should be (N, K)"
        )

    pyx = pred_probs * inverse_noise_matrix.diagonal() / noise_matrix.diagonal()
    # Make sure valid probabilities that sum to 1.0
    return np.apply_along_axis(
        func1d=clip_values, axis=1, arr=pyx, **{"low": 0.0, "high": 1.0, "new_sum": 1.0}
    )
