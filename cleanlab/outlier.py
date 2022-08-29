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


import warnings
import numpy as np
from cleanlab.count import get_confident_thresholds
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Union, Tuple
from cleanlab.internal.label_quality_utils import (
    _subtract_confident_thresholds,
    get_normalized_entropy,
)
from cleanlab.internal.validation import assert_valid_inputs


class OutOfDistribution:
    """
     OutOfDistribution = Out of distribution detection for classification examples using intermediate feature
     embeddings or predicted probabilities. Each passed in example is given an out-of-distribution score. Scores lie in [0,1] with smaller
     values indicating examples that are less typical under the dataset distribution (values near 0 indicate outliers).

     Parameters
     ----------
     knn : sklearn.neighbors.NearestNeighbors, default = None
       Instantiated ``NearestNeighbors`` object that's been fitted on a dataset in the same feature space.
       Note that the distance metric and n_neighbors is specified when instantiating this class.
       You can also pass in a subclass of ``sklearn.neighbors.NearestNeighbors`` which allows you to use faster
       approximate neighbor libraries as long as you wrap them behind the same sklearn API.
       If you specify ``knn`` here and wish to find outliers in the same data you already passed into ``knn.fit(features)``,
       you should specify ``features = None`` here if your ``knn.kneighbors(None)``
       returns the distances to the datapoints it was ``fit()`` on.
       If ``knn = None``, then by default ``knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k, metric="cosine").fit(features)``

       See: https://scikit-learn.org/stable/modules/neighbors.html

    confident_thresholds : np.ndarray, default = None
       An array of shape ``(K, )`` where K is the number of classes.
       Confident threshold for a class j is the expected (average) "self-confidence" for that class.

    params : dict, default = None
       A dictionary of optional parameters for calculating ood scores.

    """

    OUTLIER_PARAMS = {"k", "t"}
    OOD_PARAMS = {"labels", "adjust_pred_probs", "method"}
    DEFAULT_PARAM_DICT = {
        "k": None,  # outlier param
        "t": 1,  # outlier param
        "adjust_pred_probs": True,  # ood param
        "method": "entropy",  # ood param
    }

    def __init__(self):
        self.params = {}
        self.knn = None
        self.confident_thresholds = None

    def fit_score(
        self,
        *,
        features: Optional[np.ndarray] = None,
        pred_probs: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        params: dict = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Fits an estimator and returns out-of-distribution scores following optional parameters specified in `params`.
        Scores lie in [0,1] with smaller values indicating examples that are less typical under the dataset
        distribution (values near 0 indicate outliers). Exactly one of `features` or `pred_probs` needs to be passed
        in to calculate scores.

        If `features` are passed in a `NearestNeighbors` object is fit. If `pred_probs` and 'labels' are passed in a
        `confident_thresholds` np.ndarray is fit. For details see :py:func:`fit
        <cleanlab.outlier.OutOfDistribution.fit>` function.

        Parameters
        ----------
        features : np.ndarray, optional
          Feature array of shape ``(N, M)``, where N is the number of examples and M is the number of features used to represent each example.
          For details, `features` in the same format expected by the :py:func:`fit <cleanlab.outlier.OutOfDistribution.fit>` function.

        pred_probs : np.ndarray, optional
          An array of shape ``(N, K)`` of model-predicted probabilities.
          For details, `pred_probs` in the same format expected by the :py:func:`fit <cleanlab.outlier.OutOfDistribution.fit>` function.

        labels : np.ndarray, optional
          A discrete vector of noisy labels, i.e. some labels may be erroneous of shape ``(N,)``.
          For details, `labels` in the same format expected by the :py:func:`fit <cleanlab.outlier.OutOfDistribution.fit>` function.

        params : bool, default = False
          Optional keyword arguments to that change how estimator is fit.
          For details, `params` in the same format expected by the :py:func:`fit <cleanlab.outlier.OutOfDistribution.fit>` function.

        verbose : bool, default = True
          Set to ``False`` to suppress all print statements.

        Returns
        -------
        scores : np.ndarray
          If `features` are passed in, `ood_features_scores` are returned. For details see return of :py:func:`score <cleanlab.outlier.OutOfDistribution.score>` function.
          If `pred_probs` are passed in, `ood_predictions_scores` are returned. For details see return of :py:func:`score <cleanlab.outlier.OutOfDistribution.scores>` function.

        """
        scores = self._shared_fit(
            features=features, pred_probs=pred_probs, labels=labels, params=params, verbose=verbose
        )

        return scores

    def fit(
        self,
        *,
        features: Optional[np.ndarray] = None,
        pred_probs: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        params: dict = None,
        verbose: bool = True,
    ):
        """
        Fits an estimator following optional parameters specified in `params`. One of `features` or `pred_probs`
        needs to be passed in to fit estimator object.

        If `features` are passed in a `NearestNeighbors` object is fit. For details on the object see
        :py:class:`OutOfDistribution <cleanlab.outlier.OutOfDistribution>`.

        If `pred_probs` and 'labels' are passed in a `confident_thresholds` np.ndarray is fit. For details on
        'confindent_thresholds` see :py:class:`OutOfDistribution <cleanlab.outlier.OutOfDistribution>`.

        Parameters
        ----------
        features : np.ndarray, optional
          Feature array of shape ``(N, M)``, where N is the number of examples and M is the number of features used to represent each example.
          All features should be numeric. For unstructured data (eg. images, text, categorical values, ...), you should provide
          vector embeddings to represent each example (e.g. extracted from some pretrained neural network).

        pred_probs : np.ndarray, optional
           An array of shape ``(N, K)`` of model-predicted probabilities,
          ``P(label=k|x)``. Each row of this matrix corresponds
          to an example `x` and contains the model-predicted probabilities that
          `x` belongs to each possible class, for each of the K classes. The
          columns must be ordered such that these probabilities correspond to
          class 0, 1, ..., K-1.

          **Caution**: `pred_probs` from your model must be out-of-sample!
          You should never provide predictions on the same examples used to train the model,
          as these will be overfit and unsuitable for finding label-errors.
          To obtain out-of-sample predicted probabilities for every datapoint in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`.
          Alternatively it is ok if your model was trained on a separate dataset and you are only evaluating
          data that was previously held-out.

        labels : np.ndarray, optional
          A discrete vector of noisy labels, i.e. some labels may be erroneous of shape ``(N,)``.
          *Format requirements*: for dataset with K classes, labels must be in 0, 1, ..., K-1.
          All the classes (0, 1, ..., and K-1) MUST be present in ``labels``, such that: ``len(set(labels)) == pred_probs.shape[1]``
          Note: multi-label classification is not supported by this method, each example must belong to a single class, e.g. format: ``labels = np.ndarray([1,0,2,1,1,0...])``.

        params : dict, default = None
          Optional keyword arguments to that change how estimator is fit. Types of arguments passed depends on if `OutOfDistribution` object is fit on `features` or `pred_probs`.

          If `features` is passed in, `params` could contain following keys:
            *  k : int, default=None
                  Optional number of neighbors to use when calculating outlier score (average distance to neighbors).
                  If `k` is not provided, then by default ``k = knn.n_neighbors`` or ``k = 10`` if ``knn is None``.
                  If an existing ``knn`` object is provided, you can still specify that outlier scores should use
                  a different value of `k` than originally used in the ``knn``,
                  as long as your specified value of `k` is smaller than the value originally used in ``knn``.
            *  t : int, default=1
                  Optional hyperparameter only for advanced users.
                  Controls transformation of distances between examples into similarity scores that lie in [0,1].
                  The transformation applied to distances `x` is `exp(-x*t)`.
                  If you find your scores are all too close to 1, consider increasing `t`,
                  although the relative scores of examples will still have the same ranking across the dataset.

          If `pred_probs` is passed in, `params` could contian following keys:
            *  adjust_pred_probs : bool, True
                  Account for class imbalance in the label-quality scoring by adjusting predicted probabilities
                  via subtraction of class confident thresholds and renormalization.
                  Set this to ``False`` if you prefer to skip accounting for class-imbalance.
                  See `Northcutt et al., 2021 <https://jair.org/index.php/jair/article/view/12125>`_.
            *  method : {"entropy", "least_confidence"}, default="entropy"
                  OOD scoring method.

                  Letting ``P = pred_probs[i]`` denote the given predicted class-probabilities
                  for datapoint *i*, its score can either be:

                  - ``'entropy'``: ``- sum_{j} P[j] * log(P[j])``
                  - ``'least_confidence'``: ``1 - max(P)``

        verbose : bool, default = True
          Set to ``False`` to suppress all print statements.

        """
        _ = self._shared_fit(
            features=features, pred_probs=pred_probs, labels=labels, params=params, verbose=verbose
        )

    def score(
        self, *, features: Optional[np.ndarray] = None, pred_probs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Uses fitted estimator and passed in `features` or `pred_probs` to calculate out-of-distribution scores with
        optional params passed in during ''fit()''. Score for each example that roughly corresponds to the likelihood this example stems from the same distribution as
        the dataset features (i.e. is not an outlier). Scores lie in [0,1] with smaller values indicating examples that
        are less typical under the dataset distribution (values near 0 indicate outliers).

        If `features` are passed, outlier score for each example based on its feature values is returned.
        If `pred_probs` are passed in out of distribution score for each example based on its `pred_prob` values is returned.

        Parameters
        ----------
        features : np.ndarray, optional
          Feature array of shape ``(N, M)``, where N is the number of examples and M is the number of features used to represent each example.
          For details, see `features` in :py:func:`fit <cleanlab.outlier.OutOfDistribution.fit>` function.

        pred_probs : np.ndarray, optional
          An array of shape ``(N, K)`` of model-predicted probabilities.
          For details, see `pred_probs` in :py:func:`fit <cleanlab.outlier.OutOfDistribution.fit>` function.

        Returns
        -------
        scores : np.ndarray
          If `features` are passed in, `ood_features_scores` are returned.
          The score is based on the average distance between the example and its K nearest neighbors in the dataset
          (in feature space).

          If `pred_probs` are passed in, `ood_predictions_scores` are returned.
          The score is based on the model predicted probabilities.
        """
        self._assert_valid_inputs(features, pred_probs)

        if features is not None:
            if self.knn is None:
                raise ValueError(
                    f"OOD Object needs to be fit on features first. Call fit() or fit_scores() before this function."
                )
            else:
                params = self._get_params(self.OUTLIER_PARAMS)  # get params specific to outliers
                scores = _get_ood_features_scores(
                    features, self.knn, **params, return_estimator=False
                )

        if pred_probs is not None:
            if self.confident_thresholds is None and self.params["adjust_pred_probs"]:
                raise ValueError(
                    f"OOD Object needs to be fit on features first. Call fit() or fit_scores() before this function."
                )
            else:
                params = self._get_params(self.OOD_PARAMS)  # get params specific to outliers
                scores = _get_ood_predictions_scores(
                    pred_probs,
                    confident_thresholds=self.confident_thresholds,
                    **params,
                    return_thresholds=False,
                )

        # TODO: How to fix this typing issue without assert? _get_[...]_scores should always return np.ndarray
        #  since return_estimator and return_thresholds are False.
        assert isinstance(scores, np.ndarray)
        return scores

    def _get_params(self, param_keys) -> dict:
        """
        Helper method to get function specific dictionary of parameters (i.e. only those in param_keys).
        """
        return {k: v for k, v in self.params.items() if k in param_keys}

    def _get_invalid_params(self, params, param_keys) -> list:
        """
        Helper method to get list of parameters in param that are not in param_keys.
        """
        if params is None:
            return []
        return list(set(params.keys()).difference(set(param_keys)))

    def _assert_valid_inputs(self, features, pred_probs):
        """
        Helper method to check features and pred_prob inputs are valid. Throws error if not.
        """
        if features is None and pred_probs is None:
            raise ValueError(
                f"Not enough information to compute scores. Pass in either features or pred_probs."
            )

        if features is not None and pred_probs is not None:
            raise ValueError(
                f"Cannot fit to object to both features and pred_probs. Pass in either one or the other."
            )

        if features is not None and len(features.shape) != 2:
            raise ValueError(
                f"Feature array needs to be of shape (N, M), where N is the number of examples and M is the "
                f"number of features used to represent each example. "
            )

    def _shared_fit(
        self,
        *,
        features: Optional[np.ndarray] = None,
        pred_probs: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        params: dict = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Shared fit functionality between `fit()` and `fit_score()`.
        For details, refer to :py:func:`fit <cleanlab.outlier.OutOfDistribution.fit>`
        or :py:func:`fit_score <cleanlab.outlier.OutOfDistribution.fit_score>`.
        """
        # Assert valid inputs and params
        self._assert_valid_inputs(features, pred_probs)

        # Update params
        self.params = self.DEFAULT_PARAM_DICT
        if params is not None:
            self.params = {**self.params, **params}

        if features is not None:
            # raise Error if they passed in invalid params keys for using features.
            ood_params = self._get_invalid_params(params, self.OUTLIER_PARAMS)
            if len(ood_params) > 0:
                raise ValueError(
                    f"If fit with features, passed in params dict can only contain {self.OUTLIER_PARAMS}. Remove {ood_params} from params dict."
                )
            # get outlier scores
            if verbose:
                print("Fitting OOD object based on provided features ...")
            scores, knn = _get_ood_features_scores(
                features, **self._get_params(self.OUTLIER_PARAMS), return_estimator=True
            )
            self.knn = knn  # save estimator

        if pred_probs is not None:
            # raise Error if they passed in invalid params keys for using pred_probs.
            outlier_params = self._get_invalid_params(params, self.OOD_PARAMS)
            if len(outlier_params) > 0:
                raise ValueError(
                    f"If fit with pred_probs, passed in params dict can only contain {self.OOD_PARAMS}. Remove {outlier_params} from params dict."
                )

            # get ood scores
            if verbose:
                print("Fitting OOD object based on provided pred_probs ...")
            scores, confident_thresholds = _get_ood_predictions_scores(
                pred_probs,
                labels=labels,
                **self._get_params(self.OOD_PARAMS),
                return_thresholds=True,
            )
            if confident_thresholds is None:
                warnings.warn(
                    f"Object not fit with confident_thresholds since adjust_pred_probs=False and no "
                    f"confident_thresholds were calculated.",
                    UserWarning,
                )
            self.confident_thresholds = confident_thresholds
        return scores


def _get_ood_features_scores(
    features: Optional[np.ndarray] = None,
    knn: Optional[NearestNeighbors] = None,
    k: Optional[int] = None,
    t: int = 1,
    return_estimator: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, NearestNeighbors]]:
    """Returns an outlier score for each example based on its feature values.

    Parameters
    ----------
    features : np.ndarray
      Feature array of shape ``(N, M)``, where N is the number of examples and M is the number of features used to represent each example.
      For details, `features` in the same format expected by the :py:func:`fit <cleanlab.outlier.OutOfDistribution.fit>` function.

    knn : sklearn.neighbors.NearestNeighbors, default = None
       For details on 'knn` see :py:class:`OutOfDistribution <cleanlab.outlier.OutOfDistribution>`.

    k : int, default=None
      Optional number of neighbors to use when calculating outlier score (average distance to neighbors).
      For details, `k` in the same format expected by the :py:func:`fit <cleanlab.outlier.OutOfDistribution.fit>` function.

    t : int, default=1
      Controls transformation of distances between examples into similarity scores that lie in [0,1].
      For details, `t` in the same format expected by the :py:func:`fit <cleanlab.outlier.OutOfDistribution.fit>` function.

    return_estimator : bool, default = False
      Whether the `knn` Estimator object should also be returned (eg. so it can be applied on future data).
      If True, this function returns a tuple `(ood_features_scores, knn)`.

    Returns
    -------
    ood_features_scores : np.ndarray
      If ``return_estimator = True``, then a tuple is returned
      whose first element is array of `ood_features_scores` and second is a `knn` Estimator object.
    """
    DEFAULT_K = 10
    if knn is None:  # setup default KNN estimator
        # Make sure both knn and features are not None
        if features is None:
            raise ValueError(
                f"Both knn and features arguments cannot be None at the same time. Not enough information to compute outlier scores."
            )
        if k is None:
            k = DEFAULT_K  # use default when knn and k are both None
        if k > len(features):  # Ensure number of neighbors less than number of examples
            raise ValueError(
                f"Number of nearest neighbors k={k} cannot exceed the number of examples N={len(features)} passed into the estimator (knn)."
            )
        knn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(features)
        features = None  # features should be None in knn.kneighbors(features) to avoid counting duplicate data points
    elif k is None:
        k = knn.n_neighbors

    max_k = knn.n_neighbors  # number of neighbors previously used in NearestNeighbors object
    if k > max_k:  # if k provided is too high, use max possible number of nearest neighbors
        warnings.warn(
            f"Chosen k={k} cannot be greater than n_neighbors={max_k} which was used when fitting "
            f"NearestNeighbors object! Value of k changed to k={max_k}.",
            UserWarning,
        )
        k = max_k

    # Get distances to k-nearest neighbors Note that the knn object contains the specification of distance metric
    # and n_neighbors (k value) If our query set of features matches the training set used to fit knn, the nearest
    # neighbor of each point is the point itself, at a distance of zero.
    distances, _ = knn.kneighbors(features)

    # Calculate average distance to k-nearest neighbors
    avg_knn_distances = distances[:, :k].mean(axis=1)

    # Map ood_features_scores to range 0-1 with 0 = most concerning
    ood_features_scores: np.ndarray = np.exp(-1 * avg_knn_distances * t)
    if return_estimator:
        return (ood_features_scores, knn)
    else:
        return ood_features_scores


def _get_ood_predictions_scores(
    pred_probs: np.ndarray,
    *,
    labels: Optional[np.ndarray] = None,
    confident_thresholds: Optional[np.ndarray] = None,
    adjust_pred_probs: bool = True,
    method: str = "entropy",
    return_thresholds: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Returns an OOD (out of distribution) score for each example based on it pred_prob values.

    Parameters
    ----------
    pred_probs : np.ndarray
      An array of shape ``(N, K)`` of model-predicted probabilities,
      `pred_probs` in the same format expected by the :py:func:`fit <cleanlab.outlier.OutOfDistribution.fit>` function.

    confident_thresholds : np.ndarray, default = None
      For details on 'confindent_thresholds` see :py:class:`OutOfDistribution <cleanlab.outlier.OutOfDistribution>`.

    labels : np.ndarray, optional
      `labels` in the same format expected by the :py:func:`fit <cleanlab.outlier.OutOfDistribution.fit>` function.

    adjust_pred_probs : bool, True
      Account for class imbalance in the label-quality scoring.
      For details, see `adjust_pred_probs` in :py:func:`fit <cleanlab.outlier.OutOfDistribution.fit>` function.

    method : {"entropy", "least_confidence"}, default="entropy"
      OOD scoring method.
      For details see `method` in :py:func:`fit <cleanlab.outlier.OutOfDistribution.fit>` function.


    Returns
    -------
    ood_predictions_scores : np.ndarray
      If ``return_thresholds = True``, then a tuple is returned
      whose first element is array of `ood_predictions_scores` and second is an np.ndarray of `confident_thresholds` or None is 'confident_thresholds' is not calculated.
    """

    valid_methods = [
        "entropy",
        "least_confidence",
    ]

    if (confident_thresholds is not None or labels is not None) and not adjust_pred_probs:
        warnings.warn(
            f"OOD scores are not adjusted with confident thresholds. If scores need to be adjusted set "
            f"adjusted_pred_probs = True. Otherwise passing in confident_thresholds and/or labels does not change "
            f"score calculation.",
            UserWarning,
        )

    if adjust_pred_probs:
        if confident_thresholds is None:
            if labels is None:
                raise ValueError(
                    f"Cannot calculate adjust_pred_probs without labels. Either pass in labels parameter or set "
                    f"adjusted_pred_probs = False. "
                )
            else:
                assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs, multi_label=False)
                confident_thresholds = get_confident_thresholds(
                    labels, pred_probs, multi_label=False
                )

        pred_probs = _subtract_confident_thresholds(
            None, pred_probs, multi_label=False, confident_thresholds=confident_thresholds
        )

    if method == "entropy":
        ood_predictions_scores = get_normalized_entropy(pred_probs)
    elif method == "least_confidence":
        ood_predictions_scores = 1.0 - pred_probs.max(axis=1)
    else:
        raise ValueError(
            f"""
            {method} is not a valid OOD scoring method!
            Please choose a valid scoring_method: {valid_methods}
            """
        )

    if return_thresholds:
        return (
            ood_predictions_scores,
            confident_thresholds,
        )
    else:
        return ood_predictions_scores
