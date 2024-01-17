# Copyright (C) 2017-2023  Cleanlab Inc.
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
Methods for finding out-of-distribution examples in a dataset via scores that quantify how atypical each example is compared to the others.

The underlying algorithms are described in `this paper <https://arxiv.org/abs/2207.03061>`_.
"""

import warnings
import numpy as np
from cleanlab.count import get_confident_thresholds
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import NotFittedError
from typing import Optional, Union, Tuple, Dict
from cleanlab.internal.label_quality_utils import (
    _subtract_confident_thresholds,
    get_normalized_entropy,
)
from cleanlab.internal.numerics import softmax
from cleanlab.internal.outlier import transform_distances_to_scores
from cleanlab.internal.validation import assert_valid_inputs, labels_to_array
from cleanlab.typing import LabelLike


class OutOfDistribution:
    """
    Provides scores to detect Out Of Distribution (OOD) examples that are outliers in a dataset.

    Each example's OOD score lies in [0,1] with smaller values indicating examples that are less typical under the data distribution.
    OOD scores may be estimated from either: numeric feature embeddings or predicted probabilities from a trained classifier.

    To get indices of examples that are the most severe outliers, call `~cleanlab.rank.find_top_issues` function on the returned OOD scores.

    Parameters
    ----------
    params : dict, default = {}
     Optional keyword arguments to control how this estimator is fit. Effect of arguments passed in depends on if
     `OutOfDistribution` estimator will rely on `features` or `pred_probs`. These are stored as an instance attribute `self.params`.

     If `features` is passed in during ``fit()``, `params` could contain following keys:
       *  knn: sklearn.neighbors.NearestNeighbors, default = None
             Instantiated ``NearestNeighbors`` object that's been fitted on a dataset in the same feature space.
             Note that the distance metric and `n_neighbors` is specified when instantiating this class.
             You can also pass in a subclass of ``sklearn.neighbors.NearestNeighbors`` which allows you to use faster
             approximate neighbor libraries as long as you wrap them behind the same sklearn API.
             If you specify ``knn`` here, there is no need to later call ``fit()`` before calling ``score()``.
             If ``knn = None``, then by default: ``knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k, metric=dist_metric).fit(features)``
             where ``dist_metric == "cosine"`` if ``dim(features) > 3`` or ``dist_metric == "euclidean"`` otherwise.
             See: https://scikit-learn.org/stable/modules/neighbors.html
       *  k : int, default=None
             Optional number of neighbors to use when calculating outlier score (average distance to neighbors).
             If `k` is not provided, then by default ``k = knn.n_neighbors`` or ``k = 10`` if ``knn is None``.
             If an existing ``knn`` object is provided, you can still specify that outlier scores should use
             a different value of `k` than originally used in the ``knn``,
             as long as your specified value of `k` is smaller than the value originally used in ``knn``.
       *  t : int, default=1
             Optional hyperparameter only for advanced users.
             Controls transformation of distances between examples into similarity scores that lie in [0,1].
             The transformation applied to distances `x` is ``exp(-x*t)``.
             If you find your scores are all too close to 1, consider increasing `t`,
             although the relative scores of examples will still have the same ranking across the dataset.

     If `pred_probs` is passed in during ``fit()``, `params` could contain following keys:
       *  confident_thresholds: np.ndarray, default = None
             An array of shape ``(K, )`` where K is the number of classes.
             Confident threshold for a class j is the expected (average) "self-confidence" for that class.
             If you specify `confident_thresholds` here, there is no need to later call ``fit()`` before calling ``score()``.
       *  adjust_pred_probs : bool, True
             If True, account for class imbalance by adjusting predicted probabilities
             via subtraction of class confident thresholds and renormalization.
             If False, you do not have to pass in `labels` later to fit this OOD estimator.
             See `Northcutt et al., 2021 <https://jair.org/index.php/jair/article/view/12125>`_.
       *  method : {"entropy", "least_confidence"}, default="entropy"
             Method to use when computing outlier scores based on `pred_probs`.
             Letting length-K vector ``P = pred_probs[i]`` denote the given predicted class-probabilities
             for the i-th example in dataset, its outlier score can either be:

             - ``'entropy'``: ``1 - sum_{j} P[j] * log(P[j]) / log(K)``
             - ``'least_confidence'``: ``max(P)`` (equivalent to Maximum Softmax Probability method from the OOD detection literature)
             - ``gen``: Generalized ENtropy score from the paper of Liu, Lochman, and Zach (https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.pdf)

    """

    OUTLIER_PARAMS = {"k", "t", "knn"}
    OOD_PARAMS = {"confident_thresholds", "adjust_pred_probs", "method", "M", "gamma"}
    DEFAULT_PARAM_DICT: Dict[str, Union[str, int, float, None, np.ndarray]] = {
        "k": None,  # param for feature based outlier detection (number of neighbors)
        "t": 1,  # param for feature based outlier detection (controls transformation of outlier scores to 0-1 range)
        "knn": None,  # param for features based outlier detection (precomputed nearest neighbors graph to use)
        "method": "entropy",  # param specifying which pred_probs-based outlier detection method to use
        "adjust_pred_probs": True,  # param for pred_probs based outlier detection (whether to adjust the probabilities by class thresholds or not)
        "confident_thresholds": None,  # param for pred_probs based outlier detection (precomputed confident thresholds to use for adjustment)
        "M": 100,  # param for GEN method for pred_probs based outlier detection
        "gamma": 0.1,  # param for GEN method for pred_probs based outlier detection
    }

    def __init__(self, params: Optional[dict] = None) -> None:
        self._assert_valid_params(params, self.DEFAULT_PARAM_DICT)
        self.params = self.DEFAULT_PARAM_DICT.copy()
        if params is not None:
            self.params.update(params)
        if self.params["adjust_pred_probs"] and self.params["method"] == "gen":
            print(
                "CAUTION: GEN method is not recommended for use with adjusted pred_probs. "
                "To use GEN, we recommend setting: params['adjust_pred_probs'] = False"
            )

        # scaling_factor internally used to rescale distances based on mean distances to k nearest neighbors
        self.params["scaling_factor"] = None

    def fit_score(
        self,
        *,
        features: Optional[np.ndarray] = None,
        pred_probs: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Fits this estimator to a given dataset and returns out-of-distribution scores for the same dataset.

        Scores lie in [0,1] with smaller values indicating examples that are less typical under the dataset
        distribution (values near 0 indicate outliers). Exactly one of `features` or `pred_probs` needs to be passed
        in to calculate scores.

        If `features` are passed in a ``NearestNeighbors`` object is fit. If `pred_probs` and 'labels' are passed in a
        `confident_thresholds` ``np.ndarray`` is fit. For details see `~cleanlab.outlier.OutOfDistribution.fit`.

        Parameters
        ----------
        features : np.ndarray, optional
          Feature array of shape ``(N, M)``, where N is the number of examples and M is the number of features used to represent each example.
          For details, `features` in the same format expected by the `~cleanlab.outlier.OutOfDistribution.fit` function.

        pred_probs : np.ndarray, optional
          An array of shape ``(N, K)`` of predicted class probabilities output by a trained classifier.
          For details, `pred_probs` in the same format expected by the `~cleanlab.outlier.OutOfDistribution.fit` function.

        labels : array_like, optional
          A discrete array of given class labels for the data of shape ``(N,)``.
          For details, `labels` in the same format expected by the `~cleanlab.outlier.OutOfDistribution.fit` function.

        verbose : bool, default = True
          Set to ``False`` to suppress all print statements.

        Returns
        -------
        scores : np.ndarray
          If `features` are passed in, `ood_features_scores` are returned.
          If `pred_probs` are passed in, `ood_predictions_scores` are returned.
          For details see return of `~cleanlab.outlier.OutOfDistribution.scores` function.

        """
        scores = self._shared_fit(
            features=features,
            pred_probs=pred_probs,
            labels=labels,
            verbose=verbose,
        )

        if scores is None:  # Fit was called on already fitted object so we just score vals instead
            scores = self.score(features=features, pred_probs=pred_probs)

        return scores

    def fit(
        self,
        *,
        features: Optional[np.ndarray] = None,
        pred_probs: Optional[np.ndarray] = None,
        labels: Optional[LabelLike] = None,
        verbose: bool = True,
    ):
        """
        Fits this estimator to a given dataset.

        One of `features` or `pred_probs` must be specified.

        If `features` are passed in, a ``NearestNeighbors`` object is fit.
        If `pred_probs` and 'labels' are passed in, a `confident_thresholds` ``np.ndarray`` is fit.
        For details see `~cleanlab.outlier.OutOfDistribution` documentation.

        Parameters
        ----------
        features : np.ndarray, optional
          Feature array of shape ``(N, M)``, where N is the number of examples and M is the number of features used to represent each example.
          All features should be **numeric**. For less structured data (e.g. images, text, categorical values, ...), you should provide
          vector embeddings to represent each example (e.g. extracted from some pretrained neural network).

        pred_probs : np.ndarray, optional
           An array of shape ``(N, K)`` of model-predicted probabilities,
          ``P(label=k|x)``. Each row of this matrix corresponds
          to an example `x` and contains the model-predicted probabilities that
          `x` belongs to each possible class, for each of the K classes. The
          columns must be ordered such that these probabilities correspond to
          class 0, 1, ..., K-1.

        labels : array_like, optional
          A discrete vector of given labels for the data of shape ``(N,)``. Supported `array_like` types include: ``np.ndarray`` or ``list``.
          *Format requirements*: for dataset with K classes, labels must be in 0, 1, ..., K-1.
          All the classes (0, 1, ..., and K-1) MUST be present in ``labels``, such that: ``len(set(labels)) == pred_probs.shape[1]``
          If ``params["adjust_confident_thresholds"]`` was previously set to ``False``, you do not have to pass in `labels`.
          Note: multi-label classification is not supported by this method, each example must belong to a single class, e.g. ``labels = np.ndarray([1,0,2,1,1,0...])``.

        verbose : bool, default = True
          Set to ``False`` to suppress all print statements.

        """
        _ = self._shared_fit(
            features=features,
            pred_probs=pred_probs,
            labels=labels,
            verbose=verbose,
        )

    def score(
        self, *, features: Optional[np.ndarray] = None, pred_probs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Use fitted estimator and passed in `features` or `pred_probs` to calculate out-of-distribution scores for a dataset.

        Score for each example corresponds to the likelihood this example stems from the same distribution as the dataset previously specified in ``fit()`` (i.e. is not an outlier).

        If `features` are passed, returns OOD score for each example based on its feature values.
        If `pred_probs` are passed, returns OOD score for each example based on classifier's probabilistic predictions.
        You may have to previously call ``fit()`` or call ``fit_score()`` instead.

        Parameters
        ----------
        features : np.ndarray, optional
          Feature array of shape ``(N, M)``, where N is the number of examples and M is the number of features used to represent each example.
          For details, see `features` in `~cleanlab.outlier.OutOfDistribution.fit` function.

        pred_probs : np.ndarray, optional
          An array of shape ``(N, K)``  of predicted class probabilities output by a trained classifier.
          For details, see `pred_probs` in `~cleanlab.outlier.OutOfDistribution.fit` function.

        Returns
        -------
        scores : np.ndarray
          Scores lie in [0,1] with smaller values indicating examples that are less typical under the dataset distribution
          (values near 0 indicate outliers).

          If `features` are passed, `ood_features_scores` are returned.
          The score is based on the average distance between the example and its K nearest neighbors in the dataset
          (in feature space).

          If `pred_probs` are passed, `ood_predictions_scores` are returned.
          The score is based on the uncertainty in the classifier's predicted probabilities.
        """
        self._assert_valid_inputs(features, pred_probs)

        if features is not None:
            if self.params["knn"] is None:
                raise ValueError(
                    "OOD estimator needs to be fit on features first. Call `fit()` or `fit_scores()` before this function."
                )
            scores, _ = self._get_ood_features_scores(
                features, **self._get_params(self.OUTLIER_PARAMS)
            )

        if pred_probs is not None:
            if self.params["confident_thresholds"] is None and self.params["adjust_pred_probs"]:
                raise ValueError(
                    "OOD estimator needs to be fit on pred_probs first since params['adjust_pred_probs']=True. Call `fit()` or `fit_scores()` before this function."
                )
            scores, _ = _get_ood_predictions_scores(pred_probs, **self._get_params(self.OOD_PARAMS))

        return scores

    def _get_params(self, param_keys) -> dict:
        """Get function specific dictionary of parameters (i.e. only those in param_keys)."""
        return {k: v for k, v in self.params.items() if k in param_keys}

    @staticmethod
    def _assert_valid_params(params, param_keys):
        """Validate passed in params and get list of parameters in param that are not in param_keys."""
        if params is not None:
            wrong_params = list(set(params.keys()).difference(set(param_keys)))
            if len(wrong_params) > 0:
                raise ValueError(
                    f"Passed in params dict can only contain {param_keys}. Remove {wrong_params} from params dict."
                )

    @staticmethod
    def _assert_valid_inputs(features, pred_probs):
        """Check whether features and pred_prob inputs are valid, throw error if not."""
        if features is None and pred_probs is None:
            raise ValueError(
                "Not enough information to compute scores. Pass in either features or pred_probs."
            )

        if features is not None and pred_probs is not None:
            raise ValueError(
                "Cannot fit to OOD Estimator to both features and pred_probs. Pass in either one or the other."
            )

        if features is not None and len(features.shape) != 2:
            raise ValueError(
                "Feature array needs to be of shape (N, M), where N is the number of examples and M is the "
                "number of features used to represent each example. "
            )

    def _shared_fit(
        self,
        *,
        features: Optional[np.ndarray] = None,
        pred_probs: Optional[np.ndarray] = None,
        labels: Optional[LabelLike] = None,
        verbose: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Shared fit functionality between ``fit()`` and ``fit_score()``.

        For details, refer to `~cleanlab.outlier.OutOfDistribution.fit`
        or `~cleanlab.outlier.OutOfDistribution.fit_score`.
        """
        self._assert_valid_inputs(features, pred_probs)
        scores = None  # If none scores are returned, fit was skipped

        if features is not None:
            if self.params["knn"] is not None:
                # No fitting twice if knn object already fit
                warnings.warn(
                    "A KNN estimator has previously already been fit, call score() to apply it to data, or create a new OutOfDistribution object to fit a different estimator.",
                    UserWarning,
                )
            else:
                # Get ood features scores
                if verbose:
                    print("Fitting OOD estimator based on provided features ...")
                scores, knn = self._get_ood_features_scores(
                    features, **self._get_params(self.OUTLIER_PARAMS)
                )
                self.params["knn"] = knn

        if pred_probs is not None:
            if self.params["confident_thresholds"] is not None:
                # No fitting twice if confident_thresholds object already fit
                warnings.warn(
                    "Confident thresholds have previously already been fit, call score() to apply them to data, or create a new OutOfDistribution object to fit a different estimator.",
                    UserWarning,
                )
            else:
                # Get ood predictions scores
                if verbose:
                    print("Fitting OOD estimator based on provided pred_probs ...")
                scores, confident_thresholds = _get_ood_predictions_scores(
                    pred_probs,
                    labels=labels,
                    **self._get_params(self.OOD_PARAMS),
                )
                if confident_thresholds is None:
                    warnings.warn(
                        "No estimates need to be be fit under the provided params, so you could directly call "
                        "score() as an alternative.",
                        UserWarning,
                    )
                else:
                    self.params["confident_thresholds"] = confident_thresholds
        return scores

    def _get_ood_features_scores(
        self,
        features: Optional[np.ndarray] = None,
        knn: Optional[NearestNeighbors] = None,
        k: Optional[int] = None,
        t: int = 1,
    ) -> Tuple[np.ndarray, Optional[NearestNeighbors]]:
        """
        Return outlier score based on feature values using `k` nearest neighbors.

        The outlier score for each example is computed inversely proportional to
        the average distance between this example and its K nearest neighbors (in feature space).

        Parameters
        ----------
        features : np.ndarray
        Feature array of shape ``(N, M)``, where N is the number of examples and M is the number of features used to represent each example.
        For details, `features` in the same format expected by the `~cleanlab.outlier.OutOfDistribution.fit` function.

        knn : sklearn.neighbors.NearestNeighbors, default = None
        For details, see key `knn` in the params dict arg of `~cleanlab.outlier.OutOfDistribution`.

        k : int, default=None
        Optional number of neighbors to use when calculating outlier score (average distance to neighbors).
        For details, see key `k` in the params dict arg of `~cleanlab.outlier.OutOfDistribution`.

        t : int, default=1
        Controls transformation of distances between examples into similarity scores that lie in [0,1].
        For details, see key `t` in the params dict arg of `~cleanlab.outlier.OutOfDistribution`.

        Returns
        -------
        ood_features_scores : Tuple[np.ndarray, Optional[NearestNeighbors]]
        Return a tuple whose first element is array of `ood_features_scores` and second is a `knn` Estimator object.
        """
        DEFAULT_K = 10
        # fit skip over (if knn is not None) then skipping fit and suggest score else fit.
        if knn is None:  # setup default KNN estimator
            # Make sure both knn and features are not None
            if features is None:
                raise ValueError(
                    "Both knn and features arguments cannot be None at the same time. Not enough information to compute outlier scores."
                )
            if k is None:
                k = DEFAULT_K  # use default when knn and k are both None
            if k > len(features):  # Ensure number of neighbors less than number of examples
                raise ValueError(
                    f"Number of nearest neighbors k={k} cannot exceed the number of examples N={len(features)} passed into the estimator (knn)."
                )

            if features.shape[1] > 3:  # use euclidean distance for lower dimensional spaces
                metric = "cosine"
            else:
                metric = "euclidean"

            knn = NearestNeighbors(n_neighbors=k, metric=metric).fit(features)
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

        # Fit knn estimator on the features if a non-fitted estimator is passed in
        try:
            knn.kneighbors(features)
        except NotFittedError:
            knn.fit(features)

        # Get distances to k-nearest neighbors Note that the knn object contains the specification of distance metric
        # and n_neighbors (k value) If our query set of features matches the training set used to fit knn, the nearest
        # neighbor of each point is the point itself, at a distance of zero.
        distances, _ = knn.kneighbors(features)

        # Calculate average distance to k-nearest neighbors
        avg_knn_distances = distances[:, :k].mean(axis=1)

        if self.params["scaling_factor"] is None:
            self.params["scaling_factor"] = float(
                max(np.median(avg_knn_distances), np.finfo(np.float_).eps)
            )
        scaling_factor = self.params["scaling_factor"]

        if not isinstance(scaling_factor, float):
            raise ValueError(f"Scaling factor must be a float. Got {type(scaling_factor)} instead.")

        ood_features_scores = transform_distances_to_scores(
            avg_knn_distances, t, scaling_factor=scaling_factor
        )
        return (ood_features_scores, knn)


def _get_ood_predictions_scores(
    pred_probs: np.ndarray,
    *,
    labels: Optional[LabelLike] = None,
    confident_thresholds: Optional[np.ndarray] = None,
    adjust_pred_probs: bool = True,
    method: str = "entropy",
    M: int = 100,
    gamma: float = 0.1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return an OOD (out of distribution) score for each example based on it pred_prob values.

    Parameters
    ----------
    pred_probs : np.ndarray
      An array of shape ``(N, K)`` of model-predicted probabilities,
      `pred_probs` in the same format expected by the `~cleanlab.outlier.OutOfDistribution.fit` function.

    confident_thresholds : np.ndarray, default = None
      For details, see key `confident_thresholds` in the params dict arg of `~cleanlab.outlier.OutOfDistribution`.

    labels : array_like, optional
      `labels` in the same format expected by the `~cleanlab.outlier.OutOfDistribution.fit` function.

    adjust_pred_probs : bool, True
      Account for class imbalance in the label-quality scoring.
      For details, see key `adjust_pred_probs` in the params dict arg of `~cleanlab.outlier.OutOfDistribution`.

    method : {"entropy", "least_confidence", "gen"}, default="entropy"
      Which method to use for computing outlier scores based on pred_probs.
      For details see key `method` in the params dict arg of `~cleanlab.outlier.OutOfDistribution`.

    M : int, default=100
      For GEN method only. Hyperparameter that controls the number of top classes to consider when calculating OOD scores.

    gamma : float, default=0.1
      For GEN method only. Hyperparameter that controls the weight of the second term in the GEN score.


    Returns
    -------
    ood_predictions_scores : Tuple[np.ndarray, Optional[np.ndarray]]
      Returns a tuple. First element is array of `ood_predictions_scores` and second is an np.ndarray of `confident_thresholds` or None is 'confident_thresholds' is not calculated.
    """
    valid_methods = (
        "entropy",
        "least_confidence",
        "gen",
    )

    if (confident_thresholds is not None or labels is not None) and not adjust_pred_probs:
        warnings.warn(
            "OOD scores are not adjusted with confident thresholds. If scores need to be adjusted set "
            "params['adjusted_pred_probs'] = True. Otherwise passing in confident_thresholds and/or labels does not change "
            "score calculation.",
            UserWarning,
        )

    if adjust_pred_probs:
        if confident_thresholds is None:
            if labels is None:
                raise ValueError(
                    "Cannot calculate adjust_pred_probs without labels. Either pass in labels parameter or set "
                    "params['adjusted_pred_probs'] = False. "
                )
            labels = labels_to_array(labels)
            assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs, multi_label=False)
            confident_thresholds = get_confident_thresholds(labels, pred_probs, multi_label=False)

        pred_probs = _subtract_confident_thresholds(
            None, pred_probs, multi_label=False, confident_thresholds=confident_thresholds
        )

    # Scores are flipped so ood scores are closer to 0. Scores reflect confidence example is in-distribution.
    if method == "entropy":
        ood_predictions_scores = 1.0 - get_normalized_entropy(pred_probs)
    elif method == "least_confidence":
        ood_predictions_scores = pred_probs.max(axis=1)
    elif method == "gen":
        if pred_probs.shape[1] < M:  # pragma: no cover
            warnings.warn(
                f"GEN with the default hyperparameter settings is intended for datasets with at least {M} classes. You can adjust params['M'] according to the number of classes in your dataset.",
                UserWarning,
            )
        probs = softmax(pred_probs, axis=1)
        probs_sorted = np.sort(probs, axis=1)[:, -M:]
        ood_predictions_scores = (
            1 - np.sum(probs_sorted**gamma * (1 - probs_sorted) ** (gamma), axis=1) / M
        )  # Use 1 + original gen score/M to make the scores lie in 0-1
    else:
        raise ValueError(
            f"""
            {method} is not a valid OOD scoring method!
            Please choose a valid scoring_method: {valid_methods}
            """
        )

    return (
        ood_predictions_scores,
        confident_thresholds,
    )
