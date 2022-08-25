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
Class to identify which examples are out of distribution.
"""
from typing import Optional
import numpy as np
from cleanlab.rank import get_ood_scores, get_outlier_scores


class OutOfDistribution:
    OUTLIER_PARAMS = ["features", "k", "t"]
    OOD_PARAMS = ["labels", "adjust_pred_probs", "method"]
    DEFAULT_PARAM_DICT = {
        "features": None,  # outlier param
        "k": None,  # outlier param
        "t": 1,  # outlier param
        "labels": None,  # ood param
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
        verbose=True,
    ):

        scores = self._shared_fit(
            features=features, pred_probs=pred_probs, labels=labels, params=params
        )  # some of below code belongs in def of _shared_fit() instead

        if features is not None:
            # fit to outlier scores
            if verbose:
                print("Computing outlier scores based on provided features ...")

        if pred_probs is not None:
            # fit to ood scores
            if verbose:
                print("Computing ood scores based on provided pred_probs ...")

        return scores

    def fit(
        self,
        *,
        features: Optional[np.ndarray] = None,
        pred_probs: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        params: dict = None,
        verbose=True,
    ):

        _ = self._shared_fit(
            features=features, pred_probs=pred_probs, labels=labels, params=params
        )  # some of below code belongs in def of _shared_fit() instead

        if features is not None:
            # fit to outlier scores
            if verbose:
                print("Fitting OOD object based on provided features ...")

        if pred_probs is not None:
            # fit to ood scores
            if verbose:
                print("Fitting OOD object based on provided pred_probs ...")

    def score(
        self, *, features: Optional[np.ndarray] = None, pred_probs: Optional[np.ndarray] = None
    ):

        self._assert_valid_inputs(features, pred_probs)

        if features is not None:
            if self.knn is None:
                raise ValueError(
                    f"OOD Object needs to be fit first. Call fit() or fit_scores() before this function."
                )
            else:
                params = self._get_params(self.OUTLIER_PARAMS)  # get params specific to outliers
                scores = get_outlier_scores(features, self.knn, **params, return_estimator=False)

        if pred_probs is not None:
            if self.confident_thresholds is None and self.adjust_pred_probs is not None:
                raise ValueError(
                    f"OOD Object needs to be fit first. Call fit() or fit_scores() with proper params before this function."
                )
            else:
                params = self._get_params(self.OOD_params)  # get params specific to outliers
                scores = get_outlier_scores(features, self.knn, **params, return_estimator=False)

        return scores

    def _get_params(self, param_keys):
        return {k: v for k, v in self.params.items() if k in param_keys}

    def _assert_valid_inputs(self, pred_probs, features):
        if features is not None and pred_probs is not None:
            raise ValueError(
                f"Cannot fit to object to features and pred_probs. Pass in either one or the other."
            )

        if features is None and pred_probs is None:
            raise ValueError(
                f"Not enough information to compute scores. Pass in either features or pred_probs."
            )

    def _shared_fit(
        self,
        *,
        features: Optional[np.ndarray] = None,
        pred_probs: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        params: dict = None,
    ):

        self._assert_valid_inputs(features, pred_probs)
        self.params = self.DEFAULT_PARAM_DICT

        if params is not None:
            self.params = {**self.params, **params}

        if features is not None:
            # raise Error if they passed in invalid params keys for using features.
            if self.OOD_PARAMS in params.keys():
                raise ValueError(
                    f"Passed in params dict can only contain {self.OUTLIER_PARAMS}. Remove {list(set(params.keys()) & set(self.OOD_PARAMS))} from params dict."
                )

            # get outlier scores
            if self.verbose:
                print("Computing outlier scores based on provided features ...")
            scores, knn = get_outlier_scores(features, **params, return_estimator=True)
            self.knn = knn  # save estimator

        if pred_probs is not None:
            # raise Error if they passed in invalid params keys for using pred_probs.
            if self.OUTLIER_PARAMS in params.keys():
                raise ValueError(
                    f"Passed in params dict can only contain {self.OOD_PARAMS}. Remove {list(set(params.keys()) & set(self.OUTLIER_PARAMS))} from params dict."
                )

            # get ood scores
            if self.verbose:
                print("Computing ood scores based on provided features ...")
            scores, confident_thresholds = get_ood_scores(
                pred_probs, labels=labels, **params, return_estimator=True
            )
            self.confident_thresholds = confident_thresholds  # save estimator

        return scores
