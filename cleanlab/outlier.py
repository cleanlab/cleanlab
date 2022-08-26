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
    OUTLIER_PARAMS = tuple(("k", "t"))
    OOD_PARAMS = tuple(("labels", "adjust_pred_probs", "method"))
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
        verbose: bool = True,
    ):
        _ = self._shared_fit(features=features, pred_probs=pred_probs, labels=labels, params=params)

        if features is not None:  # fit to outlier scores
            if verbose:
                print("Fitting OOD object based on provided features ...")

        if pred_probs is not None:  # fit to ood scores
            if verbose:
                print("Fitting OOD object based on provided pred_probs ...")

    def score(
        self, *, features: Optional[np.ndarray] = None, pred_probs: Optional[np.ndarray] = None
    ):

        self._assert_valid_inputs(features, pred_probs)

        if features is not None:
            if self.knn is None:
                raise ValueError(
                    f"OOD Object needs to be fit on features first. Call fit() or fit_scores() before this function."
                )
            else:
                params = self._get_params(self.OUTLIER_PARAMS)  # get params specific to outliers
                scores = get_outlier_scores(features, self.knn, **params, return_estimator=False)

        if pred_probs is not None:
            if self.confident_thresholds is None and self.params["adjust_pred_probs"]:
                raise ValueError(
                    f"OOD Object needs to be fit on pred_probs with param adjust_pred_probs=True first. Call fit() or "
                    f"fit_scores() before this function. "
                )
            else:
                params = self._get_params(self.OOD_PARAMS)  # get params specific to outliers
                scores = get_ood_scores(
                    pred_probs,
                    confident_thresholds=self.confident_thresholds,
                    **params,
                    return_thresholds=False,
                )

        return scores

    def _get_params(self, param_keys):
        return {k: v for k, v in self.params.items() if k in param_keys}

    def _get_invalid_params(self, params, param_keys):
        if params is None:
            return []
        return list(set(params.keys()).difference(set(param_keys)))

    def _assert_valid_inputs(self, features, pred_probs):
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
    ):
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
                print("Computing outlier scores based on provided features ...")
            scores, knn = get_outlier_scores(
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
                print("Computing ood scores based on provided features ...")
            scores, confident_thresholds = get_ood_scores(
                pred_probs,
                labels=labels,
                **self._get_params(self.OOD_PARAMS),
                return_thresholds=True,
            )
            self.confident_thresholds = confident_thresholds

        return scores
