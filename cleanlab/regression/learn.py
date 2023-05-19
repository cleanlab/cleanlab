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

from typing import Optional, Union
import inspect
import warnings

import math
import numpy as np
import pandas as pd

import sklearn.base
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from cleanlab.internal.util import train_val_split, subset_X_y


class CleanLearning(BaseEstimator):
    def __init__(
        self,
        model=None,
        *,
        cv_n_folds=5,
        n_boot=5,
        verbose=False,
        seed=None,
    ):
        if model is None:
            # Use linear regression if no classifier is provided.
            model = LinearRegression()

        # Make sure the given regression model has the appropriate methods defined.
        if not hasattr(model, "fit"):
            raise ValueError("The model must define a .fit() method.")
        if not hasattr(model, "predict"):
            raise ValueError("The model must define a .predict() method.")

        if seed is not None:
            np.random.seed(seed=seed)

        self.model = model
        self.seed = seed

        self.cv_n_folds = cv_n_folds
        self.n_boot = n_boot
        self.verbose = verbose

        self.label_issues_df = None
        self.label_issues_mask = None

        self._k = None  # frac flagged as issue

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
        label_issues: Optional[np.ndarray] = None,
        model_kwargs: Optional[dict] = None,
        model_final_kwargs: Optional[dict] = None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if model_final_kwargs is None:
            model_final_kwargs = {}
        model_final_kwargs = {**model_kwargs, **model_final_kwargs}

        if "sample_weight" in model_kwargs or "sample_weight" in model_final_kwargs:
            raise ValueError(
                "sample_weight should be provided directly in fit() rather than in model_kwargs or model_final_kwargs"
            )

        if sample_weight is not None:
            if "sample_weight" not in inspect.getfullargspec(self.model.fit).args:
                raise ValueError(
                    "sample_weight must be a supported fit() argument for your model in order to be specified here"
                )
            if len(sample_weight) != len(X):
                raise ValueError("sample_weight must be a 1D array that has the same length as y.")

        if label_issues is None:
            if self.label_issues_df is not None and self.verbose:
                print(
                    "If you already ran self.find_label_issues() and don't want to recompute, you "
                    "should pass the label_issues in as a parameter to this function next time."
                )

            label_issues = self.find_label_issues(
                X,
                y,
                model_kwargs=model_kwargs,
            )
        else:
            if self.verbose:
                print("Using provided label_issues instead of finding label issues.")
                if self.label_issues_df is not None:
                    print(
                        "These will overwrite self.label_issues_df and will be returned by "
                        "`self.get_label_issues()`. "
                    )

        # TODO: assert right format
        self.label_issues_df = self._process_label_issues_arg(label_issues, y)
        self.label_issues_mask = self.label_issues_df["is_label_issue"].to_numpy()

        X_mask = np.invert(self.label_issues_mask)
        X_cleaned, y_cleaned = subset_X_y(X, y, X_mask)
        if self.verbose:
            print(f"Pruning {np.sum(self.label_issues_mask)} examples with label issues ...")
            print(f"Remaining clean data has {len(y_cleaned)} examples.")

        if sample_weight is not None:
            model_final_kwargs["sample_weight"] = sample_weight[X_mask]
            if self.verbose:
                print("Fitting final model on the clean data with custom sample_weight ...")
        else:
            if self.verbose:
                print("Fitting final model on the clean data ...")

        self.model.fit(X_cleaned, y_cleaned, **model_final_kwargs)

        if self.verbose:
            print(
                "Label issues stored in label_issues_df DataFrame accessible via: self.get_label_issues(). "
                "Call self.save_space() to delete this potentially large DataFrame attribute."
            )
        return self

    def predict(self, X: np.ndarray, *args, **kwargs):
        return self.model.predict(X, *args, **kwargs)

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        if hasattr(self.model, "score"):
            if "sample_weight" in inspect.getfullargspec(self.model.score).args:
                return self.model.score(X, y, sample_weight=sample_weight)
            else:
                return self.model.score(X, y)
        else:
            return r2_score(
                y,
                self.model.predict(X),
                sample_weight=sample_weight,
            )

    def find_label_issues(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        uncertainty: Optional[Union[np.ndarray, float]] = None,
        coarse_search_range: list = [0.01, 0.05, 0.1, 0.15, 0.2],
        fine_search_size: int = 3,
        save_space: bool = False,
        model_kwargs: Optional[dict] = None,
    ):
        # TODO: add sample weight arg

        if model_kwargs is None:
            model_kwargs = {}

        if self.verbose:
            print("Identifying label issues ...")

        # compute initial values to find best k
        initial_predictions = self._get_cv_predictions(X, y, model_kwargs=model_kwargs)
        initial_residual = initial_predictions - y
        initial_sorted_index = np.argsort(abs(initial_residual))

        k = self._find_best_k(
            X=X,
            y=y,
            sorted_index=initial_sorted_index,
            coarse_search_range=coarse_search_range,
            fine_search_size=fine_search_size,
        )

        # get predictions using the best k
        predictions = self._get_cv_predictions(
            X, y, sorted_index=initial_sorted_index, k=k, model_kwargs=model_kwargs
        )
        residual = predictions - y

        if uncertainty is None:
            epistemic_uncertainty = self.get_epistemic_uncertainty(X, y)
            aleatoric_uncertainty = self.get_aleatoric_uncertainty(X, residual)
            uncertainty = epistemic_uncertainty + aleatoric_uncertainty

        label_quality_scores = np.exp(-abs(residual / uncertainty))

        label_issues_mask = np.zeros(len(y), dtype=bool)
        num_issues = math.ceil(len(y) * k)
        issues_index = np.argsort(label_quality_scores)[:num_issues]
        label_issues_mask[issues_index] = True

        # convert predictions to int if input is int
        if y.dtype == int:
            predictions = predictions.astype(int)

        label_issues_df = pd.DataFrame(
            {
                "is_label_issue": label_issues_mask,
                "label_quality": label_quality_scores,
                "given_label": y,
                "predicted_label": predictions,
            }
        )

        if self.verbose:
            print(f"Identified {np.sum(label_issues_mask)} examples with label issues.")

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
            print("Not storing label_issues as attributes since save_space was specified.")

        return label_issues_df

    def get_label_issues(self):
        if self.label_issues_df is None:
            warnings.warn(
                "Label issues have not yet been computed. Run `self.find_label_issues()` or `self.fit()` first."
            )
        return self.label_issues_df

    def get_epistemic_uncertainty(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        bootstrap_predictions = np.zeros(shape=(len(y), self.n_boot))
        for i in range(self.n_boot):
            bootstrap_predictions[:, i] = self._get_cv_predictions(X, y, cv_n_folds=2)

        return np.sqrt(np.var(bootstrap_predictions, axis=1))

    def get_aleatoric_uncertainty(
        self,
        X: np.ndarray,
        residual: np.ndarray,
    ):
        residual_predictions = self._get_cv_predictions(X, residual)
        return np.sqrt(np.var(residual_predictions))

    def _get_cv_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sorted_index: Optional[np.ndarray] = None,
        k: float = 0,
        *,
        cv_n_folds: Optional[int] = None,
        seed: Optional[int] = None,
        model_kwargs: Optional[dict] = None,
    ):
        # set to default unless specified otherwise
        if cv_n_folds is None:
            cv_n_folds = self.cv_n_folds

        if model_kwargs is None:
            model_kwargs = {}

        if k < 0 or k > 1:
            raise ValueError("k must be a value between 0 and 1")
        elif k == 0:
            if sorted_index is None:
                sorted_index = np.array(range(len(y)))
            in_sample_idx = sorted_index
        else:
            if sorted_index is None:
                # TODO: better error message
                raise ValueError(
                    "You need to pass in the index sorted by prediction quality to use with k"
                )
            num_to_drop = math.ceil(len(sorted_index) * k)
            in_sample_idx = sorted_index[:-num_to_drop]
            out_of_sample_idx = sorted_index[-num_to_drop:]

            X_out_of_sample = X[out_of_sample_idx]
            out_of_sample_predictions = np.zeros(shape=[len(out_of_sample_idx), cv_n_folds])

        predictions = np.zeros(shape=len(y))

        kf = KFold(n_splits=cv_n_folds, shuffle=True, random_state=seed)

        for k_split, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(in_sample_idx)):
            try:
                model_copy = sklearn.base.clone(self.model)  # fresh untrained copy of the model
            except Exception:
                raise ValueError(
                    "`model` must be clonable via: sklearn.base.clone(model). "
                    "You can either implement instance method `model.get_params()` to produce a fresh untrained copy of this model, "
                    "or you can implement the cross-validation outside of cleanlab "
                    "and pass in the obtained `pred_probs` to skip cleanlab's internal cross-validation"
                )

            # map the index to the actual index in the original dataset
            data_idx_train, data_idx_holdout = (
                in_sample_idx[cv_train_idx],
                in_sample_idx[cv_holdout_idx],
            )

            X_train_cv, X_holdout_cv, y_train_cv, y_holdout_cv = train_val_split(
                X, y, data_idx_train, data_idx_holdout
            )

            model_copy.fit(X_train_cv, y_train_cv, **model_kwargs)
            predictions_cv = model_copy.predict(X_holdout_cv)

            predictions[data_idx_holdout] = predictions_cv

            if k != 0:
                out_of_sample_predictions[:, k_split] = model_copy.predict(X_out_of_sample)

        if k != 0:
            out_of_sample_predictions_avg = np.mean(out_of_sample_predictions, axis=1)
            predictions[out_of_sample_idx] = out_of_sample_predictions_avg

        return predictions

    def _find_best_k(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sorted_index: np.ndarray,
        coarse_search_range: list = [0.01, 0.05, 0.1, 0.15, 0.2],
        fine_search_size: int = 3,
    ):
        if len(coarse_search_range) == 0:
            raise ValueError("coarse_search_range must have at least 1 value of k")
        elif len(coarse_search_range) == 1:
            best_k = coarse_search_range[0]
        else:
            # conduct coarse search
            coarse_search_range = np.sort(coarse_search_range)  # sort to conduct fine search well
            r2_coarse = np.full(len(coarse_search_range), np.NaN)
            for i in range(len(coarse_search_range)):
                curr_k = coarse_search_range[i]
                predictions = self._get_cv_predictions(
                    X=X,
                    y=y,
                    sorted_index=sorted_index,
                    k=curr_k,
                )
                r2_coarse[i] = r2_score(y, predictions)

            max_r2_ind = np.argmax(r2_coarse)

            # conduct fine search
            if fine_search_size < 0:
                raise ValueError("fine_search_size must at least 0")
            elif fine_search_size == 0:
                best_k = coarse_search_range[np.argmax(r2_coarse)]
            else:
                fine_search_range = np.array([])
                if max_r2_ind != 0:
                    fine_search_range = np.append(
                        np.linspace(
                            coarse_search_range[max_r2_ind - 1],
                            coarse_search_range[max_r2_ind],
                            fine_search_size + 1,
                            endpoint=False,
                        )[1:],
                        fine_search_range,
                    )
                if max_r2_ind != len(coarse_search_range) - 1:
                    fine_search_range = np.append(
                        fine_search_range,
                        np.linspace(
                            coarse_search_range[max_r2_ind],
                            coarse_search_range[max_r2_ind + 1],
                            fine_search_size + 1,
                            endpoint=False,
                        )[1:],
                    )

                r2_fine = np.full(len(fine_search_range), np.NaN)
                for i in range(len(fine_search_range)):
                    curr_k = fine_search_range[i]
                    predictions = self._get_cv_predictions(
                        X=X,
                        y=y,
                        sorted_index=sorted_index,
                        k=curr_k,
                    )
                    r2_fine[i] = r2_score(y, predictions)

                # check the max between coarse and fine search
                if max(r2_coarse) > max(r2_fine):
                    best_k = coarse_search_range[np.argmax(r2_coarse)]
                else:
                    best_k = fine_search_range[np.argmax(r2_fine)]

        self._k = best_k
        return best_k

    def _process_label_issues_arg(
        self,
        label_issues: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
    ):
        """
        Helper method to process the label_issues input into a well-formatted DataFrame.
        """

        if isinstance(label_issues, pd.DataFrame):
            if "is_label_issue" not in label_issues.columns:
                raise ValueError(
                    "DataFrame label_issues must contain column: 'is_label_issue'. "
                    "See CleanLearning.fit() documentation for label_issues column descriptions."
                )
            if len(label_issues) != len(y):
                raise ValueError("label_issues and labels must have same length")
            if "given_label" in label_issues.columns and np.any(
                label_issues["given_label"].to_numpy() != y
            ):
                raise ValueError("labels must match label_issues['given_label']")
            return label_issues

        elif isinstance(label_issues, np.ndarray):
            if label_issues.dtype is not np.dtype("bool"):
                raise ValueError("If label_issues is numpy.array, dtype must be 'bool'.")
            if label_issues.shape != y.shape:
                raise ValueError("label_issues must have same shape as labels")
            return pd.DataFrame({"is_label_issue": label_issues, "given_label": y})

        else:
            raise ValueError("label_issues must be either pandas.DataFrame or numpy.ndarray")
