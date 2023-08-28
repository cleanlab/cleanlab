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
cleanlab can be used for learning with noisy data for any dataset and regression model.

For regression tasks, the :py:class:`regression.learn.CleanLearning <cleanlab.regression.learn.CleanLearning>`
class wraps any instance of an sklearn model to allow you to train more robust regression models,
or use the model to identify corrupted values in the dataset.
The wrapped model must adhere to the `sklearn estimator API
<https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator>`_,
meaning it must define three functions:

* ``model.fit(X, y, sample_weight=None)``
* ``model.predict(X)``
* ``model.score(X, y, sample_weight=None)``

where ``X`` contains the data (i.e. features, covariates, independant variables) and ``y`` contains the target 
value (i.e. label, response/dependant variable). The first index of ``X`` and of ``y`` should correspond to the different 
examples in the dataset, such that ``len(X) = len(y) = N`` (sample-size).

Your model should be correctly clonable via
`sklearn.base.clone <https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html>`_:
cleanlab internally creates multiple instances of the model, and if you e.g. manually wrap a 
PyTorch model, ensure that every call to the estimator's ``__init__()`` creates an independent 
instance of the model (for sklearn compatibility, the weights of neural network models should typically 
be initialized inside of ``clf.fit()``).

Example
-------
>>> from cleanlab.regression.learn import CleanLearning
>>> from sklearn.linear_model import LinearRegression 
>>> cl = CleanLearning(clf=LinearRegression()) # Pass in any model.
>>> cl.fit(X, y_with_noise)
>>> # Estimate the predictions as if you had trained without label issues.
>>> predictions = cl.predict(y)

If your model is not sklearn-compatible by default, it might be the case that standard packages can adapt 
the model. For example, you can adapt PyTorch models using `skorch <https://skorch.readthedocs.io/>`_ 
and adapt Keras models using `SciKeras <https://www.adriangb.com/scikeras/>`_.

If an adapter doesn't already exist, you can manually wrap your 
model to be sklearn-compatible. This is made easy by inheriting from
`sklearn.base.BaseEstimator
<https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>`_:

.. code:: python

    from sklearn.base import BaseEstimator

    class YourModel(BaseEstimator):
        def __init__(self, ):
            pass
        def fit(self, X, y):
            pass
        def predict(self, X):
            pass
        def score(self, X, y):
            pass
            
"""

from typing import Optional, Union, Tuple
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

from cleanlab.typing import LabelLike
from cleanlab.internal.constants import TINY_VALUE
from cleanlab.internal.util import train_val_split, subset_X_y
from cleanlab.internal.regression_utils import assert_valid_regression_inputs
from cleanlab.internal.validation import labels_to_array


class CleanLearning(BaseEstimator):
    """
    CleanLearning = Machine Learning with cleaned data (even when training on messy, error-ridden data).

    Automated and robust learning with noisy labels using any dataset and any regression model.
    For regression tasks, this class trains a ``model`` with error-prone, noisy labels
    as if the model had been instead trained on a dataset with perfect labels.
    It achieves this by estimating which labels are noisy (you might solely use CleanLearning for this estimation)
    and then removing examples estimated to have noisy labels, such that a more robust copy of the same model can be
    trained on the remaining clean data.

    Parameters
    ----------
    model :
        Any regression model implementing the `sklearn estimator API <https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator>`_,
        defining the following functions:

        - ``model.fit(X, y)``
        - ``model.predict(X)``
        - ``model.score(X, y)``

        Default model used is `sklearn.linear_model.LinearRegression
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.

    cv_n_folds :
        This class needs holdout predictions for every data example and if not provided,
        uses cross-validation to compute them. This argument sets the number of cross-validation
        folds used to compute out-of-sample predictions for each example in ``X``. Default is 5.
        Larger values may produce better results, but requires longer to run.

    n_boot :
        Number of bootstrap resampling rounds used to estimate the model's epistemic uncertainty.
        Default is 5. Larger values are expected to produce better results but require longer runtimes.
        Set as 0 to skip estimating the epistemic uncertainty and get results faster.

    include_aleatoric_uncertainty :
        Specifies if the aleatoric uncertainty should be estimated during label error detection.
        ``True`` by default, which is expected to produce better results but require longer runtimes.

    verbose :
        Controls how much output is printed. Set to ``False`` to suppress print statements. Default `False`.

    seed :
        Set the default state of the random number generator used to split
        the data. By default, uses ``np.random`` current random state.
    """

    def __init__(
        self,
        model: Optional[BaseEstimator] = None,
        *,
        cv_n_folds: int = 5,
        n_boot: int = 5,
        include_aleatoric_uncertainty: bool = True,
        verbose: bool = False,
        seed: Optional[bool] = None,
    ):
        if model is None:
            # Use linear regression if no model is provided.
            model = LinearRegression()

        # Make sure the given regression model has the appropriate methods defined.
        if not hasattr(model, "fit"):
            raise ValueError("The model must define a .fit() method.")
        if not hasattr(model, "predict"):
            raise ValueError("The model must define a .predict() method.")

        if seed is not None:
            np.random.seed(seed=seed)

        if n_boot < 0:
            raise ValueError("n_boot cannot be a negative value")
        if cv_n_folds < 2:
            raise ValueError("cv_n_folds must be at least 2")

        self.model: BaseEstimator = model
        self.seed: Optional[int] = seed
        self.cv_n_folds: int = cv_n_folds
        self.n_boot: int = n_boot
        self.include_aleatoric_uncertainty: bool = include_aleatoric_uncertainty
        self.verbose: bool = verbose
        self.label_issues_df: Optional[pd.DataFrame] = None
        self.label_issues_mask: Optional[np.ndarray] = None
        self.k: Optional[float] = None  # frac flagged as issue

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: LabelLike,
        *,
        label_issues: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
        find_label_issues_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        model_final_kwargs: Optional[dict] = None,
    ) -> BaseEstimator:
        """
        Train regression ``model`` with error-prone, noisy labels as if the model had been instead trained
        on a dataset with the correct labels. ``fit`` achieves this by first training ``model`` via
        cross-validation on the noisy data, using the resulting predicted probabilities to identify label issues,
        pruning the data with label issues, and finally training ``model`` on the remaining clean data.

        Parameters
        ----------
        X :
            Data features (i.e. covariates, independent variables), typically an array of shape ``(N, ...)``,
            where N is the number of examples (sample-size).
            Your ``model`` must be able to ``fit()`` and ``predict()`` data of this format.

        y :
            An array of shape ``(N,)`` of noisy labels (i.e. target/response/dependant variable), where some values may be erroneous.

        label_issues :
            Optional already-identified label issues in the dataset (if previously estimated).
            Specify this to avoid re-estimating the label issues if already done.
            If ``pd.DataFrame``, must be formatted as the one returned by:
            :py:meth:`self.find_label_issues <cleanlab.regression.learn.CleanLearning.find_label_issues>` or
            :py:meth:`self.get_label_issues <cleanlab.regression.learn.CleanLearning.get_label_issues>`. The DataFrame must
            have a column named ``is_label_issue``.

            If ``np.ndarray``, the input must be a boolean mask of length ``N`` where examples that have label issues
            have the value ``True``, and the rest of the examples have the value ``False``.

        sample_weight :
            Optional array of weights with shape ``(N,)`` that are assigned to individual samples. Specifies how to weight the examples in
            the loss function while training.

        find_label_issues_kwargs:
            Optional keyword arguments to pass into :py:meth:`self.find_label_issues <cleanlab.regression.learn.CleanLearning.find_label_issues>`.

        model_kwargs :
            Optional keyword arguments to pass into model's ``fit()`` method.

        model_final_kwargs :
            Optional extra keyword arguments to pass into the final model's ``fit()`` on the cleaned data,
            but not the ``fit()`` in each fold of cross-validation on the noisy data.
            The final ``fit()`` will also receive the arguments in `clf_kwargs`, but these may be overwritten
            by values in `clf_final_kwargs`. This can be useful for training differently in the final ``fit()``
            than during cross-validation.

        Returns
        -------
        self : CleanLearning
            Fitted estimator that has all the same methods as any sklearn estimator.

            After calling ``self.fit()``, this estimator also stores extra attributes such as:

            - ``self.label_issues_df``: a ``pd.DataFrame`` containing label quality scores, boolean flags
                indicating which examples have label issues, and predicted label values for each example.
                Accessible via :py:meth:`self.get_label_issues <cleanlab.regression.learn.CleanLearning.get_label_issues>`,
                of similar format as the one returned by :py:meth:`self.find_label_issues <cleanlab.regression.learn.CleanLearning.find_label_issues>`.
                See documentation of :py:meth:`self.find_label_issues <cleanlab.regression.learn.CleanLearning.find_label_issues>`
                for column descriptions.
            - ``self.label_issues_mask``: a ``np.ndarray`` boolean mask indicating if a particular
                example has been identified to have issues.
        """
        assert_valid_regression_inputs(X, y)

        if find_label_issues_kwargs is None:
            find_label_issues_kwargs = {}
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
            if "sample_weight" not in inspect.signature(self.model.fit).parameters:
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
                **find_label_issues_kwargs,
            )
        else:
            if self.verbose:
                print("Using provided label_issues instead of finding label issues.")
                if self.label_issues_df is not None:
                    print(
                        "These will overwrite self.label_issues_df and will be returned by "
                        "`self.get_label_issues()`. "
                    )

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

    def predict(self, X: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Predict class labels using your wrapped model.
        Works just like ``model.predict()``.

        Parameters
        ----------
        X : np.ndarray or DatasetLike
            Test data in the same format expected by your wrapped regression model.

        Returns
        -------
        predictions : np.ndarray
            Predictions for the test examples.
        """
        return self.model.predict(X, *args, **kwargs)

    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: LabelLike,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """Evaluates your wrapped regression model's score on a test set `X` with target values `y`.
        Uses your model's default scoring function, or r-squared score if your model as no ``"score"`` attribute.

        Parameters
        ----------
        X :
            Test data in the same format expected by your wrapped model.

        y :
            Test labels in the same format as labels previously used in ``fit()``.

        sample_weight :
            Optional array of shape ``(N,)`` or ``(N, 1)`` used to weight each test example when computing the score.

        Returns
        -------
        score : float
            Number quantifying the performance of this regression model on the test data.
        """
        if hasattr(self.model, "score"):
            if "sample_weight" in inspect.signature(self.model.score).parameters:
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
        X: Union[np.ndarray, pd.DataFrame],
        y: LabelLike,
        *,
        uncertainty: Optional[Union[np.ndarray, float]] = None,
        coarse_search_range: list = [0.01, 0.05, 0.1, 0.15, 0.2],
        fine_search_size: int = 3,
        save_space: bool = False,
        model_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Identifies potential label issues (corrupted `y`-values) in the dataset, and estimates how noisy each label is.

        Note: this method estimates the label issues from scratch. To access previously-estimated label issues from
        this :py:class:`CleanLearning <cleanlab.regression.learn.CleanLearning>` instance, use the
        :py:meth:`self.get_label_issues <cleanlab.regression.learn.CleanLearning.get_label_issues>` method.

        This is the method called to find label issues inside
        :py:meth:`CleanLearning.fit() <cleanlab.regression.learn.CleanLearning.fit>`
        and they share mostly the same parameters.

        Parameters
        ----------
        X :
            Data features (i.e. covariates, independent variables), typically an array of shape ``(N, ...)``,
            where N is the number of examples (sample-size).
            Your ``model``, must be able to ``fit()`` and ``predict()`` data of this format.

        y :
            An array of shape ``(N,)`` of noisy labels (i.e. target/response/dependant variable), where some values may be erroneous.

        uncertainty :
            Optional estimated uncertainty for each example. Should be passed in as a float (constant uncertainty throughout all examples),
            or a numpy array of length ``N`` (estimated uncertainty for each example).
            If not provided, this method will estimate the uncertainty as the sum of the epistemic and aleatoric uncertainty.

        save_space :
            If True, then returned ``label_issues_df`` will not be stored as attribute.
            This means some other methods like :py:meth:`self.get_label_issues <cleanlab.regression.learn.CleanLearning.get_label_issues>` will no longer work.

        coarse_search_range :
            The coarse search range to find the value of ``k``, which estimates the fraction of data which have label issues.
            More values represent a more thorough search (better expected results but longer runtimes).

        fine_search_size :
            Size of fine-grained search grid to find the value of ``k``, which represents our estimate of the fraction of data which have label issues.
            A higher number represents a more thorough search (better expected results but longer runtimes).


        For info about the **other parameters**, see the docstring of :py:meth:`CleanLearning.fit()
        <cleanlab.regression.learn.CleanLearning.fit>`.

        Returns
        -------
        label_issues_df : pd.DataFrame
            DataFrame with info about label issues for each example.
            Unless `save_space` argument is specified, same DataFrame is also stored as `self.label_issues_df` attribute accessible via
            :py:meth:`get_label_issues<cleanlab.regression.learn.CleanLearning.get_label_issues>`.

            Each row represents an example from our dataset and the DataFrame may contain the following columns:

            - *is_label_issue*: boolean mask for the entire dataset where ``True`` represents a label issue and ``False`` represents an example
              that is accurately labeled with high confidence.
            - *label_quality*: Numeric score that measures the quality of each label (how likely it is to be correct,
              with lower scores indicating potentially erroneous labels).
            - *given_label*: Values originally given for this example (same as `y` input).
            - *predicted_label*: Values predicted by the trained model.
        """

        X, y = assert_valid_regression_inputs(X, y)

        if model_kwargs is None:
            model_kwargs = {}

        if self.verbose:
            print("Identifying label issues ...")

        # compute initial values to find best k
        initial_predictions = self._get_cv_predictions(X, y, model_kwargs=model_kwargs)
        initial_residual = initial_predictions - y
        initial_sorted_index = np.argsort(abs(initial_residual))
        initial_r2 = r2_score(y, initial_predictions)

        self.k, r2 = self._find_best_k(
            X=X,
            y=y,
            sorted_index=initial_sorted_index,
            coarse_search_range=coarse_search_range,
            fine_search_size=fine_search_size,
        )

        # check if initial r2 score (ie. not removing anything) is the best
        if initial_r2 >= r2:
            self.k = 0

        # get predictions using the best k
        predictions = self._get_cv_predictions(
            X, y, sorted_index=initial_sorted_index, k=self.k, model_kwargs=model_kwargs
        )
        residual = predictions - y

        if uncertainty is None:
            epistemic_uncertainty = self.get_epistemic_uncertainty(X, y, predictions=predictions)
            if self.include_aleatoric_uncertainty:
                aleatoric_uncertainty = self.get_aleatoric_uncertainty(X, residual)
            else:
                aleatoric_uncertainty = 0
            uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        else:
            if isinstance(uncertainty, np.ndarray) and len(y) != len(uncertainty):
                raise ValueError(
                    "If uncertainty is passed in as an array, it must have the same length as y."
                )

        label_quality_scores = np.exp(-abs(residual) / (uncertainty + TINY_VALUE))

        label_issues_mask = np.zeros(len(y), dtype=bool)
        num_issues = math.ceil(len(y) * self.k)
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
            self.label_issues_mask = label_issues_df["is_label_issue"].to_numpy()
        elif self.verbose:
            print("Not storing label_issues as attributes since save_space was specified.")

        return label_issues_df

    def get_label_issues(self) -> Optional[pd.DataFrame]:
        """
        Accessor, returns `label_issues_df` attribute if previously computed.
        This ``pd.DataFrame`` describes the issues identified for each example (each row corresponds to an example).
        For column definitions, see the documentation of
        :py:meth:`CleanLearning.find_label_issues<cleanlab.regression.learn.CleanLearning.find_label_issues>`.

        Returns
        -------
        label_issues_df : pd.DataFrame
            DataFrame with (precomputed) info about the label issues for each example.
        """
        if self.label_issues_df is None:
            warnings.warn(
                "Label issues have not yet been computed. Run `self.find_label_issues()` or `self.fit()` first."
            )
        return self.label_issues_df

    def get_epistemic_uncertainty(
        self,
        X: np.ndarray,
        y: np.ndarray,
        predictions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute the epistemic uncertainty of the regression model for each example. This uncertainty is estimated using the bootstrapped
        variance of the model predictions.

        Parameters
        ----------
        X :
            Data features (i.e. training inputs for ML), typically an array of shape ``(N, ...)``, where N is the number of examples.

        y :
            An array of shape ``(N,)`` of target values (dependant variables), where some values may be erroneous.

        predictions :
            Model predicted values of y, will be used as an extra bootstrap iteration to calculate the variance.

        Returns
        _______
        epistemic_uncertainty : np.ndarray
            The estimated epistemic uncertainty for each example.
        """
        X, y = assert_valid_regression_inputs(X, y)

        if self.n_boot == 0:  # does not estimate epistemic uncertainty
            return np.zeros(len(y))
        else:
            bootstrap_predictions = np.zeros(shape=(len(y), self.n_boot))
            for i in range(self.n_boot):
                bootstrap_predictions[:, i] = self._get_cv_predictions(X, y, cv_n_folds=2)

            # add a set of predictions from model that was already trained
            if predictions is not None:
                _, predictions = assert_valid_regression_inputs(X, predictions)
                bootstrap_predictions = np.hstack(
                    [bootstrap_predictions, predictions.reshape(-1, 1)]
                )

            return np.sqrt(np.var(bootstrap_predictions, axis=1))

    def get_aleatoric_uncertainty(
        self,
        X: np.ndarray,
        residual: np.ndarray,
    ) -> float:
        """
        Compute the aleatoric uncertainty of the data. This uncertainty is estimated by predicting the standard deviation
        of the regression error.

        Parameters
        ----------
        X :
            Data features (i.e. training inputs for ML), typically an array of shape ``(N, ...)``, where N is the number of examples.

        residual :
            The difference between the given value and the model predicted value of each examples, ie.
            `predictions - y`.

        Returns
        _______
        aleatoric_uncertainty : float
            The overall estimated aleatoric uncertainty for this dataset.
        """
        X, residual = assert_valid_regression_inputs(X, residual)
        residual_predictions = self._get_cv_predictions(X, residual)
        return np.sqrt(np.var(residual_predictions))

    def save_space(self):
        """
        Clears non-sklearn attributes of this estimator to save space (in-place).
        This includes the DataFrame attribute that stored label issues which may be large for big datasets.
        You may want to call this method before deploying this model (i.e. if you just care about producing predictions).
        After calling this method, certain non-prediction-related attributes/functionality will no longer be available
        """
        if self.label_issues_df is None and self.verbose:
            print("self.label_issues_df is already empty")

        self.label_issues_df = None
        self.label_issues_mask = None
        self.k = None

        if self.verbose:
            print("Deleted non-sklearn attributes such as label_issues_df to save space.")

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
    ) -> np.ndarray:
        """
        Helper method to get out-of-fold predictions using cross validation.
        This method also allows us to filter out the bottom k percent of label errors before training the cross-validation models
        (both ``sorted_index`` and ``k`` has to be provided for this).

        Parameters
        ----------
        X :
            Data features (i.e. training inputs for ML), typically an array of shape ``(N, ...)``, where N is the number of examples.

        y :
            An array of shape ``(N,)`` of target values (dependant variables), where some values may be erroneous.

        sorted_index :
            Index of each example sorted by their residuals in ascending order.

        k :
            The fraction of examples to hold out from the training sets. Usually this is the fraction of examples that are
            deemed to contain errors.

        """
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

        if len(in_sample_idx) < cv_n_folds:
            raise ValueError(
                f"There are too few examples to conduct {cv_n_folds}-fold cross validation. "
                "You can either reduce cv_n_folds for cross validation, or decrease k to exclude less data."
            )

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
    ) -> Tuple[float, float]:
        """
        Helper method that conducts a coarse and fine grained grid search to determine the best value
        of k, the fraction of the dataset that contains issues.

        Returns a tuple containing the the best value of k (ie. the one that has the best r squared score),
        and the corrsponding r squared score obtained when dropping k% of the data.
        """
        if len(coarse_search_range) == 0:
            raise ValueError("coarse_search_range must have at least 1 value of k")
        elif len(coarse_search_range) == 1:
            curr_k = coarse_search_range[0]
            num_examples_kept = math.floor(len(y) * (1 - curr_k))
            if num_examples_kept < self.cv_n_folds:
                raise ValueError(
                    f"There are too few examples to conduct {self.cv_n_folds}-fold cross validation. "
                    "You can either reduce self.cv_n_folds for cross validation, or decrease k to exclude less data."
                )
            predictions = self._get_cv_predictions(
                X=X,
                y=y,
                sorted_index=sorted_index,
                k=curr_k,
            )
            best_r2 = r2_score(y, predictions)
            best_k = coarse_search_range[0]
        else:
            # conduct coarse search
            coarse_search_range = sorted(coarse_search_range)  # sort to conduct fine search well
            r2_coarse = np.full(len(coarse_search_range), np.NaN)
            for i in range(len(coarse_search_range)):
                curr_k = coarse_search_range[i]
                num_examples_kept = math.floor(len(y) * (1 - curr_k))
                # check if there are too few examples to do cross val
                if num_examples_kept < self.cv_n_folds:
                    r2_coarse[i] = -1e30  # arbitrary large negative number
                else:
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
                best_r2 = np.max(r2_coarse)
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
                    num_examples_kept = math.floor(len(y) * (1 - curr_k))
                    # check if there are too few examples to do cross val
                    if num_examples_kept < self.cv_n_folds:
                        r2_fine[i] = -1e30  # arbitrary large negative number
                    else:
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
                    best_r2 = np.max(r2_coarse)
                else:
                    best_k = fine_search_range[np.argmax(r2_fine)]
                    best_r2 = np.max(r2_fine)

        return best_k, best_r2

    def _process_label_issues_arg(
        self,
        label_issues: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: LabelLike,
    ) -> pd.DataFrame:
        """
        Helper method to process the label_issues input into a well-formatted DataFrame.
        """
        y = labels_to_array(y)

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

        elif isinstance(label_issues, (pd.Series, np.ndarray)):
            if label_issues.dtype is not np.dtype("bool"):
                raise ValueError("If label_issues is numpy.array, dtype must be 'bool'.")
            if label_issues.shape != y.shape:
                raise ValueError("label_issues must have same shape as labels")
            return pd.DataFrame({"is_label_issue": label_issues, "given_label": y})

        else:
            raise ValueError(
                "label_issues must be either pandas.DataFrame, pandas.Series or numpy.ndarray"
            )
