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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional
import numpy as np
import pandas as pd

from cleanlab.regression.learn import CleanLearning
from cleanlab.datalab.internal.issue_manager import IssueManager
from cleanlab.regression.rank import get_label_quality_scores

if TYPE_CHECKING:  # pragma: no cover
    from cleanlab.datalab.datalab import Datalab


class RegressionLabelIssueManager(IssueManager):
    """Manages label issues in a Datalab for regression tasks.

    Parameters
    ----------
    datalab :
        A Datalab instance.

    clean_learning_kwargs :
        Keyword arguments to pass to the :py:meth:`regression.learn.CleanLearning <cleanlab.regression.learn.CleanLearning>` constructor.

    threshold :
        The threshold to use to determine if an example has a label issue. It is a multiplier
        of the median label quality score that sets the absolute threshold. Only used if
        predictions are provided to `~RegressionLabelIssueManager.find_issues`, not if
        features are provided. Default is 0.05.
    """

    description: ClassVar[
        str
    ] = """Examples whose given label is estimated to be potentially incorrect
    (e.g. due to annotation error) are flagged as having label issues.
    """

    issue_name: ClassVar[str] = "label"
    verbosity_levels = {
        0: [],
        1: [],
        2: [],
        3: [],  # TODO
    }

    def __init__(
        self,
        datalab: Datalab,
        clean_learning_kwargs: Optional[Dict[str, Any]] = None,
        threshold: float = 0.05,
        health_summary_parameters: Optional[Dict[str, Any]] = None,
        **_,
    ):
        super().__init__(datalab)
        self.cl = CleanLearning(**(clean_learning_kwargs or {}))
        # This is a field for prioritizing features only when using a custom model
        self._uses_custom_model = "model" in (clean_learning_kwargs or {})
        self.threshold = threshold

    def find_issues(
        self,
        features: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Find label issues in the datalab.

        .. admonition:: Priority Order for finding issues:

            1. Custom Model: Requires `features` to be passed to this method. Used if a model is set up in the constructor.
            2. Predictions: Uses `predictions` if provided and no model is set up in the constructor.
            3. Default Model: Defaults to a standard model using `features` if no model or predictions are provided.
        """
        if features is None and predictions is None:
            raise ValueError(
                "Regression requires numerical `features` or `predictions` "
                "to be passed in as an argument to `find_issues`."
            )
        if features is None and self._uses_custom_model:
            raise ValueError(
                "Regression requires numerical `features` to be passed in as an argument to `find_issues` "
                "when using a custom model."
            )
        # If features are provided and either a custom model is used or no predictions are provided
        use_features = features is not None and (self._uses_custom_model or predictions is None)
        labels = self.datalab.labels
        if not isinstance(labels, np.ndarray):
            error_msg = (
                f"Expected labels to be a numpy array of shape (n_samples,) to use with RegressionLabelIssueManager, "
                f"but got {type(labels)} instead."
            )
            raise TypeError(error_msg)
        if use_features:
            assert features is not None  # mypy won't narrow the type for some reason
            self.issues = find_issues_with_features(
                features=features,
                y=labels,
                cl=self.cl,
                **kwargs,  # function sanitizes kwargs
            )
            self.issues.rename(columns={"label_quality": self.issue_score_key}, inplace=True)

        # Otherwise, if predictions are provided, process them
        else:
            assert predictions is not None  # mypy won't narrow the type for some reason
            self.issues = find_issues_with_predictions(
                predictions=predictions,
                y=labels,
                **{**kwargs, **{"threshold": self.threshold}},  # function sanitizes kwargs
            )

        # Get a summarized dataframe of the label issues
        self.summary = self.make_summary(score=self.issues[self.issue_score_key].mean())

        # Collect info about the label issues
        self.info = self.collect_info(issues=self.issues)

        # Drop columns from issues that are in the info
        self.issues = self.issues.drop(columns=["given_label", "predicted_label"])

    def collect_info(self, issues: pd.DataFrame) -> dict:
        issues_info = {
            "num_label_issues": sum(issues[f"is_{self.issue_name}_issue"]),
            "average_label_quality": issues[self.issue_score_key].mean(),
            "given_label": issues["given_label"].tolist(),
            "predicted_label": issues["predicted_label"].tolist(),
        }

        # health_summary_info, cl_info kept just for consistency with classification, but it could be just return issues_info
        health_summary_info: dict = {}
        cl_info: dict = {}

        info_dict = {
            **issues_info,
            **health_summary_info,
            **cl_info,
        }

        return info_dict


def find_issues_with_predictions(
    predictions: np.ndarray,
    y: np.ndarray,
    threshold: float,
    **kwargs,
) -> pd.DataFrame:
    """Find label issues in a regression dataset based on predictions.
    This uses a threshold to determine if an example has a label issue
    based on the quality score.

    Parameters
    ----------
    predictions :
        The predictions from a regression model.

    y :
        The given labels.

    threshold :
        The threshold to use to determine if an example has a label issue. It is a multiplier
        of the median label quality score that sets the absolute threshold.

    **kwargs :
        Various keyword arguments.

    Returns
    -------
    issues :
        A dataframe of the issues. It contains the following columns:
        - is_label_issue : bool
            True if the example has a label issue.
        - label_score : float
            The quality score of the label.
        - given_label : float
            The given label. It is the same as the y parameter.
        - predicted_label : float
            The predicted label. It is the same as the predictions parameter.
    """
    _accepted_kwargs = ["method"]
    _kwargs = {k: kwargs.get(k) for k in _accepted_kwargs}
    _kwargs = {k: v for k, v in _kwargs.items() if v is not None}
    quality_scores = get_label_quality_scores(labels=y, predictions=predictions, **_kwargs)

    median_score = np.median(quality_scores)
    is_label_issue_mask = quality_scores < median_score * threshold

    issues = pd.DataFrame(
        {
            "is_label_issue": is_label_issue_mask,
            "label_score": quality_scores,
            "given_label": y,
            "predicted_label": predictions,
        }
    )
    return issues


def find_issues_with_features(
    features: np.ndarray,
    y: np.ndarray,
    cl: CleanLearning,
    **kwargs,
) -> pd.DataFrame:
    """Find label issues in a regression dataset based on features.
    This delegates the work to the CleanLearning.find_label_issues method.

    Parameters
    ----------
    features :
        The numerical features from a regression dataset.

    y :
        The given labels.

    **kwargs :
        Various keyword arguments.

    Returns
    -------
    issues :
        A dataframe of the issues. It contains the following columns:
        - is_label_issue : bool
            True if the example has a label issue.
        - label_score : float
            The quality score of the label.
        - given_label : float
            The given label. It is the same as the y parameter.
        - predicted_label : float
            The predicted label. It is determined by the CleanLearning.find_label_issues method.
    """
    _accepted_kwargs = [
        "uncertainty",
        "coarse_search_range",
        "fine_search_size",
        "save_space",
        "model_kwargs",
    ]
    _kwargs = {k: v for k, v in kwargs.items() if k in _accepted_kwargs and v is not None}
    return cl.find_label_issues(X=features, y=y, **_kwargs)
