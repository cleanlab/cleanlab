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

from cleanlab.regression.learn import CleanLearning
from cleanlab.datalab.internal.issue_manager import IssueManager

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from cleanlab.datalab.datalab import Datalab


class RegressionLabelIssueManager(IssueManager):
    """Manages label issues in a Datalab.

    Parameters
    ----------
    datalab :
        A Datalab instance.

    clean_learning_kwargs :
        Keyword arguments to pass to the :py:meth:`CleanLearning <cleanlab.regression.learn.CleanLearning>` constructor.

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
        health_summary_parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(datalab)
        self.cl = CleanLearning(**(clean_learning_kwargs or {}))

    @staticmethod
    def _process_find_label_issues_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Searches for keyword arguments that are meant for the
        CleanLearning.find_label_issues method call

        Examples
        --------
        >>> from cleanlab.datalab.internal.issue_manager.regression.label import LabelIssueManager
        >>> RegressionLabelIssueManager._process_find_label_issues_kwargs({'coarse_search_range': [0.1, 0.9]})
        {'coarse_search_range': [0.1, 0.9]}
        """
        accepted_kwargs = [
            "uncertainty",
            "coarse_search_range",
            "fine_search_size",
            "save_space",
            "model_kwargs",
        ]
        return {k: v for k, v in kwargs.items() if k in accepted_kwargs and v is not None}

    def find_issues(
        self,
        **kwargs,
    ) -> None:
        """Find label issues in the datalab."""

        # Find examples with label issues
        X_with_y = self.datalab.data.to_pandas()
        X = X_with_y.drop(columns=self.datalab.label_name)
        y = X_with_y[self.datalab.label_name]
        self.issues = self.cl.find_label_issues(
            X=X,
            y=y,
            **self._process_find_label_issues_kwargs(kwargs),
        )
        self.issues.rename(columns={"label_quality": self.issue_score_key}, inplace=True)

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
