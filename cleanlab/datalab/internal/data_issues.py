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
Module for the :py:class:`DataIssues` class, which serves as a central repository for storing
information and statistics about issues found in a dataset.

It collects information from various
:py:class:`IssueManager <cleanlab.datalab.internal.issue_manager.issue_manager.IssueManager>`
instances and keeps track of each issue, a summary for each type of issue,
related information and statistics about the issues.

The collected information can be accessed using the
`~cleanlab.datalab.internal.data_issues.DataIssues.get_info` method.
We recommend using that method instead of this module, which is just intended for internal use.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union
import numpy as np

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from cleanlab.datalab.internal.data import Data
    from cleanlab.datalab.internal.issue_manager import IssueManager
    from cleanvision import Imagelab


class _InfoStrategy(ABC):
    """
    Abstract base class for strategies that fetch information about data issues.

    Subclasses must implement the `get_info` method, which takes a `Data` object, a dictionary of
    information about data issues, and an optional issue name, and returns a dictionary of
    information about the specified issue, augmented with dataset about the dataset as a whole.

    This class also provides a helper method, `_get_info_helper`, which takes an information
    dictionary and an optional issue name, and returns a copy of the information dictionary for
    the specified issue. If the issue name is `None`, this method returns `None`.
    """

    @staticmethod
    @abstractmethod
    def get_info(
        data: Data,
        info: Dict[str, Dict[str, Any]],
        issue_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get information about a data issue from an information dictionary.

        Parameters
        ----------
        info : dict
            A dictionary of information about data issues.
        issue_name : str or None, optional (default=None)
            The name of the issue to get information about. If `None`, this method returns `None`.

        Returns
        -------
        dict or None
            A copy of the information dictionary for the specified issue, or `None` if the issue
            name is `None`.

        Raises
        ------
        ValueError
            If the specified issue name is not found in the information dictionary.
        """
        pass  # pragma: no cover

    @staticmethod
    def _get_info_helper(
        info: Dict[str, Dict[str, Any]],
        issue_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if issue_name is None:
            return None
        if issue_name not in info:
            raise ValueError(
                f"issue_name {issue_name} not found in self.info. These have not been computed yet."
            )
        info = info[issue_name].copy()
        return info


class _ClassificationInfoStrategy(_InfoStrategy):
    """Strategy for computing information about data issues related to classification tasks."""

    @staticmethod
    def get_info(
        data: Data,
        info: Dict[str, Dict[str, Any]],
        issue_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        info_extracted = _InfoStrategy._get_info_helper(info=info, issue_name=issue_name)
        info = info_extracted if info_extracted is not None else info
        if issue_name in ["label", "class_imbalance"]:
            if data.labels.is_available is False:
                raise ValueError(
                    "The labels are not available. "
                    "Most likely, no label column was provided when creating the Data object."
                )
            # Labels that are stored as integers may need to be converted to strings.
            label_map = data.labels.label_map
            if not label_map:
                raise ValueError("The label map is not available.")
            for key in ["given_label", "predicted_label"]:
                labels = info.get(key, None)
                if labels is not None:
                    info[key] = np.vectorize(label_map.get)(labels)
            info["class_names"] = list(label_map.values())
        return info


class _RegressionInfoStrategy(_InfoStrategy):
    """Strategy for computing information about data issues related to regression tasks."""

    @staticmethod
    def get_info(
        data: Data,
        info: Dict[str, Dict[str, Any]],
        issue_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        info_extracted = _InfoStrategy._get_info_helper(info=info, issue_name=issue_name)
        info = info_extracted if info_extracted is not None else info
        if issue_name == "label":
            for key in ["given_label", "predicted_label"]:
                labels = info.get(key, None)
                if labels is not None:
                    info[key] = labels
        return info


class _MultilabelInfoStrategy(_InfoStrategy):
    """Strategy for computing information about data issues related to multilabel tasks."""

    @staticmethod
    def get_info(
        data: Data,
        info: Dict[str, Dict[str, Any]],
        issue_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        info_extracted = _InfoStrategy._get_info_helper(info=info, issue_name=issue_name)
        info = info_extracted if info_extracted is not None else info
        if issue_name == "label":
            if data.labels.is_available is False:
                raise ValueError(
                    "The labels are not available. "
                    "Most likely, no label column was provided when creating the Data object."
                )
            # Labels that are stored as integers may need to be converted to strings.
            label_map = data.labels.label_map
            if not label_map:
                raise ValueError("The label map is not available.")
            for key in ["given_label", "predicted_label"]:
                labels = info.get(key, None)
                if labels is not None:
                    info[key] = [list(map(label_map.get, label)) for label in labels]
            info["class_names"] = list(label_map.values())
        return info


class DataIssues:
    """
    Class that collects and stores information and statistics on issues found in a dataset.

    Parameters
    ----------
    data :
        The data object for which the issues are being collected.
    strategy :
        Strategy used for processing info dictionaries.

    Attributes
    ----------
    issues : pd.DataFrame
        Stores information about each individual issue found in the data,
        on a per-example basis.
    issue_summary : pd.DataFrame
        Summarizes the overall statistics for each issue type.
    info : dict
        A dictionary that contains information and statistics about the data and each issue type.
    """

    def __init__(self, data: Data, strategy: Type[_InfoStrategy]) -> None:
        self.issues: pd.DataFrame = pd.DataFrame(index=range(len(data)))
        self.issue_summary: pd.DataFrame = pd.DataFrame(
            columns=["issue_type", "score", "num_issues"]
        ).astype({"score": np.float64, "num_issues": np.int64})
        self.info: Dict[str, Dict[str, Any]] = {
            "statistics": get_data_statistics(data),
        }
        self._data = data
        self._strategy = strategy

    def get_info(self, issue_name: Optional[str] = None) -> Dict[str, Any]:
        return self._strategy.get_info(data=self._data, info=self.info, issue_name=issue_name)

    @property
    def statistics(self) -> Dict[str, Any]:
        """Returns the statistics dictionary.

        Shorthand for self.info["statistics"].
        """
        return self.info["statistics"]

    def get_issues(self, issue_name: Optional[str] = None) -> pd.DataFrame:
        """
        Use this after finding issues to see which examples suffer from which types of issues.

        Parameters
        ----------
        issue_name : str or None
            The type of issue to focus on. If `None`, returns full DataFrame summarizing all of the types of issues detected in each example from the dataset.

        Raises
        ------
        ValueError
            If `issue_name` is not a type of issue previously considered in the audit.

        Returns
        -------
        specific_issues :
            A DataFrame where each row corresponds to an example from the dataset and columns specify:
            whether this example exhibits a particular type of issue and how severely (via a numeric quality score where lower values indicate more severe instances of the issue).

            Additional columns may be present in the DataFrame depending on the type of issue specified.
        """
        if issue_name is None:
            return self.issues

        columns = [col for col in self.issues.columns if issue_name in col]
        if not columns:
            raise ValueError(f"No columns found for issue type '{issue_name}'.")
        specific_issues = self.issues[columns]
        info = self.get_info(issue_name=issue_name)

        if issue_name == "label":
            specific_issues = specific_issues.assign(
                given_label=info["given_label"], predicted_label=info["predicted_label"]
            )

        if issue_name == "near_duplicate":
            column_dict = {
                k: info.get(k)
                for k in ["near_duplicate_sets", "distance_to_nearest_neighbor"]
                if info.get(k) is not None
            }
            specific_issues = specific_issues.assign(**column_dict)

        if issue_name == "class_imbalance":
            specific_issues = specific_issues.assign(given_label=info["given_label"])
        return specific_issues

    def get_issue_summary(self, issue_name: Optional[str] = None) -> pd.DataFrame:
        """Summarize the issues found in dataset of a particular type,
        including how severe this type of issue is overall across the dataset.

        Parameters
        ----------
        issue_name :
            Name of the issue type to summarize. If `None`, summarizes each of the different issue types previously considered in the audit.

        Returns
        -------
        issue_summary :
            DataFrame where each row corresponds to a type of issue, and columns quantify:
            the number of examples in the dataset estimated to exhibit this type of issue,
            and the overall severity of the issue across the dataset (via a numeric quality score where lower values indicate that the issue is overall more severe).
        """
        if self.issue_summary.empty:
            raise ValueError(
                "No issues found in the dataset. "
                "Call `find_issues` before calling `get_issue_summary`."
            )

        if issue_name is None:
            return self.issue_summary

        row_mask = self.issue_summary["issue_type"] == issue_name
        if not any(row_mask):
            raise ValueError(f"Issue type {issue_name} not found in the summary.")
        return self.issue_summary[row_mask].reset_index(drop=True)

    def collect_statistics(self, issue_manager: Union[IssueManager, "Imagelab"]) -> None:
        """Update the statistics in the info dictionary.

        Parameters
        ----------
        statistics :
            A dictionary of statistics to add/update in the info dictionary.

        Examples
        --------

        A common use case is to reuse the KNN-graph across multiple issue managers.
        To avoid recomputing the KNN-graph for each issue manager,
        we can pass it as a statistic to the issue managers.

        >>> from scipy.sparse import csr_matrix
        >>> weighted_knn_graph = csr_matrix(...)
        >>> issue_manager_that_computes_knn_graph = ...

        """
        key = "statistics"
        statistics: Dict[str, Any] = issue_manager.info.get(key, {})
        if statistics:
            self.info[key].update(statistics)

    def _update_issues(self, issue_manager):
        overlapping_columns = list(set(self.issues.columns) & set(issue_manager.issues.columns))
        if overlapping_columns:
            warnings.warn(
                f"Overwriting columns {overlapping_columns} in self.issues with "
                f"columns from issue manager {issue_manager}."
            )
            self.issues.drop(columns=overlapping_columns, inplace=True)
        self.issues = self.issues.join(issue_manager.issues, how="outer")

    def _update_issue_info(self, issue_name, new_info):
        if issue_name in self.info:
            warnings.warn(f"Overwriting key {issue_name} in self.info")
        self.info[issue_name] = new_info

    def collect_issues_from_issue_manager(self, issue_manager: IssueManager) -> None:
        """
        Collects results from an IssueManager and update the corresponding
        attributes of the Datalab object.

        This includes:
        - self.issues
        - self.issue_summary
        - self.info

        Parameters
        ----------
        issue_manager :
            IssueManager object to collect results from.
        """
        self._update_issues(issue_manager)

        if issue_manager.issue_name in self.issue_summary["issue_type"].values:
            warnings.warn(
                f"Overwriting row in self.issue_summary with "
                f"row from issue manager {issue_manager}."
            )
            self.issue_summary = self.issue_summary[
                self.issue_summary["issue_type"] != issue_manager.issue_name
            ]
        issue_column_name: str = f"is_{issue_manager.issue_name}_issue"
        num_issues: int = int(issue_manager.issues[issue_column_name].sum())
        self.issue_summary = pd.concat(
            [
                self.issue_summary,
                issue_manager.summary.assign(num_issues=num_issues),
            ],
            axis=0,
            ignore_index=True,
        )
        self._update_issue_info(issue_manager.issue_name, issue_manager.info)

    def collect_issues_from_imagelab(self, imagelab: "Imagelab", issue_types: List[str]) -> None:
        pass  # pragma: no cover

    def set_health_score(self) -> None:
        """Set the health score for the dataset based on the issue summary.

        Currently, the health score is the mean of the scores for each issue type.
        """
        self.info["statistics"]["health_score"] = self.issue_summary["score"].mean()


def get_data_statistics(data: Data) -> Dict[str, Any]:
    """Get statistics about a dataset.

    This function is called to initialize the "statistics" info in all `Datalab` objects.

    Parameters
    ----------
    data : Data
        Data object containing the dataset.
    """
    statistics: Dict[str, Any] = {
        "num_examples": len(data),
        "multi_label": False,
        "health_score": None,
    }
    if data.labels.is_available:
        class_names = data.class_names
        statistics["class_names"] = class_names
        statistics["num_classes"] = len(class_names)
    return statistics
