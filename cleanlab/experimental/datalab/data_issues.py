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
:py:class:`IssueManager <cleanlab.experimental.datalab.issue_manager.IssueManager>`
instances and keeps track of each issue, a summary each type of issue,
related information and statistics about the issues.

The collected information can be accessed using the 
:py:meth:`get_info <cleanlab.experimental.datalab.data_issues.DataIssues.get_info>` method.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from cleanlab.experimental.datalab.data import Data
    from cleanlab.experimental.datalab.issue_manager import IssueManager


class DataIssues:
    """
    Class that collects and stores information and statistics on issues found in a dataset.

    Parameters
    ----------
    data :
        The data object for which the issues are being collected.

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

    def __init__(self, data: Data) -> None:
        self.issues: pd.DataFrame = pd.DataFrame(index=range(len(data)))
        self.issue_summary: pd.DataFrame = pd.DataFrame(columns=["issue_type", "score"])
        class_names = data.class_names
        self.info: Dict[str, Dict[str, Any]] = {
            "data": {
                "num_examples": len(data),
                "class_names": class_names,
                "num_classes": len(class_names),
                "multi_label": False,  # TODO: Add multi-label support.
                "health_score": None,
            },
            "statistics": {},
        }

    def get_info(self, issue_name: str) -> Dict[str, Any]:
        """Get the info for the issue_name key (and any subkeys, if provided).

        This function is used to get the info for a specific issue_name. If the info is not computed yet, it will raise an error.
        If subkeys are provided, it will get the info for the subkeys.

        Parameters
        ----------
        issue_name : str
            The issue name for which the info is required.
        subkeys : str
            If the info is a dictionary, then you can provide the subkeys to get the info for nested dictionaries.

        Returns
        -------
        info:
            The info for the issue_name and subkeys.
        """
        info = self.info.get(issue_name, None)
        if info is None:
            raise ValueError(
                f"issue_name {issue_name} not found in self.info. These have not been computed yet."
            )
        return info

    def _collect_results_from_issue_manager(self, issue_manager: IssueManager) -> None:
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
        overlapping_columns = list(set(self.issues.columns) & set(issue_manager.issues.columns))
        if overlapping_columns:
            warnings.warn(
                f"Overwriting columns {overlapping_columns} in self.issues with "
                f"columns from issue manager {issue_manager}."
            )
            self.issues.drop(columns=overlapping_columns, inplace=True)
        self.issues = self.issues.join(issue_manager.issues, how="outer")

        if issue_manager.issue_name in self.issue_summary["issue_type"].values:
            warnings.warn(
                f"Overwriting row in self.issue_summary with "
                f"row from issue manager {issue_manager}."
            )
            self.issue_summary = self.issue_summary[
                self.issue_summary["issue_type"] != issue_manager.issue_name
            ]
        self.issue_summary = pd.concat(
            [self.issue_summary, issue_manager.summary],
            axis=0,
            ignore_index=True,
        )

        if issue_manager.issue_name in self.info:
            warnings.warn(
                f"Overwriting key {issue_manager.issue_name} in self.info with "
                f"key from issue manager {issue_manager}."
            )
        self.info[issue_manager.issue_name] = issue_manager.info
