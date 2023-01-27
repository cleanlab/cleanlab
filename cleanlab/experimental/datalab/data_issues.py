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
Module for class that collects and stores information and statistics on
issues found in the data.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING
import pandas as pd
import warnings

if TYPE_CHECKING:  # pragma: no cover
    from cleanlab.experimental.datalab.data import Data
    from cleanlab.experimental.datalab.issue_manager import IssueManager


class DataIssues:
    def __init__(self, data: Data) -> None:
        self.issues: pd.DataFrame = pd.DataFrame(index=range(len(data)))
        self.issue_summary: pd.DataFrame = pd.DataFrame(columns=["issue_type", "score"])
        class_names = data.class_names
        self.info = {
            "data": {
                "num_examples": len(data),
                "class_names": class_names,
                "num_classes": len(class_names),
                "multi_label": False,  # TODO: Add multi-label support.
                "health_score": None,
            },
            "statistics": {},
        }

    def get_info(self, issue_name: str, *subkeys: str) -> Any:
        if issue_name in self.info:
            info = self.info[issue_name]
            if subkeys:
                for sub_id, subkey in enumerate(subkeys):
                    if not isinstance(info, dict):
                        raise ValueError(
                            f"subkey {subkey} at index {sub_id} is not a valid key in info dict."
                            f"info is {info} and remaining subkeys are {subkeys[sub_id:]}."
                        )
                    sub_info = info.get(subkey)
                    info = sub_info
            return info
        else:
            raise ValueError(
                f"issue_name {issue_name} not found in self.info. These have not been computed yet."
            )
            # could alternatively consider:
            # raise ValueError("issue_name must be a valid key in Datalab.info dict.")

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
