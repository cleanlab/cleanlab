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
from typing import Any
from cleanlab.experimental.datalab.data import Data
import pandas as pd


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
