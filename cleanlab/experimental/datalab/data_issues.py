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
