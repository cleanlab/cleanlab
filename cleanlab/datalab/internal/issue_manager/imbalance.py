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

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
from cleanlab.datalab.internal.issue_manager import IssueManager

if TYPE_CHECKING:  # pragma: no cover
    from cleanlab.datalab.datalab import Datalab


class ClassImbalanceIssueManager(IssueManager):
    """Manages issues related to imbalance class examples.

    Parameters
    ----------
    datalab:
        The Datalab instance that this issue manager searches for issues in.

    threshold:
        Minimum fraction of samples of each class that are present in a dataset without class imbalance.

    """

    description: ClassVar[
        str
    ] = """Examples belonging to the most under-represented class in the dataset."""

    issue_name: ClassVar[str] = "class_imbalance"
    verbosity_levels = {
        0: [],
        1: [],
        2: [],
    }

    def __init__(self, datalab: Datalab, threshold: float = 0.1):
        super().__init__(datalab)
        self.threshold = threshold

    def find_issues(
        self,
        **kwargs,
    ) -> None:
        labels = self.datalab.labels
        K = len(self.datalab.class_names)
        class_probs = np.bincount(labels) / len(labels)
        rarest_class_idx = int(np.argmin(class_probs))
        imbalance_exists = class_probs[rarest_class_idx] < self.threshold * (1 / K)
        rarest_class = rarest_class_idx if imbalance_exists else -1
        is_issue_column = labels == rarest_class
        scores = np.where(is_issue_column, class_probs[rarest_class], 1)

        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": is_issue_column,
                self.issue_score_key: scores,
            },
        )
        self.summary = self.make_summary(score=class_probs[rarest_class_idx])
        self.info = self.collect_info()

    def collect_info(self) -> dict:
        params_dict = {"threshold": self.threshold}
        info_dict = {**params_dict}
        return info_dict
