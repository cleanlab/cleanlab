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

from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd
from cleanlab.datalab.issue_manager import IssueManager

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from cleanlab.datalab.datalab import Datalab


class ClassImbalanceIssueManager(IssueManager):
    """Manages issues related to imbalance class examples."""

    description: ClassVar[
        str
    ] = """Examples belonging to the most under-represented class in the dataset.

    Parameters
    ----------
    datalab :
        The Datalab instance that this issue manager searches for issues in.

    fraction:
        Minimum fraction of samples of each class that are present in a dataset without class imbalance.
    """
    issue_name: ClassVar[str] = "class_imbalance"
    verbosity_levels = {
        0: [],
        1: [],
        2: ["threshold"],
    }

    def __init__(
        self,
        datalab: Datalab,
        fraction: float = 0.1,
        **kwargs,
    ):
        super().__init__(datalab)
        self.fraction = fraction

    def find_issues(
        self,
        **kwargs,
    ) -> None:

        labels = self.datalab.labels
        K = len(self.datalab.class_names)
        class_probs = np.bincount(labels) / len(labels)
        imbalance_exists = class_probs.min() < self.fraction * (1 / K)
        is_issue_column = np.full(len(labels), False)
        scores = np.ones(len(labels))
        if imbalance_exists:
            rarest_class = np.argmin(class_probs)
            is_issue_column[labels == rarest_class] = True
            scores[labels == rarest_class] = class_probs[rarest_class]

        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": is_issue_column,
                self.issue_score_key: scores,
            },
        )
        self.summary = self.make_summary(score=scores.mean())
