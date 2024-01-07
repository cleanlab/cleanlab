import numpy as np
from typing import Tuple, Optional

from cleanlab.datalab.internal.issue_manager import IssueManager


class IdentifierIssueManager(IssueManager):
    """An issue manager that keeps track of issues related to identifier
    columns in tabular datasets."""

    description: str = "Identifies columns with sequential integers."
    issue_name: str = "identifier"
    verbosity_levels = {
        0: ["identifier_column"],
        1: [],
        2: [],
    }

    @staticmethod
    def _identifier_column(features: np.ndarray) -> Tuple[int, Optional[int]]:
        num_rows, num_columns = features.shape

        for i in range(num_columns):
            unique_values = np.unique(features[:, i])

            if np.array_equal(unique_values, np.arange(num_rows)):
                return 0, i  # Found a column that meets the condition

        return 1, None  # No such column found

    def find_issues(self, features: np.ndarray, **_) -> None:
        if features is None:
            raise ValueError("Features must be provided to check for issues")

        score, column_index = self._identifier_column(features)

        self.summary = self.make_summary(score=score)

        self.info = {"identifier_column": [column_index] if column_index is not None else []}
