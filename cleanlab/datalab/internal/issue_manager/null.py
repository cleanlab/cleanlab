from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, List, Union

import numpy as np
import pandas as pd
from numpy import ndarray

from cleanlab.datalab.internal.issue_manager import IssueManager

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from cleanlab.datalab.datalab import Datalab


class NullIssueManager(IssueManager):
    """Manages issues related to null/missing values in the rows of features.
    Parameters
    ----------
    datalab :
        The Datalab instance that this issue manager searches for issues in.
    """

    description: ClassVar[
        str
    ] = """Whether the dataset has any missing/null values
        """
    issue_name: ClassVar[str] = "null"
    verbosity_levels = {
        0: ["average_null_score"],
        1: [
            "most_common_issue",
        ],
        2: [],
    }

    def __init__(self, datalab: Datalab):
        super().__init__(datalab)

    @staticmethod
    def _calculate_null_issues(features: npt.NDArray) -> tuple[ndarray, ndarray, Any]:
        rows = features.shape[0]
        cols = features.shape[1]
        scores = np.zeros(rows).astype(np.float32)
        is_null_issue = np.full(rows, False)
        null_tracker = np.isnan(features)
        if null_tracker.any():
            for row in range(rows):
                if null_tracker[row].any():
                    null_row_count = np.count_nonzero(~null_tracker[row])
                    scores[row] = null_row_count / cols
                    if scores[row] == 0.00:
                        is_null_issue[row] = True
        return is_null_issue, scores, null_tracker

    def find_issues(
        self,
        features: Optional[npt.NDArray] = None,
        **kwargs,
    ) -> None:
        if features is None:
            raise ValueError("features must be provided to check for null values.")
        is_null_issue, scores, null_tracker = self._calculate_null_issues(features=features)

        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": is_null_issue,
                self.issue_score_key: scores,
            },
        )

        self.summary = self.make_summary(score=scores.mean())
        self.info = self.collect_info(null_tracker)

    @staticmethod
    def most_common_issue(
        null_tracker: np.ndarray,
    ) -> Dict[str, List[int], int]:
        """
        Identify and return the most common null value pattern across all rows
        and count the number of rows with this pattern.

        Parameters
        ------------
        null_tracker : np.ndarray
            A boolean array of the same shape as features, where True indicates null/missing entries.

        Returns
        --------
        Dict[str, Any]
            A dictionary containing the most common issue pattern and the count of rows with this pattern.
        """
        # Convert the boolean null_tracker matrix into a list of strings.
        most_frequent_pattern = "no_null"
        rows_affected = []
        occurrence_of_most_frequent_pattern = 0
        if null_tracker.any():
            null_patterns_as_strings = [
                "".join(map(str, row.astype(int).tolist())) for row in null_tracker if row.any()
            ]

            # Use Counter to efficiently count occurrences and find the most common pattern.
            pattern_counter = Counter(null_patterns_as_strings)
            (
                most_frequent_pattern,
                occurrence_of_most_frequent_pattern,
            ) = pattern_counter.most_common(1)[0]
            rows_affected = []
            for idx, row in enumerate(null_patterns_as_strings):
                if row == most_frequent_pattern:
                    rows_affected.append(idx)
        return {
            "most_common_issue": {
                "pattern": most_frequent_pattern,
                "rows_affected": rows_affected,
                "count": occurrence_of_most_frequent_pattern,
            }
        }

    @staticmethod
    def column_impact(null_tracker: np.ndarray) -> Dict[str, List[float]]:
        """
        Calculate and return the impact of null values per column, represented as the proportion
        of rows having null values in each column.

        Parameters
        ----------
        null_tracker : np.ndarray
            A boolean array of the same shape as features, where True indicates null/missing entries.

        Returns
        -------
        Dict[str, List[float]]
            A dictionary containing the impact per column, with values being a list
            where each element is the percentage of rows having null values in the corresponding column.
        """
        # Calculate proportion of nulls in each column
        proportion_of_nulls_per_column = null_tracker.mean(axis=0)

        # Return result as a dictionary containing a list of proportions
        return {"column_impact": proportion_of_nulls_per_column.tolist()}

    def collect_info(self, null_tracker: np.ndarray) -> dict:
        most_common_issue = self.most_common_issue(null_tracker=null_tracker)
        column_impact = self.column_impact(null_tracker=null_tracker)
        average_null_score = {"average_null_score": self.issues[self.issue_score_key].mean()}
        issues_dict = {**average_null_score, **most_common_issue, **column_impact}
        info_dict: Dict[str, Any] = {**issues_dict}
        return info_dict
