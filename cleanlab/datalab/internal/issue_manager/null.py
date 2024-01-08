from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, List

import numpy as np
import pandas as pd
from numpy import ndarray

from cleanlab.datalab.internal.issue_manager import IssueManager

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt


class NullIssueManager(IssueManager):
    """Manages issues related to null/missing values in the rows of features.

    Parameters
    ----------
    datalab :
        The Datalab instance that this issue manager searches for issues in.
    """

    description: ClassVar[
        str
    ] = """Examples identified with the null issue correspond to rows that have null/missing values across all feature columns (i.e. the entire row is missing values).
        """
    issue_name: ClassVar[str] = "null"
    verbosity_levels = {
        0: [],
        1: [],
        2: ["most_common_issue"],
    }

    @staticmethod
    def _calculate_null_issues(features: npt.NDArray) -> tuple[ndarray, ndarray, Any]:
        """Tracks the number of null values in each row of a feature array,
        computes quality scores based on the fraction of null values in each row,
        and returns a boolean array indicating whether each row only has null values."""
        cols = features.shape[1]
        null_tracker = np.isnan(features)
        non_null_count = cols - null_tracker.sum(axis=1)
        scores = non_null_count / cols
        is_null_issue = non_null_count == 0
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
    def _most_common_issue(
        null_tracker: np.ndarray,
    ) -> dict[str, dict[str, str | int | list[int] | list[int | None]]]:
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
        rows_affected: List[int] = []
        occurrence_of_most_frequent_pattern = 0
        if null_tracker.any():
            null_row_indices = np.where(null_tracker.any(axis=1))[0]
            null_patterns_as_strings = [
                "".join(map(str, null_tracker[i].astype(int).tolist())) for i in null_row_indices
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
                    rows_affected.append(int(null_row_indices[idx]))
        return {
            "most_common_issue": {
                "pattern": most_frequent_pattern,
                "rows_affected": rows_affected,
                "count": occurrence_of_most_frequent_pattern,
            }
        }

    @staticmethod
    def _column_impact(null_tracker: np.ndarray) -> Dict[str, List[float]]:
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
        most_common_issue = self._most_common_issue(null_tracker=null_tracker)
        column_impact = self._column_impact(null_tracker=null_tracker)
        average_null_score = {"average_null_score": self.issues[self.issue_score_key].mean()}
        issues_dict = {**average_null_score, **most_common_issue, **column_impact}
        info_dict: Dict[str, Any] = {**issues_dict}
        return info_dict

    @classmethod
    def report(cls, *args, **kwargs) -> str:
        """
        Return a report of issues found by the NullIssueManager.

        This method extends the superclass method by identifying and reporting
        specific issues related to null values in the dataset.

        Parameters
        ----------
        *args : list
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Returns
        -------
        report_str :
            A string containing the report.

        See Also
        --------
        :meth:`cleanlab.datalab.Datalab.report`

        Notes
        -----
        This method differs from other IssueManager report methods. It checks for issues
        and prompts the user to address them to enable other issue managers to run effectively.
        """
        # Generate the base report using the superclass method
        original_report = super().report(*args, **kwargs)

        # Retrieve the 'issues' dataframe from keyword arguments
        issues = kwargs["issues"]

        # Identify examples that have null values in all features
        issue_filter = f"is_{cls.issue_name}_issue"
        examples_with_full_nulls = issues.query(issue_filter).index.tolist()

        # Identify examples that have some null values (but not in all features)
        partial_null_filter = f"{cls.issue_score_key} < 1.0 and not {issue_filter}"
        examples_with_partial_nulls = issues.query(partial_null_filter).index.tolist()

        # Append information about examples with null values in all features
        if examples_with_full_nulls:
            report_addition = (
                f"\n\nFound {len(examples_with_full_nulls)} examples with null values in all features. "
                f"These examples should be removed from the dataset before running other issue managers."
                # TODO: Add a link to the documentation on how to handle null examples
            )
            original_report += report_addition

        # Append information about examples with some null values
        if examples_with_partial_nulls:
            report_addition = (
                f"\n\nFound {len(examples_with_partial_nulls)} examples with null values in some features. "
                f"Please address these issues before running other issue managers."
                # TODO: Add a link to the documentation on how to handle partially null examples
            )
            original_report += report_addition

        return original_report
