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
Module that handles reporting of all types of issues identified in the data.
"""

from typing import TYPE_CHECKING, List

import pandas as pd

from cleanlab.datalab.internal.adapter.constants import DEFAULT_CLEANVISION_ISSUES
from cleanlab.datalab.internal.issue_manager_factory import _IssueManagerFactory

if TYPE_CHECKING:  # pragma: no cover
    from cleanlab.datalab.internal.data_issues import DataIssues


class Reporter:
    """Class that generates a report about the issues stored in a :py:class:`DataIssues` object.

    Parameters
    ----------
    data_issues :
        The :py:class:`DataIssues` object containing the issues to report on. This is usually
        generated by the :py:class:`Datalab` class, stored in the :py:attr:`data_issues` attribute,
        and then passed to the :py:class:`Reporter` class to generate a report.

    task :
        Specific machine learning task like classification or regression.

    verbosity :
        The default verbosity of the report to generate. Each :py:class`IssueManager`
        specifies the available verbosity levels and what additional information
        is included at each level.

    include_description :
        Whether to include the description of each issue type in the report. The description
        is included by default, but can be excluded by setting this parameter to ``False``.

    Note
    ----
    This class is not intended to be used directly. Instead, use the
    `Datalab.find_issues` method which internally utilizes an IssueFinder instance.
    """

    def __init__(
        self,
        data_issues: "DataIssues",
        task: str,
        verbosity: int = 1,
        include_description: bool = True,
        show_summary_score: bool = False,
        show_all_issues: bool = False,
        **kwargs,
    ):
        self.data_issues = data_issues
        self.task = task
        self.verbosity = verbosity
        self.include_description = include_description
        self.show_summary_score = show_summary_score
        self.show_all_issues = show_all_issues

    def _get_empty_report(self) -> str:
        """This method is used to return a report when there are
        no issues found in the data with Datalab.find_issues().
        """
        report_str = "No issues found in the data. Good job!"
        if not self.show_summary_score:
            recommendation_msg = (
                "Try re-running Datalab.report() with "
                "`show_summary_score = True` and `show_all_issues = True`."
            )
            report_str += f"\n\n{recommendation_msg}"
        return report_str

    def report(self, num_examples: int) -> None:
        """Prints a report about identified issues in the data.

        Parameters
        ----------
        num_examples :
            The number of examples to include in the report for each issue type.
        """
        print(self.get_report(num_examples=num_examples))

    def get_report(self, num_examples: int) -> str:
        """Constructs a report about identified issues in the data.

        Parameters
        ----------
        num_examples :
            The number of examples to include in the report for each issue type.


        Returns
        -------
        report_str :
            A string containing the report.

        Examples
        --------
        >>> from cleanlab.datalab.internal.report import Reporter
        >>> reporter = Reporter(data_issues=data_issues, include_description=False)
        >>> report_str = reporter.get_report(num_examples=5)
        >>> print(report_str)
        """
        report_str = ""
        issue_summary = self.data_issues.issue_summary
        should_return_empty_report = not (
            self.show_all_issues or issue_summary.empty or issue_summary["num_issues"].sum() > 0
        )

        if should_return_empty_report:
            return self._get_empty_report()
        issue_summary_sorted = issue_summary.sort_values(by="num_issues", ascending=False)
        report_str += self._write_summary(summary=issue_summary_sorted)

        issue_types = self._get_issue_types(issue_summary_sorted)

        def add_issue_to_report(issue_name: str) -> bool:
            """Returns True if the issue should be added to the report.
            It is excluded if show_all_issues is False and there are no issues of that type
            found in the data.
            """
            if self.show_all_issues:
                return True
            summary = self.data_issues.get_issue_summary(issue_name=issue_name)
            has_issues = summary["num_issues"][0] > 0
            return has_issues

        issue_reports = [
            _IssueManagerFactory.from_str(issue_type=key, task=self.task).report(
                issues=self.data_issues.get_issues(issue_name=key),
                summary=self.data_issues.get_issue_summary(issue_name=key),
                info=self.data_issues.get_info(issue_name=key),
                num_examples=num_examples,
                verbosity=self.verbosity,
                include_description=self.include_description,
            )
            for key in issue_types
            if add_issue_to_report(key)
        ]

        report_str += "\n\n\n".join(issue_reports)
        return report_str

    def _write_summary(self, summary: pd.DataFrame) -> str:
        statistics = self.data_issues.get_info("statistics")
        num_examples = statistics["num_examples"]
        num_classes = statistics.get(
            "num_classes"
        )  # This may not be required for all types of datasets  in the future (e.g. unlabeled/regression)

        dataset_information = f"Dataset Information: num_examples: {num_examples}"
        if num_classes is not None:
            dataset_information += f", num_classes: {num_classes}"

        if self.show_summary_score:
            return (
                "Here is a summary of the different kinds of issues found in the data:\n\n"
                + summary.to_string(index=False)
                + "\n\n"
                + "(Note: A lower score indicates a more severe issue across all examples in the dataset.)\n\n"
                + f"{dataset_information}\n\n\n"
            )

        return (
            "Here is a summary of the different kinds of issues found in the data:\n\n"
            + summary.drop(columns=["score"]).to_string(index=False)
            + "\n\n"
            + f"{dataset_information}\n\n\n"
        )

    def _get_issue_types(self, issue_summary: pd.DataFrame) -> List[str]:
        issue_types = [
            issue_type
            for issue_type, num_issues in zip(
                issue_summary["issue_type"].tolist(), issue_summary["num_issues"].tolist()
            )
            if issue_type not in DEFAULT_CLEANVISION_ISSUES
            and (self.show_all_issues or num_issues > 0)
        ]
        return issue_types
