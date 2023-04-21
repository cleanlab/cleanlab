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
Module that handles the string representation of Datalab objects.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cleanlab.datalab.datalab import Datalab  # pragma: no cover


class _Displayer:
    def __init__(self, datalab: "Datalab") -> None:
        self.datalab = datalab

    def __repr__(self) -> str:
        """What is displayed in console if user executes: >>> datalab"""
        checks_run = not self.datalab.issues.empty
        display_str = f"checks_run={checks_run}"
        num_examples = self.datalab.get_info("statistics")["num_examples"]
        if num_examples is not None:
            display_str += f", num_examples={num_examples}"
        num_classes = self.datalab.get_info("statistics")["num_classes"]
        if num_classes is not None:
            display_str += f", num_classes={num_classes}"
        if checks_run:
            issues_identified = self.datalab.issue_summary["num_issues"].sum()
            display_str += f", issues_identified={issues_identified}"
        return f"Datalab({display_str})"

    def __str__(self) -> str:
        """What is displayed if user executes: print(datalab)"""
        checks_run = not self.datalab.issues.empty
        num_examples = self.datalab.get_info("statistics").get("num_examples")
        num_classes = self.datalab.get_info("statistics").get("num_classes")

        issues_identified = (
            self.datalab.issue_summary["num_issues"].sum() if checks_run else "Not checked"
        )
        info_list = [
            f"Checks run: {'Yes' if checks_run else 'No'}",
            f"Number of examples: {num_examples if num_examples is not None else 'Unknown'}",
            f"Number of classes: {num_classes if num_classes is not None else 'Unknown'}",
            f"Issues identified: {issues_identified}",
        ]

        return "Datalab:\n" + "\n".join(info_list)
