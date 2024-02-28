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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Type

from cleanlab.datalab.internal.task import Task

if TYPE_CHECKING:  # pragma: no cover
    from cleanlab.datalab.internal.data_issues import DataIssues


class RepresentationStrategy(ABC):
    def __init__(self, data_issues: "DataIssues"):
        self.data_issues = data_issues

    @property
    def checks_run(self) -> bool:
        return not self.data_issues.issues.empty

    @property
    def num_examples(self) -> Optional[int]:
        return self.data_issues.get_info("statistics").get("num_examples")

    @property
    def num_classes(self) -> Optional[int]:
        return self.data_issues.get_info("statistics").get("num_classes")

    @property
    def issues_identified(self) -> str:
        return (
            self.data_issues.issue_summary["num_issues"].sum() if self.checks_run else "Not checked"
        )

    def show_task(self, task: "Task") -> str:
        return f"task={str(task).capitalize()}"

    def show_checks_run(self) -> str:
        return f"checks_run={self.checks_run}"

    def show_num_examples(self) -> str:
        return f"num_examples={self.num_examples}" if self.num_examples is not None else ""

    def show_num_classes(self) -> str:
        return f"num_classes={self.num_classes}" if self.num_classes is not None else ""

    def show_issues_identified(self) -> str:
        return f"issues_identified={self.issues_identified}"

    @abstractmethod
    def represent(self) -> str:
        pass

    def to_string(self, task: "Task") -> str:
        """What is displayed if user executes: print(datalab)"""
        info_list = [
            f"Task: {str(task).capitalize()}",
            f"Checks run: {'Yes' if self.checks_run else 'No'}",
            f"Number of examples: {self.num_examples if self.num_examples is not None else 'Unknown'}",
            f"Number of classes: {self.num_classes if self.num_classes is not None else 'Unknown'}",
            f"Issues identified: {self.issues_identified}",
        ]

        return "Datalab:\n" + "\n".join(info_list)


class ClassificationRepresentation(RepresentationStrategy):
    def represent(self) -> str:
        display_strings: List[str] = [
            self.show_task(Task.CLASSIFICATION),
            self.show_checks_run(),
            self.show_num_examples(),
            self.show_num_classes(),
            self.show_issues_identified(),
        ]
        # Drop empty strings
        display_strings = [s for s in display_strings if bool(s)]
        display_str = ", ".join(display_strings)
        return f"Datalab({display_str})"


class RegressionRepresentation(RepresentationStrategy):
    def represent(self) -> str:
        display_strings: List[str] = [
            self.show_task(Task.REGRESSION),
            self.show_checks_run(),
            self.show_num_examples(),
            self.show_issues_identified(),
        ]
        # Drop empty strings
        display_strings = [s for s in display_strings if bool(s)]
        display_str = ", ".join(display_strings)
        return f"Datalab({display_str})"


class MultilabelRepresentation(RepresentationStrategy):
    def represent(self) -> str:
        display_strings: List[str] = [
            self.show_task(Task.MULTILABEL),
            self.show_checks_run(),
            self.show_num_examples(),
            self.show_num_classes(),
            self.show_issues_identified(),
        ]
        # Drop empty strings
        display_strings = [s for s in display_strings if bool(s)]
        display_str = ", ".join(display_strings)
        return f"Datalab({display_str})"


class _Displayer:
    def __init__(self, data_issues: "DataIssues", task: "Task") -> None:
        self.data_issues = data_issues
        self.task = task
        self.representation_strategy = self._get_representation_strategy()

    def _get_representation_strategy(self) -> RepresentationStrategy:
        strategies: Dict[str, Type[RepresentationStrategy]] = {
            "classification": ClassificationRepresentation,
            "regression": RegressionRepresentation,
            "multilabel": MultilabelRepresentation,
        }
        strategy_class = strategies.get(self.task.value)
        if strategy_class is None:
            raise ValueError(f"Unsupported task type: {self.task}")
        return strategy_class(self.data_issues)

    def __repr__(self) -> str:
        """What is displayed in console if user executes: >>> datalab"""
        return self.representation_strategy.represent()

    def __str__(self) -> str:
        """What is displayed if user executes: print(datalab)"""
        return self.representation_strategy.to_string(self.task)
