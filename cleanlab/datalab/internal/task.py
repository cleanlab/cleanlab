# Copyright (C) 2017-2024  Cleanlab Inc.
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
from enum import Enum


class Task(Enum):
    """
    Represents a task supported by Datalab.

    Datalab supports the following tasks:
    - Classification: for predicting discrete class labels.
    - Regression: for predicting continuous numerical values.
    - Multilabel: for predicting multiple binary labels simultaneously.

    Usage:
    >>> task = Task.CLASSIFICATION
    >>> print(task)
    classification

    >>> task = Task.from_str("regression")
    >>> print(task)
    regression

    >>> print(task.is_classification)
    False

    >>> print(task.is_regression)
    True

    >>> print(task.is_multilabel)
    False
    """

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTILABEL = "multilabel"

    def __str__(self):
        """
        Returns the string representation of the task.

        Returns:
            str: The string representation of the task.
        """
        return self.value

    @classmethod
    def from_str(cls, task_str: str) -> "Task":
        """
        Converts a string representation of a task to a Task enum value.

        Args:
            task_str (str): The string representation of the task.

        Returns:
            Task: The corresponding Task enum value.

        Raises:
            ValueError: If the provided task_str is not a valid task supported by Datalab.
        """
        _value_to_enum = {task.value: task for task in Task}
        try:
            return _value_to_enum[task_str]
        except KeyError:
            valid_tasks = list(_value_to_enum.keys())
            raise ValueError(f"Invalid task: {task_str}. Datalab only supports {valid_tasks}.")

    @property
    def is_classification(self):
        """
        Checks if the task is classification.

        Returns:
            bool: True if the task is classification, False otherwise.
        """
        return self == Task.CLASSIFICATION

    @property
    def is_regression(self):
        """
        Checks if the task is regression.

        Returns:
            bool: True if the task is regression, False otherwise.
        """
        return self == Task.REGRESSION

    @property
    def is_multilabel(self):
        """
        Checks if the task is multilabel.

        Returns:
            bool: True if the task is multilabel, False otherwise.
        """
        return self == Task.MULTILABEL
