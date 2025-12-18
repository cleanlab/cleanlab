"""
This module contains the Task enum, which internally represents the tasks
supported by Datalab, so that the appropriate task-specific logic can be applied.
This class and associated naming conventions are subject to change and is not meant
to be used by users.
"""

from enum import Enum


class Task(Enum):
    """
    Represents a task supported by Datalab.

    Datalab supports the following tasks:

    * **Classification**: for predicting discrete class labels.
    * **Regression**: for predicting continuous numerical values.
    * **Multilabel**: for predicting multiple binary labels simultaneously.

    Example
    -------
    >>> task = Task.CLASSIFICATION
    >>> task
    <Task.CLASSIFICATION: 'classification'>
    """

    CLASSIFICATION = "classification"
    """Classification task."""
    REGRESSION = "regression"
    """Regression task."""
    MULTILABEL = "multilabel"
    """Multilabel task."""

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

        Parameters
        ----------
        task_str :
            The string representation of the task.

        Returns
        -------
        Task :
            The corresponding Task enum value.

        Raises
        ------
        ValueError :
            If the provided task_str is not a valid task supported by Datalab.

        Examples
        --------
        >>> Task.from_str("classification")
        <Task.CLASSIFICATION: 'classification'>
        >>> print(Task.from_str("regression"))
        regression
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

        Returns
        -------
        bool :
            True if the task is classification, False otherwise.

        Examples
        --------
        >>> task = Task.CLASSIFICATION
        >>> print(task.is_classification)
        True
        """
        return self == Task.CLASSIFICATION

    @property
    def is_regression(self):
        """
        Checks if the task is regression.

        Returns
        -------
        bool :
            True if the task is regression, False otherwise.

        Examples
        --------
        >>> task = Task.CLASSIFICATION
        >>> print(task.is_regression)
        False
        """
        return self == Task.REGRESSION

    @property
    def is_multilabel(self):
        """
        Checks if the task is multilabel.

        Returns
        -------
        bool :
            True if the task is multilabel, False otherwise.

        Examples
        --------
        >>> task = Task.CLASSIFICATION
        >>> print(task.is_multilabel)
        False
        """
        return self == Task.MULTILABEL
