from enum import Enum


class Task(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTILABEL = "multilabel"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, task_str: str) -> "Task":
        _value_to_enum = {task.value: task for task in Task}
        try:
            return _value_to_enum[task_str]
        except KeyError:
            valid_tasks = list(_value_to_enum.keys())
            raise ValueError(f"Invalid task: {task_str}. Datalab only supports {valid_tasks}.")

    @property
    def is_classification(self):
        return self == Task.CLASSIFICATION

    @property
    def is_regression(self):
        return self == Task.REGRESSION

    @property
    def is_multilabel(self):
        return self == Task.MULTILABEL
