"""Classes and methods for datasets that are loaded into Datalab."""

import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Union, cast, TYPE_CHECKING, Tuple

from cleanlab.datalab.internal.task import Task

try:
    import datasets
except ImportError as error:
    raise ImportError(
        "Cannot import datasets package. "
        "Please install it and try again, or just install cleanlab with "
        "all optional dependencies via: `pip install 'cleanlab[all]'`"
    ) from error
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from datasets.arrow_dataset import Dataset
from datasets import ClassLabel

# Import Column types for compatibility with datasets 4.0.0+
try:
    from datasets.arrow_dataset import Column
    from datasets.iterable_dataset import IterableColumn
except ImportError:
    # For backwards compatibility with older datasets versions
    Column = None
    IterableColumn = None

from cleanlab.internal.validation import labels_to_array, labels_to_list_multilabel


if TYPE_CHECKING:  # pragma: no cover
    DatasetLike = Union[Dataset, pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], str]


class DataFormatError(ValueError):
    """Exception raised when the data is not in a supported format."""

    def __init__(self, data: Any):
        self.data = data
        message = (
            f"Unsupported data type: {type(data)}\n"
            "Supported types: "
            "datasets.Dataset, pandas.DataFrame, dict, list, str"
        )
        super().__init__(message)


class DatasetDictError(ValueError):
    """Exception raised when a DatasetDict is passed to Datalab.

    Usually, this means that a dataset identifier was passed to Datalab, but
    the dataset is a DatasetDict, which contains multiple splits of the dataset.

    """

    def __init__(self):
        message = (
            "Please pass a single dataset, not a DatasetDict. "
            "Try specifying a split, e.g. `dataset = load_dataset('dataset', split='train')` "
            "then pass `dataset` to Datalab."
        )
        super().__init__(message)


class DatasetLoadError(ValueError):
    """Exception raised when a dataset cannot be loaded.

    Parameters
    ----------
    dataset_type: type
        The type of dataset that failed to load.
    """

    def __init__(self, dataset_type: type):
        message = f"Failed to load dataset from {dataset_type}.\n"
        super().__init__(message)


class Data:
    """
    Class that holds and validates datasets for Datalab.

    Internally, the data is stored as a datasets.Dataset object and the labels
    are integers (ranging from 0 to K-1, where K is the number of classes) stored
    in a numpy array.

    Parameters
    ----------
    data :
        Dataset to be audited by Datalab.
        Several formats are supported, which will internally be converted to a Dataset object.

        Supported formats:
            - datasets.Dataset
            - pandas.DataFrame
            - dict
                - keys are strings
                - values are arrays or lists of equal length
            - list
                - list of dictionaries with the same keys
            - str
                - path to a local file
                    - Text (.txt)
                    - CSV (.csv)
                    - JSON (.json)
                - or a dataset identifier on the Hugging Face Hub
            It checks if the string is a path to a file that exists locally, and if not,
            it assumes it is a dataset identifier on the Hugging Face Hub.

    label_name : Union[str, List[str]]
        Name of the label column in the dataset.

    task :
        The task associated with the dataset. This is used to determine how to
        to format the labels.

        Note:

          - If the task is a classification task, the labels
          will be mapped to integers, e.g. [0, 1, ..., K-1] where K is the number
          of classes. If the task is a regression task, the labels will not be
          mapped to integers.

          - If the task is a multilabel task, the labels will be formatted as a
            list of lists, e.g. [[0, 1], [1, 2], [0, 2]] where each sublist contains
            the labels for a single example. If the task is not a multilabel task,
            the labels will be formatted as a 1D numpy array.

    Warnings
    --------
    Optional dependencies:

    - datasets :
        Dataset, DatasetDict and load_dataset are imported from datasets.
        This is an optional dependency of cleanlab, but is required for
        :py:class:`Datalab <cleanlab.datalab.datalab.Datalab>` to work.
    """

    def __init__(
        self,
        data: "DatasetLike",
        task: Task,
        label_name: Optional[str] = None,
    ) -> None:
        self._validate_data(data)
        self._data = self._load_data(data)
        self._validate_labels(self._data, label_name)
        self._data_hash = hash(self._data)
        self.labels: Label
        label_class = MultiLabel if task.is_multilabel else MultiClass
        map_to_int = task.is_classification
        self.labels = label_class(data=self._data, label_name=label_name, map_to_int=map_to_int)

    def _load_data(self, data: "DatasetLike") -> Dataset:
        """Checks the type of dataset and uses the correct loader method and
        assigns the result to the data attribute."""
        dataset_factory_map: Dict[type, Callable[..., Dataset]] = {
            Dataset: lambda x: x,
            pd.DataFrame: Dataset.from_pandas,
            dict: self._load_dataset_from_dict,
            list: self._load_dataset_from_list,
            str: self._load_dataset_from_string,
        }
        if not isinstance(data, tuple(dataset_factory_map.keys())):
            raise DataFormatError(data)
        return dataset_factory_map[type(data)](data)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> bool:
        if isinstance(other, Data):
            # Equality checks
            hashes_are_equal = self._data_hash == other._data_hash
            labels_are_equal = self.labels == other.labels
            return all([hashes_are_equal, labels_are_equal])
        return False

    def __hash__(self) -> int:
        return self._data_hash

    @property
    def class_names(self) -> List[str]:
        return self.labels.class_names

    @property
    def has_labels(self) -> bool:
        """Check if labels are available."""
        return self.labels.is_available

    @staticmethod
    def _validate_data(data) -> None:
        if isinstance(data, datasets.DatasetDict):
            raise DatasetDictError()
        if not isinstance(data, (Dataset, pd.DataFrame, dict, list, str)):
            raise DataFormatError(data)

    @staticmethod
    def _validate_labels(data: Dataset, label_name: Optional[str]) -> None:
        if label_name is None or label_name not in data.column_names:
            return
        labels = data[label_name]
        if pd.isna(labels).any():
            raise ValueError(
                f"Label column '{label_name}' contains null or NaN values. "
                "Datalab does not support missing labels during initialization."
            )

    @staticmethod
    def _load_dataset_from_dict(data_dict: Dict[str, Any]) -> Dataset:
        try:
            return Dataset.from_dict(data_dict)
        except Exception as error:
            raise DatasetLoadError(dict) from error

    @staticmethod
    def _load_dataset_from_list(data_list: List[Dict[str, Any]]) -> Dataset:
        try:
            return Dataset.from_list(data_list)
        except Exception as error:
            raise DatasetLoadError(list) from error

    @staticmethod
    def _load_dataset_from_string(data_string: str) -> Dataset:
        if not os.path.exists(data_string):
            try:
                dataset = datasets.load_dataset(data_string)
                return cast(Dataset, dataset)
            except Exception as error:
                raise DatasetLoadError(str) from error

        factory: Dict[str, Callable[[str], Any]] = {
            ".txt": Dataset.from_text,
            ".csv": Dataset.from_csv,
            ".json": Dataset.from_json,
        }

        extension = os.path.splitext(data_string)[1]
        if extension not in factory:
            raise DatasetLoadError(type(data_string))

        dataset = factory[extension](data_string)
        dataset_cast = cast(Dataset, dataset)
        return dataset_cast


class Label(ABC):
    def __init__(self, data: Dataset, label_name: Optional[str], map_to_int: bool = False):
        self.label_name = label_name
        self.map_to_int = map_to_int
        self._data = data
        self.labels = self._extract_labels(data, label_name)
        self.label_map = self._create_label_map()

    @property
    def class_names(self) -> List[str]:
        return list(self.label_map.values()) if self.label_map else []

    @property
    def is_available(self) -> bool:
        return self.label_name is not None and self.label_name in self._data.column_names

    @abstractmethod
    def _extract_labels(self, data: Dataset, label_name: Optional[str]):
        pass

    @abstractmethod
    def _create_label_map(self) -> Dict[int, str]:
        pass

    def __eq__(self, other) -> bool:
        if not isinstance(other, Label):
            return False
        return (
            self.label_name == other.label_name
            and self.map_to_int == other.map_to_int
            and self.label_map == other.label_map
            and self.labels == other.labels
        )


class MultiClass(Label):
    def _extract_labels(self, data: Dataset, label_name: Optional[str]) -> np.ndarray:
        if not self.is_available:
            return np.array([])
        labels = data[label_name]
        if self.map_to_int:
            labels = labels_to_array(labels)
        return np.asarray(labels)

    def _create_label_map(self) -> Dict[int, str]:
        if not self.is_available:
            return {}
        labels = self._data[self.label_name]
        feature = self._data.features.get(self.label_name)
        if isinstance(feature, ClassLabel):
            return dict(enumerate(feature.names))
        unique_labels = sorted(set(labels))
        if self.map_to_int:
            return {i: str(label) for i, label in enumerate(unique_labels)}
        return {int(label): str(label) for label in unique_labels}


class MultiLabel(Label):
    def _extract_labels(self, data: Dataset, label_name: Optional[str]) -> List[List[int]]:
        if not self.is_available:
            return []
        labels = data[label_name]
        return labels_to_list_multilabel(labels)

    def _create_label_map(self) -> Dict[int, str]:
        if not self.is_available:
            return {}
        labels = self._data[self.label_name]
        flattened = sorted(set(label for row in labels for label in row))
        return {i: str(label) for i, label in enumerate(flattened)}
