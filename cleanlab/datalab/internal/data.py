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
    """
    Class to represent labels in a dataset.

    It stores the labels as a numpy array and maps them to integers if necessary.
    If a mapping is not necessary, e.g. for regression tasks, the mapping will be an empty dictionary.

    Parameters
    ----------
    data :
        A Hugging Face Dataset object.

    label_name : str
        Name of the label column in the dataset.

    map_to_int : bool
        Whether to map the labels to integers, e.g. [0, 1, ..., K-1] where K is the number of classes.
        If False, the labels are not mapped to integers, e.g. for regression tasks.
    """

    def __init__(
        self, *, data: Dataset, label_name: Optional[str] = None, map_to_int: bool = True
    ) -> None:
        self._data = data
        self.label_name = label_name
        self.labels = labels_to_array([])
        self.label_map: Mapping[Union[str, int], Any] = {}
        if label_name is not None:
            self.labels, self.label_map = self._extract_labels(data, label_name, map_to_int)
            self._validate_labels()

    def __len__(self) -> int:
        if self.labels is None:
            return 0
        return len(self.labels)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Label):
            labels_are_equal = np.array_equal(self.labels, __value.labels)
            names_are_equal = self.label_name == __value.label_name
            maps_are_equal = self.label_map == __value.label_map
            return all([labels_are_equal, names_are_equal, maps_are_equal])
        return False

    def __getitem__(self, __index: Union[int, slice, np.ndarray]) -> np.ndarray:
        return self.labels[__index]

    def __bool__(self) -> bool:
        return self.is_available

    @property
    def class_names(self) -> List[str]:
        """A list of class names that are present in the dataset.

        Without labels, this will return an empty list.
        """
        return list(self.label_map.values())

    @property
    def is_available(self) -> bool:
        """Check if labels are available."""
        empty_labels = self.labels is None or len(self.labels) == 0
        empty_label_map = self.label_map is None or len(self.label_map) == 0
        return not (empty_labels or empty_label_map)

    def _validate_labels(self) -> None:
        if self.label_name not in self._data.column_names:
            raise ValueError(f"Label column '{self.label_name}' not found in dataset.")
        labels = self._data[self.label_name]
        assert isinstance(labels, (np.ndarray, list))
        assert len(labels) == len(self._data)

    @abstractmethod
    def _extract_labels(self, *args, **kwargs) -> Any:
        """Extract labels from the dataset and formats them"""
        raise NotImplementedError


class MultiLabel(Label):
    def __init__(self, data, label_name, map_to_int):
        super().__init__(data=data, label_name=label_name, map_to_int=map_to_int)

    def _extract_labels(
        self, data: Dataset, label_name: str, map_to_int: bool
    ) -> Tuple[List[List[int]], Dict[int, Any]]:
        labels: List[List[int]] = labels_to_list_multilabel(data[label_name])
        # label_map needs to be lexicographically sorted. np.unique should sort it
        unique_labels = np.unique([x for ele in labels for x in ele])
        label_map = {label: i for i, label in enumerate(unique_labels)}
        formatted_labels = [[label_map[item] for item in label] for label in labels]
        inverse_map = {i: label for label, i in label_map.items()}
        return formatted_labels, inverse_map


class MultiClass(Label):
    def __init__(self, data, label_name, map_to_int):
        super().__init__(data=data, label_name=label_name, map_to_int=map_to_int)

    def _extract_labels(self, data: Dataset, label_name: str, map_to_int: bool):
        """
        Picks out labels from the dataset and formats them to be [0, 1, ..., K-1]
        where K is the number of classes. Also returns a mapping from the formatted
        labels to the original labels in the dataset.

        Note: This function is not meant to be used directly. It is used by
        ``cleanlab.data.Data`` to extract the formatted labels from the dataset
        and stores them as attributes.

        Parameters
        ----------
        data : datasets.Dataset
            A Hugging Face Dataset object.

        label_name : str
            Name of the column in the dataset that contains the labels.

        map_to_int : bool
            Whether to map the labels to integers, e.g. [0, 1, ..., K-1] where K is the number of classes.
            If False, the labels are not mapped to integers, e.g. for regression tasks.
        Returns
        -------
        formatted_labels : np.ndarray
            Labels in the format [0, 1, ..., K-1] where K is the number of classes.

        inverse_map : dict
            Mapping from the formatted labels to the original labels in the dataset.
        """

        labels = labels_to_array(data[label_name])  # type: ignore[assignment]
        if labels.ndim != 1:
            raise ValueError("labels must be 1D numpy array.")

        if not map_to_int:
            # Don't map labels to integers, e.g. for regression tasks
            return labels, {}
        label_name_feature = data.features[label_name]
        if isinstance(label_name_feature, ClassLabel):
            label_map = {
                label: label_name_feature.str2int(label) for label in label_name_feature.names
            }
            formatted_labels = labels
        else:
            label_map = {label: i for i, label in enumerate(np.unique(labels))}
            formatted_labels = np.vectorize(label_map.get)(labels)
        inverse_map = {i: label for label, i in label_map.items()}

        return formatted_labels, inverse_map
