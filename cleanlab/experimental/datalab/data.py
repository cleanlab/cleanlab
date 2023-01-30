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
"""Module for class and functions that hold and validate datasets that are loaded into DataLab."""

from typing import Mapping, Union, cast

import datasets
import numpy as np
from datasets.arrow_dataset import Dataset

from cleanlab.internal.validation import labels_to_array


class Data:
    """
    Class that holds and validates datasets for DataLab.

    Parameters
    ----------
    data : Dataset
        Dataset to be used for DataLab.

    label_name : Union[str, list[str]]
        Name of the label column in the dataset.
    """

    def __init__(self, data: Dataset, label_name: Union[str, list[str]]) -> None:
        self._validate_data(data)
        self._validate_data_and_labels(data, label_name)
        self._data = data
        self._data_hash = hash(data)
        self._data.set_format(type="numpy")
        self._labels, self._label_map = _extract_labels(data, label_name)
        self._label_name = label_name

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> bool:
        if isinstance(other, Data):
            # Equality checks
            hashes = self._data_hash == other._data_hash
            labels = np.array_equal(self._labels, other._labels)
            label_names = self._label_name == other._label_name
            label_maps = self._label_map == other._label_map
            return all([hashes, labels, label_names, label_maps])
        return False

    def __hash__(self) -> int:
        return self._data_hash

    @property
    def class_names(self) -> list:
        label_name = cast(str, self._label_name)
        return self._data.unique(label_name)

    @staticmethod
    def _validate_data(data) -> None:
        if isinstance(data, datasets.DatasetDict):
            raise ValueError(
                "Please pass a single dataset, not a DatasetDict. "
                "Try initializing with data['train'] instead."
            )

        assert isinstance(data, Dataset)

    @staticmethod
    def _validate_data_and_labels(data, labels) -> None:
        if isinstance(labels, np.ndarray):
            assert labels.shape[0] == data.shape[0]

        if isinstance(labels, str):
            pass


def _extract_labels(data: Dataset, label_name: Union[str, list[str]]) -> tuple[np.ndarray, Mapping]:
    """
    Picks out labels from the dataset and formats them to be [0, 1, ..., K-1]
    where K is the number of classes. Also returns a mapping from the formatted
    labels to the original labels in the dataset.

    Note: This function is not meant to be used directly. It is used by
    `cleanlab.data.Data` to extract the formatted labels from the dataset
    and stores them as attributes.

    Parameters
    ----------
    label_name : str or list[str]
        Name of the column in the dataset that contains the labels.

    Returns
    -------
    formatted_labels : np.ndarray
        Labels in the format [0, 1, ..., K-1] where K is the number of classes.

    inverse_map : dict
        Mapping from the formatted labels to the original labels in the dataset.
    """

    if isinstance(label_name, list):

        raise NotImplementedError("TODO")

        # _labels = np.vstack([my_data[label] for label in labels]).T

    labels = labels_to_array(data[label_name])  # type: ignore[assignment]
    if labels.ndim != 1:
        raise ValueError("labels must be 1D numpy array.")

    unique_labels = np.unique(labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    # labels 0, 1, ..., K-1
    formatted_labels = np.array([label_map[label] for label in labels])
    inverse_map = {i: label for label, i in label_map.items()}

    return formatted_labels, inverse_map
