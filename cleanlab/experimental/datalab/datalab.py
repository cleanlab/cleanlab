# Copyright (C) 2017-2022  Cleanlab Inc.
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
Implements cleanlab's DataLab interface as a one-stop-shop for tracking
and managing all kinds of issues in datasets.
"""

from typing import Any, Union, Mapping, Protocol

import datasets
from datasets import Dataset
import pandas as np
import numpy as np
from cleanlab.classification import CleanLearning
from cleanlab.internal.validation import labels_to_array


class Datalab:
    """
    A single object to find all kinds of issues in datasets.
    It tracks intermediate state from certain functions that can be
    re-used across other functions.  This will become the main way 90%
    of users interface with cleanlab library.

    Parameters
    ----------
    ... : ...
        ...
    """

    def __init__(
        self,
        data: Dataset,
        labels: Union[str, list[str]],
    ) -> None:
        self._validate_data(data)
        self._validate_data_and_labels(data, labels)

        self.data = data
        self.labels = labels
        self.issues = None
        self.results = None
        self._labels = self._extract_labels(self.labels)
        class_names = self.data.unique(self.labels)  # TODO
        self.info = {
            "num_examples": len(self.data),
            "class_names": class_names,
            "num_classes": len(class_names),
        }
        self._silo = self.info.copy()

    def find_issues(
        self,
        issue_types: dict = None,
        pred_probs=None,
        feature_values=None,  # embeddings of data
        model=None,  # sklearn.Estimator compatible object
    ) -> Any:
        """
        Checks for all sorts of issues in the data, including in labels and in features.

        Can utilize either provided model or pred_probs.

        Parameters
        ----------
        issue_types :
            Collection of the types of issues to search for.

        pred_probs :
            Out-of-sample predicted probabilities made on the data.

        feature_values :
            Precomputed embeddings of the features in the dataset.

            WARNING
            -------
            This is not yet implemented.

        model :
            sklearn compatible model used to compute out-of-sample predicted probability for the labels.

            WARNING
            -------
            This is not yet implemented.
        """
        cl = CleanLearning()

        if pred_probs is None and model is not None:
            raise NotImplementedError("TODO: We assume pred_probs is provided.")

        if pred_probs:
            self.issues = cl.find_label_issues(labels=self._labels, pred_probs=pred_probs)

    def _extract_labels(self, labels: Union[str, list[str]]) -> tuple[np.ndarray, Mapping]:
        """
        Extracts labels from the data and stores it in self._labels.

        Parameters
        ----------
        ... : ...
            ...
        """
        if isinstance(labels, str):
            _labels = self.data[labels]

            _labels = labels_to_array(_labels)
            if _labels.ndim != 1:
                raise ValueError("labels must be 1D numpy array.")

            unique_labels = np.unique(_labels)
            label_map = {label: i for i, label in enumerate(unique_labels)}
            formatted_labels = np.array([label_map[l] for l in labels])
            inverse_map = {i: label for label, i in label_map.items()}

        elif isinstance(labels, list[str]):

            raise NotImplementedError("TODO")

            # _labels = np.vstack([my_data[label] for label in labels]).T

        return labels, inverse_map

    @staticmethod
    def _validate_data(data) -> None:
        assert not isinstance(
            data, datasets.DatasetDict
        ), "Please pass a single dataset, not a DatasetDict. Try initializing with data['train'] instead."

        assert isinstance(data, Dataset)

    @staticmethod
    def _validate_data_and_labels(data, labels) -> None:
        if isinstance(labels, np.ndarray):
            assert labels.shape[0] == data.shape[0]

        if isinstance(labels, str):
            pass

    def save(self, path, save_data=False):
        """Saves this Lab to file (all files are in folder at path).
        Uses nice format for the DF attributes (csv) and dict attributes (eg. json if possible).
        We do not guarantee saved Lab can be loaded from future versions of cleanlab.
        """
        if save_data:
            pass  # TODO
            # also call Dataset.save(...) to a location inside the path folder. Otherwise set self.

    @classmethod
    def load(path, data=None):
        """Loads Lab from file. Folder could ideally be zipped or unzipped.
        Checks which cleanlab version Lab was previously saved from and raises warning if they dont match.
        """
        ...
