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
from __future__ import annotations

import os
import json
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

import cleanlab
from cleanlab.datalab.internal.data import Data, Label

if TYPE_CHECKING:  # pragma: no cover
    from datasets.arrow_dataset import Dataset

    from cleanlab.datalab.datalab import Datalab


# Constants:
OBJECT_FILENAME = "datalab.json"
ISSUES_FILENAME = "issues.csv"
ISSUE_SUMMARY_FILENAME = "summary.csv"
DATA_DIRNAME = "data"


class _Serializer:
    @staticmethod
    def _save_data_issues(path: str, datalab: Datalab) -> None:
        """Saves the issues to disk."""
        issues_path = os.path.join(path, ISSUES_FILENAME)
        datalab.data_issues.issues.to_csv(issues_path, index=False)

        issue_summary_path = os.path.join(path, ISSUE_SUMMARY_FILENAME)
        datalab.data_issues.issue_summary.to_csv(issue_summary_path, index=False)

    @staticmethod
    def _save_data(path: str, datalab: Datalab) -> None:
        """Saves the dataset to disk."""
        data_path = os.path.join(path, DATA_DIRNAME)
        datalab.data.save_to_disk(data_path)

    @staticmethod
    def _validate_version(datalab: Datalab) -> None:
        current_version = cleanlab.__version__  # type: ignore[attr-defined]
        datalab_version = datalab.cleanlab_version
        if current_version != datalab_version:
            warnings.warn(
                f"Saved Datalab was created using different version of cleanlab "
                f"({datalab_version}) than current version ({current_version}). "
                f"Things may be broken!"
            )

    @classmethod
    def serialize(cls, path: str, datalab: Datalab, force: bool) -> None:
        """Serializes the datalab object to disk.

        Parameters
        ----------
        path : str
            Path to save the datalab object to.

        datalab : Datalab
            The datalab object to save.

        force : bool
            If True, will overwrite existing files at the specified path.
        """
        path_exists = os.path.exists(path)
        if not path_exists:
            os.mkdir(path)
        else:
            if not force:
                raise FileExistsError("Please specify a new path or set force=True")
            print(f"WARNING: Existing files will be overwritten by newly saved files at: {path}")

        def custom_serializer(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Label):
                return {
                    "label_map": obj.label_map,
                    "labels": obj.labels,
                }
            else:
                raise TypeError(f"Type {type(obj)} is not serializable")

        with open(os.path.join(path, OBJECT_FILENAME), "w") as f:
            json.dump(
                {
                    "task": str(datalab.task),
                    "label_name": datalab.label_name,
                    "cleanlab_version": datalab.cleanlab_version,
                    "verbosity": datalab.verbosity,
                    "info": datalab.info,
                    "_labels": datalab._labels,
                    "_data_hash": datalab._data_hash,
                },
                f,
                default=custom_serializer,
            )

        # Save the issues to disk. Use placeholder method for now.
        cls._save_data_issues(path=path, datalab=datalab)

        # Save the dataset to disk
        cls._save_data(path=path, datalab=datalab)

    @classmethod
    def deserialize(cls, path: str, data: Optional[Dataset] = None) -> Datalab:
        """Deserializes the datalab object from disk."""

        if not os.path.exists(path):
            raise ValueError(f"No folder found at specified path: {path}")

        with open(os.path.join(path, OBJECT_FILENAME), "rb") as f:
            datalab_metadata = json.load(f)
            task = datalab_metadata["task"]
            verbosity = datalab_metadata["verbosity"]
            from cleanlab.datalab.datalab import Datalab

            datalab: Datalab = Datalab(data=[], task=task, verbosity=verbosity)
            datalab.label_name = datalab_metadata["label_name"]
            datalab.cleanlab_version = datalab_metadata["cleanlab_version"]
            datalab.info = datalab_metadata["info"]
            datalab._data_hash = datalab_metadata["_data_hash"]
            datalab._labels.labels = datalab_metadata["_labels"]["labels"]
            datalab._labels.label_map = datalab_metadata["_labels"]["label_map"]

        cls._validate_version(datalab)

        # Load the issues from disk.
        issues_path = os.path.join(path, ISSUES_FILENAME)
        if os.path.exists(issues_path):
            datalab.data_issues.issues = pd.read_csv(issues_path)

        issue_summary_path = os.path.join(path, ISSUE_SUMMARY_FILENAME)
        if os.path.exists(issue_summary_path):
            datalab.data_issues.issue_summary = pd.read_csv(issue_summary_path)

        if data is not None:
            if hash(data) != datalab._data_hash:
                raise ValueError(
                    "Data has been modified since Lab was saved. "
                    "Cannot load Lab with modified data."
                )

            if len(data) != len(datalab.labels):
                raise ValueError(
                    f"Length of data ({len(data)}) does not match length of labels ({len(datalab.labels)})"
                )

            datalab._data = Data(data, datalab.task, datalab.label_name)
            datalab.data = datalab._data._data

        return datalab
