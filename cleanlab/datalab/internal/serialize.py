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
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from cleanlab.outlier import OutOfDistribution

import cleanlab
from cleanlab.datalab.internal.data import Data, Label, MultiLabel, MultiClass

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
            """Custom serializer for handling specific data types."""
            if isinstance(obj, np.integer):
                return {"__type__": type(obj).__name__, "value": int(obj)}

            if isinstance(obj, np.ndarray):
                return {"__type__": "ndarray", "data": obj.tolist()}

            if isinstance(obj, csr_matrix):
                return {
                    "__type__": "csr_matrix",
                    "data": obj.data.tolist(),
                    "indices": obj.indices.tolist(),
                    "indptr": obj.indptr.tolist(),
                    "shape": obj.shape,
                }

            if isinstance(obj, pd.DataFrame):
                return {
                    "__type__": "DataFrame",
                    "data": obj.to_dict(orient="records"),
                    "columns": obj.columns.tolist(),
                }

            if isinstance(obj, Label):
                return {
                    "__type__": "Label",
                    "class_name": obj.__class__.__name__,
                    "label_map": obj.label_map,
                    "labels": obj.labels,
                }

            if isinstance(obj, OutOfDistribution):
                return {
                    "__type__": "OutOfDistribution",
                    "params": {k: v for k, v in obj.params.items() if k != "scaling_factor"},
                }

            if isinstance(obj, NearestNeighbors):
                return {
                    "__type__": "NearestNeighbors",
                    "n_neighbors": obj.n_neighbors,
                    "metric": obj.metric,
                    "algorithm": obj.algorithm,
                    "leaf_size": obj.leaf_size,
                    "p": obj.p,
                    "_fit_X": obj._fit_X.tolist(),
                }

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

        def custom_deserializer(obj):
            """Custom deserializer for handling specific data types."""
            if "__type__" in obj:
                obj_type = obj["__type__"]

                if obj_type.startswith("int") or obj_type.startswith("uint"):
                    np_type = getattr(np, obj_type, None)
                    if np_type is not None:
                        return np_type(obj["value"])

                if obj_type == "ndarray":
                    return np.array(obj["data"])

                if obj_type == "csr_matrix":
                    return csr_matrix(
                        (obj["data"], obj["indices"], obj["indptr"]), shape=obj["shape"]
                    )

                if obj_type == "DataFrame":
                    return pd.DataFrame(obj["data"], columns=obj["columns"])

                if obj_type == "Label":
                    class_name = obj.get("class_name")
                    if not class_name:
                        raise ValueError("Missing 'class_name' in serialized Label object.")

                    # Dynamically resolve subclass
                    subclass = globals().get(class_name)
                    if not subclass or not issubclass(subclass, Label):
                        raise ValueError(f"Invalid class '{class_name}' for Label.")

                    # Create instance with placeholders
                    instance = subclass(data=None, label_name=None, map_to_int=False)

                    # Manually set attributes
                    instance.label_map = obj["label_map"]

                    # Handle labels dynamically based on subclass
                    if class_name == "MultiClass":
                        instance.labels = np.array(obj["labels"])  # Ensure 1D array
                    elif class_name == "MultiLabel":
                        instance.labels = [
                            np.array(label) for label in obj["labels"]
                        ]  # Ensure 2D format

                    # Final validation
                    if not isinstance(instance.labels, (np.ndarray, list)):
                        raise ValueError(
                            f"Deserialized labels have invalid type: {type(instance.labels)}"
                        )

                    return instance

                if obj_type == "OutOfDistribution":
                    return OutOfDistribution(params=obj["params"])

                if obj_type == "NearestNeighbors":
                    return NearestNeighbors(
                        n_neighbors=obj["n_neighbors"],
                        metric=obj["metric"],
                        algorithm=obj["algorithm"],
                        leaf_size=obj["leaf_size"],
                        p=obj["p"],
                    ).fit(np.array(obj["_fit_X"]))

                raise ValueError(f"Unsupported type during deserialization: {obj_type}")

            return obj

        with open(os.path.join(path, OBJECT_FILENAME), "rb") as f:
            datalab_metadata = json.load(f, object_hook=custom_deserializer)
            task = datalab_metadata["task"]
            verbosity = datalab_metadata["verbosity"]
            from cleanlab.datalab.datalab import Datalab

            datalab: Datalab = Datalab(data=[], task=task, verbosity=verbosity)
            datalab.label_name = datalab_metadata["label_name"]
            datalab.cleanlab_version = datalab_metadata["cleanlab_version"]
            datalab.info = datalab_metadata["info"]
            datalab._data_hash = datalab_metadata["_data_hash"]
            datalab._labels = datalab_metadata["_labels"]

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
