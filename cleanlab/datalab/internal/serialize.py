# Copyright (C) 2017-2023  Cleanlab Inc.
# ... (license header) ...
from __future__ import annotations

import json
import os
import pickle
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.neighbors import NearestNeighbors

import cleanlab
from cleanlab.datalab.internal.data import Data
from cleanlab.datalab.internal.task import Task
from cleanlab.outlier import OutOfDistribution

if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset
    from cleanlab.datalab.datalab import Datalab


# --- Component-based file constants ---
INFO_FILENAME: str = "info.json"
LABELS_FILENAME: str = "labels.parquet"
ISSUES_FILENAME: str = "issues.parquet"
ISSUE_SUMMARY_FILENAME: str = "issue_summary.parquet"
DATA_DIRNAME: str = "data"
LEGACY_OBJECT_FILENAME: str = "datalab.pkl"


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively converts numpy types to standard Python types for JSON compatibility.

    This ensures that data structures containing numpy integers, floats, or arrays
    can be safely serialized to a .json file.

    Args:
        obj (Any): The Python object to sanitize. Can be a dict, list, or primitive.

    Returns:
        Any: The sanitized object with all numpy types converted to standard Python types.
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="split")

    if isinstance(obj, OutOfDistribution):
        return _sanitize_for_json(obj.__dict__)
    if isinstance(obj, NearestNeighbors):
        # Replace the unserializable scikit-learn model with a placeholder string
        return f"<Unserializable: {obj.__class__.__name__}>"

    if isinstance(obj, dict):
        return {key: _sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(element) for element in obj]
    return obj


class _Serializer:
    """Handles saving and loading a Datalab object to/from disk."""

    @staticmethod
    def _save_data(path: str, datalab: Datalab) -> None:
        """Saves the raw Arrow dataset from a Datalab object to disk.

        Args:
            path (str): The root directory where the Datalab is being saved.
            datalab (Datalab): The Datalab instance containing the data to save.
        """
        data_path: str = os.path.join(path, DATA_DIRNAME)
        datalab.data.save_to_disk(data_path)

    @staticmethod
    def _validate_version(datalab: Datalab) -> None:
        """Compares the Datalab's saved version with the current cleanlab version and warns on mismatch.

        Args:
            datalab (Datalab): The Datalab instance loaded from disk.
        """
        current_version: str = cleanlab.__version__
        datalab_version: str = datalab.cleanlab_version
        if current_version != datalab_version:
            warnings.warn(
                f"Saved Datalab was created with cleanlab version {datalab_version}, "
                f"but you are using version {current_version}. Datalab may not work properly."
            )

    @classmethod
    def serialize(cls, path: str, datalab: Datalab, force: bool) -> None:
        """Serializes a Datalab object to a specified directory using a component-based format.

        This method saves the Datalab's components (metadata, labels, issues, etc.)
        into separate, type-appropriate files (JSON, Parquet) for security and efficiency.

        Args:
            path (str): The directory path to save the Datalab object to.
            datalab (Datalab): The Datalab instance to serialize.
            force (bool): If True, overwrite the directory if it already exists.

        Raises:
            FileExistsError: If the path already exists and `force` is False.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        elif not force:
            raise FileExistsError(f"Directory already exists: {path}. Use force=True to overwrite.")

        info_data: Dict[str, Any] = {
            "cleanlab_version": str(datalab.cleanlab_version),
            "task": str(datalab.task),
            "label_name": str(datalab.label_name),
            "info": _sanitize_for_json(datalab.info),
        }
        info_path: str = os.path.join(path, INFO_FILENAME)
        with open(info_path, "w") as f:
            json.dump(info_data, f, indent=2)

        labels_path: str = os.path.join(path, LABELS_FILENAME)
        pq.write_table(
            pa.Table.from_arrays([pa.array(datalab.labels)], names=["label"]), labels_path
        )

        issues_path: str = os.path.join(path, ISSUES_FILENAME)
        pq.write_table(pa.Table.from_pandas(datalab.issues), issues_path)

        issue_summary_path: str = os.path.join(path, ISSUE_SUMMARY_FILENAME)
        pq.write_table(pa.Table.from_pandas(datalab.issue_summary), issue_summary_path)

        cls._save_data(path=path, datalab=datalab)

    @classmethod
    def deserialize(cls, path: str, data: Optional[Dataset] = None) -> Datalab:
        """Deserializes a Datalab object from a directory.

        This method supports both the new component-based format and provides
        automatic, seamless migration from the legacy pickle format.

        Args:
            path (str): The directory path where the Datalab object was saved.
            data (Optional[Dataset]): If provided, this dataset will be used as the
                Datalab's primary data, overriding any data found in the saved directory.

        Returns:
            Datalab: The re-hydrated Datalab instance.

        Raises:
            FileNotFoundError: If no Datalab object (new or legacy) is found at the path.
        """
        from cleanlab.datalab.datalab import Datalab

        if not os.path.exists(path):
            raise FileNotFoundError(f"No Datalab folder found at: {path}")

        info_path: str = os.path.join(path, INFO_FILENAME)
        legacy_path: str = os.path.join(path, LEGACY_OBJECT_FILENAME)
        datalab: Datalab

        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                info_data: Dict[str, Any] = json.load(f)

            labels_path: str = os.path.join(path, LABELS_FILENAME)
            labels_table: pa.Table = pq.read_table(labels_path)
            labels: np.ndarray = labels_table["label"].to_numpy()

            label_name: str = info_data.get("label_name", "label")
            temp_data_dict: Dict[str, Union[List[int], np.ndarray]] = {
                "placeholder": [0] * len(labels),
                label_name: labels,
            }

            datalab = Datalab(data=temp_data_dict, label_name=label_name)

            datalab.cleanlab_version = info_data.get("cleanlab_version", "")
            datalab.info = info_data.get("info", {})
            task_str: str = info_data.get("task", "classification")
            datalab.task = Task.from_str(task_str)

            issues_path: str = os.path.join(path, ISSUES_FILENAME)
            datalab.issues = pd.read_parquet(issues_path)

            issue_summary_path: str = os.path.join(path, ISSUE_SUMMARY_FILENAME)
            datalab.issue_summary = pd.read_parquet(issue_summary_path)

            if datalab.issues.shape == (0, 0) and len(labels) > 0:
                datalab.issues = pd.DataFrame(index=range(len(labels)))

            datalab.data_issues.issues = datalab.issues
            datalab.data_issues.issue_summary = datalab.issue_summary

        elif os.path.exists(legacy_path):
            warnings.warn(
                f"Migrating legacy '.pkl' Datalab to the new secure component format. "
                f"The original '.pkl' file will be removed after migration.",
                UserWarning,
            )
            try:
                with open(legacy_path, "rb") as f:
                    datalab = pickle.load(f)
            except pickle.UnpicklingError as e:
                raise pickle.UnpicklingError(
                    f"Failed to load {legacy_path}: invalid pickle data"
                ) from e

            cls.serialize(path=path, datalab=datalab, force=True)
            os.remove(legacy_path)

        else:
            raise FileNotFoundError(
                f"Cannot load Datalab. No '{INFO_FILENAME}' or '{LEGACY_OBJECT_FILENAME}' found in {path}"
            )

        cls._validate_version(datalab)

        data_path: str = os.path.join(path, DATA_DIRNAME)
        if data is None and os.path.exists(data_path):
            from datasets import load_from_disk

            reloaded_data: Dataset = load_from_disk(data_path)
            datalab._data = Data(reloaded_data, datalab.task, datalab.label_name)
            datalab.data = datalab._data._data
        elif data is not None:
            datalab._data = Data(data, datalab.task, datalab.label_name)
            datalab.data = datalab._data._data

        return datalab
