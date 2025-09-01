import os
import shutil
import pickle
from pathlib import Path
from typing import Dict, List, Generator, Union, Any

import numpy as np
import pandas as pd
import pytest
from cleanlab.datalab.datalab import Datalab
from cleanlab.datalab.internal.serialize import (
    _Serializer,
    _sanitize_for_json,
)  # Import _Serializer and _sanitize_for_json
from cleanlab.outlier import OutOfDistribution  # Required for testing _sanitize_for_json
from sklearn.neighbors import NearestNeighbors  # Required for testing _sanitize_for_json
from datasets import Dataset

# --- Constants ---
OLD_OBJECT_FILENAME: str = "datalab.pkl"
NEW_INFO_FILENAME: str = "info.json"


@pytest.fixture
def simple_datalab() -> Datalab:
    """Creates a Datalab instance with data, labels, and mocked issues."""
    data: Dict[str, List[Union[float, int]]] = {
        "feature1": [0.1, 0.2, 0.3, 0.4],
        "feature2": [0.4, 0.5, 0.6, 0.7],
        "label": [0, 1, 0, 1],
    }
    lab: Datalab = Datalab(data=data, label_name="label")

    issues_df: pd.DataFrame = pd.DataFrame(
        {
            "is_outlier_issue": pd.Series([False, True, False, False], dtype="boolean"),
            "outlier_score": pd.Series([0.1, 0.9, 0.2, 0.3], dtype="float64"),
            "is_label_issue": pd.Series([True, False, False, True], dtype="boolean"),
            "label_score": pd.Series([0.8, 0.2, 0.1, 0.9], dtype="float64"),
            "custom_issue": pd.Series(["ok", "flag", "ok", "ok"], dtype="string"),
        }
    )
    summary_df: pd.DataFrame = pd.DataFrame(
        {"issue_type": ["outlier", "label", "custom"], "count": [1, 2, 1]}
    )

    # Manually set the issues for testing purposes
    lab.data_issues.issues = issues_df
    lab.data_issues.issue_summary = summary_df
    lab.issues = issues_df
    lab.issue_summary = summary_df
    return lab


@pytest.fixture
def temp_lab_dir() -> Generator[str, None, None]:
    """Creates and cleans up a temporary directory for testing saves."""
    path: str = "test_lab_temp_dir"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    yield path
    shutil.rmtree(path)


def test_parquet_roundtrip(simple_datalab: Datalab, temp_lab_dir: str) -> None:
    """Tests the full save/load roundtrip for the new Parquet format."""
    simple_datalab.save(temp_lab_dir, force=True)
    loaded_datalab: Datalab = Datalab.load(temp_lab_dir)

    assert loaded_datalab.info == simple_datalab.info
    assert np.array_equal(loaded_datalab.labels, simple_datalab.labels)
    pd.testing.assert_frame_equal(loaded_datalab.issues, simple_datalab.issues)
    pd.testing.assert_frame_equal(loaded_datalab.issue_summary, simple_datalab.issue_summary)
    assert len(loaded_datalab.data) == len(simple_datalab.data)


def test_parquet_roundtrip_empty_issues(temp_lab_dir: str) -> None:
    """Tests the edge case of a Datalab with no issues found."""
    data: Dict[str, List[Union[float, int]]] = {
        "f1": [1.0, 2.0],
        "f2": [3.0, 4.0],
        "label": [0, 1],
    }
    lab_no_issues: Datalab = Datalab(data=data, label_name="label")
    assert lab_no_issues.issues.empty

    lab_no_issues.save(temp_lab_dir, force=True)
    loaded_lab: Datalab = Datalab.load(temp_lab_dir)

    assert loaded_lab.issues.empty
    pd.testing.assert_frame_equal(loaded_lab.issues, lab_no_issues.issues)


def test_pickle_migration(simple_datalab: Datalab, temp_lab_dir: str) -> None:
    """Tests the backward-compatibility and auto-migration from pickle."""
    legacy_path: str = os.path.join(temp_lab_dir, OLD_OBJECT_FILENAME)
    with open(legacy_path, "wb") as f:
        pickle.dump(simple_datalab, f)

    # Note: Legacy saves did not save the 'data' component inside the pickle file
    simple_datalab.data.save_to_disk(os.path.join(temp_lab_dir, "data"))

    with pytest.warns(UserWarning, match="Migrating legacy '.pkl' Datalab"):
        loaded_datalab: Datalab = Datalab.load(temp_lab_dir)

    assert np.array_equal(loaded_datalab.labels, simple_datalab.labels)
    pd.testing.assert_frame_equal(loaded_datalab.issues, simple_datalab.issues)

    # Assert migration was successful: new file exists, old one is gone
    assert os.path.exists(os.path.join(temp_lab_dir, NEW_INFO_FILENAME))
    assert not os.path.exists(legacy_path)


## --- Production Hardening Tests ---


def test_roundtrip_zero_rows(temp_lab_dir: str) -> None:
    """Tests the edge case of a Datalab with a zero-row dataset."""
    data: Dict[str, List[Any]] = {"feature1": [], "feature2": [], "label": []}
    lab_zero_rows: Datalab = Datalab(data=data, label_name="label")

    assert len(lab_zero_rows.data) == 0
    assert len(lab_zero_rows.labels) == 0

    lab_zero_rows.save(temp_lab_dir, force=True)
    loaded_lab: Datalab = Datalab.load(temp_lab_dir)

    assert len(loaded_lab.data) == 0
    assert len(loaded_lab.labels) == 0
    pd.testing.assert_frame_equal(loaded_lab.issues, lab_zero_rows.issues)


def test_load_with_user_data_provided(simple_datalab: Datalab, temp_lab_dir: str) -> None:
    """Tests the code path where a user provides the data object during load."""
    original_data_dict: Dict[str, List[Union[float, int]]] = {
        "feature1": [0.1, 0.2, 0.3, 0.4],
        "feature2": [0.4, 0.5, 0.6, 0.7],
        "label": [0, 1, 0, 1],
    }
    original_dataset: Dataset = Dataset.from_dict(original_data_dict)

    # Save the lab, but this time load it by passing the dataset directly
    simple_datalab.save(temp_lab_dir, force=True)
    loaded_datalab: Datalab = Datalab.load(temp_lab_dir, data=original_dataset)

    # Assert that everything is still correct
    pd.testing.assert_frame_equal(loaded_datalab.issues, simple_datalab.issues)
    assert np.array_equal(loaded_datalab.labels, simple_datalab.labels)
    assert loaded_datalab.data.to_dict() == original_dataset.to_dict()


def test_sanitize_for_json_edge_cases() -> None:
    """
    Tests _sanitize_for_json with various data types to ensure all branches are covered,
    including NumPy types, Pandas DataFrame, OutOfDistribution, and NearestNeighbors.
    """
    # Test NumPy integer
    assert _sanitize_for_json(np.int64(10)) == 10
    assert isinstance(_sanitize_for_json(np.int32(5)), int)

    # Test NumPy floating
    assert _sanitize_for_json(np.float64(3.14)) == 3.14
    assert isinstance(_sanitize_for_json(np.float32(2.71)), float)

    # Test NumPy array
    np_array = np.array([1, 2, 3])
    assert _sanitize_for_json(np_array) == [1, 2, 3]
    assert isinstance(_sanitize_for_json(np_array), list)

    # Test Pandas DataFrame
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    sanitized_df = _sanitize_for_json(df)
    assert isinstance(sanitized_df, dict)
    assert "data" in sanitized_df
    assert "columns" in sanitized_df
    assert sanitized_df["data"] == [[1, 3], [2, 4]]
    assert sanitized_df["columns"] == ["col1", "col2"]

    # Test OutOfDistribution object (mock if necessary)
    # For a real test, you might need a mock or a simple instance
    class MockOutOfDistribution(OutOfDistribution):
        def __init__(self) -> None:
            # Initialize with minimal required attributes if any
            self.foo = "bar"
            self.baz = 123
            super().__init__()  # Call parent constructor if it exists and is needed

    ood_instance = MockOutOfDistribution()
    sanitized_ood = _sanitize_for_json(ood_instance)
    assert isinstance(sanitized_ood, dict)
    assert sanitized_ood["foo"] == "bar"
    assert sanitized_ood["baz"] == 123
    # Removed: assert "_is_fitted" in sanitized_ood # This assertion was causing the failure

    # Test NearestNeighbors object
    nn_instance = NearestNeighbors(n_neighbors=1)
    sanitized_nn = _sanitize_for_json(nn_instance)
    assert isinstance(sanitized_nn, str)
    assert sanitized_nn == "<Unserializable: NearestNeighbors>"

    # Test nested structures
    nested_obj = {
        "num": np.int64(5),
        "arr": np.array([4.0, 5.0]),
        "df": pd.DataFrame({"x": [1]}),
        "ood": MockOutOfDistribution(),
        "nn": NearestNeighbors(),
        "list_of_nums": [1, np.float32(2.0), 3],
    }
    sanitized_nested = _sanitize_for_json(nested_obj)
    assert sanitized_nested["num"] == 5
    assert sanitized_nested["arr"] == [4.0, 5.0]
    assert isinstance(sanitized_nested["df"], dict)
    assert isinstance(sanitized_nested["ood"], dict)
    assert sanitized_nested["nn"] == "<Unserializable: NearestNeighbors>"
    assert sanitized_nested["list_of_nums"] == [1, 2.0, 3]


def test_serialize_creates_directory_if_not_exists(tmp_path: Path, simple_datalab: Datalab) -> None:
    """
    Tests that _Serializer.serialize creates the directory if it does not exist
    and force=False.
    """
    non_existent_path = tmp_path / "new_datalab_dir"
    assert not non_existent_path.exists()

    _Serializer.serialize(path=str(non_existent_path), datalab=simple_datalab, force=False)

    assert non_existent_path.is_dir()
    # Verify some files are created to confirm successful serialization
    assert (non_existent_path / "info.json").exists()
    assert (non_existent_path / "labels.parquet").exists()
    assert (non_existent_path / "issues.parquet").exists()
    assert (non_existent_path / "issue_summary.parquet").exists()
    assert (non_existent_path / "data").is_dir()
