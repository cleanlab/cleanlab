import numpy as np
import pandas as pd
import pytest

from cleanlab.datalab.internal.data import Data
from cleanlab.datalab.internal.task import Task

try:
    from datasets import Dataset as HFDataset
    from datasets.arrow_dataset import Column
except Exception:
    HFDataset = None
    Column = None


def test_validate_labels_accepts_hf_column_when_available():
    if HFDataset is None:
        pytest.skip("datasets not installed")

    # Build a minimal HF dataset that yields a Column for label access
    df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 0]})
    ds = HFDataset.from_pandas(df, preserve_index=False)

    # Sanity: ensure type is Column in this environment
    labels_col = ds["y"]
    assert isinstance(labels_col, (list, np.ndarray, Column))

    # Should not raise: Data -> Label._validate_labels should coerce to list/ndarray
    d = Data(ds, Task.from_str("classification"), label_name="y")
    assert len(d) == 3
    assert d.has_labels
