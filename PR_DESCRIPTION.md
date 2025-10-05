# Datalab: accept HF Datasets Column for labels (convert to list)

## Summary

🎯 Purpose: Make Datalab robust when the Hugging Face `datasets` library returns a `Column` for label lookup via `dataset["y"]`.

In certain `datasets` versions, accessing a column via `dataset["y"]` yields a `datasets.arrow_dataset.Column` (instead of a Python `list`/`np.ndarray`). Cleanlab's `Label._validate_labels` previously asserted that labels must be a `list` or `np.ndarray`, causing an `AssertionError` for valid HF datasets inputs.

This PR converts `Column` (or any object with `to_pylist`) to a plain Python list prior to the existing type/length validation. No public APIs change; behavior for existing `list`/`np.ndarray` inputs remains the same.

### Example Usage (MWE)

```python
import pandas as pd
from datasets import Dataset as HFDataset
from cleanlab import Datalab

# Build a tiny tabular dataset
df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 0]})
ds = HFDataset.from_pandas(df, preserve_index=False)

# In newer datasets releases, ds["y"] is a Column, not list/ndarray
print("Type of ds['y']:", type(ds["y"]))  # e.g., datasets.arrow_dataset.Column

# Previously this could assert inside Label._validate_labels.
# After this PR, it should succeed.
lab = Datalab(ds, label_name="y")
print("Datalab initialized. Number of examples:", len(lab.data))
```

## Impact

- Areas affected:
  - `cleanlab/datalab/internal/data.py`
    - Method: `Label._validate_labels`
      - New behavior: detect HF `Column` (or any object with `to_pylist`) and convert to Python `list` before validation; fallback to `list(labels)` when appropriate; then apply original assertions (type and length).
  - New test: `tests/datalab/test_label_validation_hf_column.py`
    - Ensures that when `dataset["y"]` yields a `Column`, `Data(..., label_name="y")` initializes without error (classification task).

- Who’s affected:
  - Users who pass a Hugging Face `Dataset` directly to `Datalab` and rely on column access for labels. They will no longer see an `AssertionError` when `dataset[label_name]` is a `Column`.

## Screenshots

N/A — this is a behavioral fix without UI/docs changes.

## Testing

- Unit test added: `tests/datalab/test_label_validation_hf_column.py`
  - Builds an HF `Dataset` from a pandas DataFrame and asserts that `Data(ds, Task.from_str("classification"), label_name="y")` succeeds even if `ds["y"]` is a `Column`.

- Local checks (manual):
  - Verified with `datasets` ≥ 2.14 and `pyarrow` 14.0.x that:
    - `ds["y"]` is a `Column`.
    - `Datalab(ds, label_name="y")` initializes successfully.
    - `lab.find_issues(...)` runs end-to-end on a small tabular dataset.

### Unaddressed Cases

- Multilabel paths continue to rely on `labels_to_list_multilabel`; after conversion to a Python list, behavior remains consistent.
- If future `datasets` types expose a different interface for column-like objects without `to_pylist` but are still iterable, the code falls back to `list(labels)` before asserting type/length.

## Links to Relevant Issues or Conversations

- Not tied to a tracked issue. Happy to open one if maintainers prefer.

## References

- `datasets` behavior: newer versions often return `datasets.arrow_dataset.Column` for `Dataset.__getitem__("col_name")`.
- Cleanlab expectation: `Label._validate_labels` asserts labels are `list`/`np.ndarray` and match dataset length; this PR broadens accepted types without changing public APIs.

Additional snippet (alternative that users can always use, even without this fix):

```python
# Construct dict-of-lists to avoid Column entirely
lab = Datalab({"x": df["x"].tolist(), "y": df["y"].tolist()}, label_name="y")
```

## Reviewer Notes

- Implementation is intentionally minimal and localized:
  - Convert `Column` (or objects exposing `to_pylist`) to list; otherwise try `list(labels)`.
  - Preserve existing assertions (type and length) and existing label mapping logic.
- Alternative considered: forcing `dataset.with_format("python")` — rejected to avoid mutating user-provided datasets or relying on specific formatting semantics. The current approach keeps Datalab tolerant of upstream library changes with minimal surface area.
