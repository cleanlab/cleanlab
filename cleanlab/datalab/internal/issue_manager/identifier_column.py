from typing import ClassVar, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from cleanlab.datalab.internal.issue_manager import IssueManager


class IdentifierColumnIssueManager(IssueManager):
    """Manages issues related to identifier columns in feature columns"""

    description: ClassVar[
        str
    ] = """Checks whether there is an identifier_column in the features of a dataset.
   Identifier columns are defined as a column i in features such that
   set(features[:,i]) = set(c, c+1, ..., c+n) for some integer c,
   where n = num-rows of features. If there is such a column, the dataset has
   the identifier_column issue
   """
    issue_name: ClassVar[str] = "identifier_column"
    verbosity_levels = {
        0: [],
        1: ["identifier_columns"],
        2: [],
    }

    def _is_sequential(self, arr: npt.NDArray) -> bool:
        """
        Check if the elements in the array are sequential.

        Parameters:
            arr: The input array.

        Returns:
            A boolean indicating whether the elements in the array are sequential.
        """
        if arr.size == 0:
            return False
        unique_sorted = np.unique(arr)  # Returns a sorted unique list
        min_val, max_val = unique_sorted[0], unique_sorted[-1]
        expected_range = np.arange(min_val, max_val + 1)
        if expected_range.size == 1 or unique_sorted.size != expected_range.size:
            return False
        return bool((expected_range == unique_sorted).all())

    def _to_array(self, values: Union[npt.NDArray, list]) -> npt.NDArray:
        arr = np.array(values)
        if arr.size and isinstance(arr[0], str):
            return arr.astype(str)
        return arr

    def _prepare_from_dataframe(self, features: pd.DataFrame) -> List[npt.NDArray]:
        # Keep string columns as strings for consistent downstream handling.
        return [
            np.array(features[col].values).astype(str)
            if pd.api.types.is_string_dtype(features[col])
            else self._to_array(features[col].values)
            for col in features.columns
        ]

    def _prepare_from_dict(self, features: dict) -> List[npt.NDArray]:
        return [self._to_array(value) for value in features.values()]

    def _prepare_from_list(self, features: list) -> List[npt.NDArray]:
        if any(not isinstance(col_list, (list, np.ndarray)) for col_list in features):
            raise ValueError(
                "features must be a list of lists or numpy arrays if a list is passed."
            )
        return [self._to_array(col_list) for col_list in features]

    def _prepare_features(
        self, features: Optional[Union[npt.NDArray, pd.DataFrame, list, dict]]
    ) -> Union[npt.NDArray, List[npt.NDArray]]:
        """
        Prepare the features for issue check.

        Args:
            features: The input features.

        Returns:
            features: features as npt.NDArray
        """
        if isinstance(features, np.ndarray):
            return features.T  # Transpose if it's a NumPy array
        if isinstance(features, pd.DataFrame):
            return self._prepare_from_dataframe(features)
        if isinstance(features, dict):
            return self._prepare_from_dict(features)
        if isinstance(features, list):
            return self._prepare_from_list(features)
        raise ValueError("features must be a numpy array, pandas DataFrame, list, or dict.")

    def find_issues(
        self, features: Optional[Union[npt.NDArray, pd.DataFrame, list, dict]], **kwargs
    ) -> None:
        """
        Find identifier columns in the given dataset.

        Parameters:
            features (Optional[npt.NDArray | pd.DataFrame | list | dict]):
            The dataset to check for identifier columns.
        Returns:
            None
        """
        if features is None:
            raise ValueError("features must be provided to check for identifier columns.")
        processed_features = self._prepare_features(features)

        is_identifier_column = np.array(
            [
                np.issubdtype(feature.dtype, np.integer) and self._is_sequential(feature)
                for feature in processed_features
            ]
        )
        identifier_column_indices = np.where(is_identifier_column)
        # this issue does not reflect rows at all so we set the score to 1.0 for all rows in the issue attribute
        # and set the is_identifier_column_issue to False
        num_rows = processed_features[0].size
        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": False,
                self.issue_score_key: np.ones(num_rows),
            },
        )
        # score in summary should be 1.0 if the issue is not present and 0.0 if at least one column is an identifier column
        self.summary = self.make_summary(score=1.0 - float(is_identifier_column.any()))
        # more elegant way to set the score in summary
        self.info = {
            "identifier_columns": identifier_column_indices[0].tolist(),
            "num_identifier_columns": identifier_column_indices[0].size,
        }
