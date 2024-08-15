from typing import ClassVar, Optional, Union

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

    def _prepare_features(
        self, features: Optional[Union[npt.NDArray, pd.DataFrame, list, dict]]
    ) -> npt.NDArray:
        """
        Prepare the features for issue check.

        Args:
            features: The input features.

        Returns:
            features: features as npt.NDArray
        """
        if isinstance(features, np.ndarray):
            return features.T
        elif isinstance(features, pd.DataFrame) or isinstance(features, dict):
            feature_list = list()
            for _, feature_value in features.items():
                feature_list.append(np.array(feature_value))
            return feature_list
        elif isinstance(features, list):
            for col_list in features:
                if not isinstance(col_list, list):
                    raise ValueError("features must be a list of lists if it features is a list.")
                feature_list = [np.array(col_list) for col_list in features]
            return (
                feature_list  # don't need to transpose, format needs to be a list of column lists
            )
        else:
            raise ValueError(
                "features must be a numpy array or a pandas DataFrame. or list\
                    or dict that can be converted to a numpy array."
            )

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

        scores = np.array(
            [
                np.issubdtype(feature.dtype, np.integer) and self._is_sequential(feature)
                for feature in processed_features
            ]
        )
        issue_indices = np.where(scores)
        # this issue does not reflect rows at all so we set the score to 1.0 for all rows in the issue attribute
        # and set the is_identifier_column_issue to False
        print(f"shape of features raw: {features.shape}")
        print("features: ", features)
        print(f"processed_features: {processed_features}")
        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": False,
                self.issue_score_key: [1.0 for _ in range(features.shape[0])],
            },
        )
        print(f"self.issues: {self.issues}")
        # score in summary should be 1.0 if the issue is not presend and 0.0 if at least one column is an identifier column
        self.summary = self.make_summary(score=1.0 if scores.sum() == 0 else 0.0)
        self.info = {
            "identifier_columns": issue_indices,
            "num_identifier_columns": scores.sum(),
        }
