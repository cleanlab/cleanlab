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
            return features
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()
            return features
        if isinstance(features, list):
            features = np.array(features)
            return features
        if isinstance(features, dict):
            df = pd.DataFrame(features)
            features = df.to_numpy()
            return features
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

        features = self._prepare_features(features)
        scores = np.array(
            [self._is_sequential(features[:, col]) for col in range(features.shape[1])]
        )
        issue_indices = np.where(scores)
        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": scores,
                self.issue_score_key: scores,
            },
        )

        self.summary = self.make_summary(score=min(scores.sum(), 1))
        self.info = {
            "identifier_columns": issue_indices,
            "num_identifier_columns": scores.sum(),
        }
