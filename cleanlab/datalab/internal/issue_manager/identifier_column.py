from typing import ClassVar, Optional

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
        min_val, max_val = arr.min(), arr.max()
        unique_sorted = set(np.unique(np.sort(arr)).tolist())

        expected_set = set(range(min_val, max_val + 1))
        return expected_set == unique_sorted

    def _prepare_features(
        self, features: Optional[npt.NDArray | pd.DataFrame | list | dict]
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
        self, features: Optional[npt.NDArray | pd.DataFrame | list | dict], **kwargs
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
            raise ValueError(
                "features must be provided to check for identifier columns."
            )

        features = self._prepare_features(features)
        score = np.array(
            [self._is_sequential(features[:, i]) for i in range(features.shape[1])]
        )
        indices = [i for i in range(features.shape[1])]
        issue_indices = [i for i in indices if score[i]]
        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": score,
                self.issue_score_key: score,
            },
            index=indices,
        )

        self.summary = self.make_summary(score=min(sum(score), 1))
        self.info = {
            "identifier_columns": issue_indices,
            "num_identifier_columns": sum(score),
            "message": f"""There are probably {sum(score)} identifier columns in your dataset. An identifier column is a
            column i in features such that set(features[:,i]) = set(c, c+1, ..., c+n) for some
            integer c, where n = num-rows of features.""",
        }
