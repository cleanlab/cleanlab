from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional

import numpy as np
import pandas as pd

from cleanlab.datalab.internal.issue_manager import IssueManager

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from cleanlab.datalab.datalab import Datalab


class NullIssueManager(IssueManager):
    """Manages issues related to null/missing values in the rows of features.
    Parameters
    ----------
    datalab :
        The Datalab instance that this issue manager searches for issues in.
    """

    description: ClassVar[
        str
    ] = """Whether the dataset has any missing/null values
        """
    issue_name: ClassVar[str] = "null"
    verbosity_levels = {
        0: ["average_null_score"],
        1: [],
        2: [],
    }

    def __init__(self, datalab: Datalab):
        super().__init__(datalab)

    def find_issues(
        self,
        features: Optional[npt.NDArray] = None,
        **kwargs,
    ) -> None:
        if features is None:
            raise ValueError("features must be provided to check for null values.")
        rows = features.shape[0]
        cols = features.shape[1]
        scores = np.zeros(rows).astype(np.float32)
        is_null_issue = np.full(rows, False)
        null_tracker = np.isnan(features)
        if null_tracker.any():
            for row in range(rows):
                if null_tracker[row].any():
                    is_null_issue[row] = True
                    null_row_count = np.count_nonzero(null_tracker[row])
                    scores[row] = null_row_count / cols

        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": is_null_issue,
                self.issue_score_key: scores,
            },
        )

        self.summary = self.make_summary(score=scores.mean())
        self.info = self.collect_info()

    def collect_info(self) -> dict:
        issues_dict = {"average_null_score": self.issues[self.issue_score_key].mean()}
        info_dict: Dict[str, Any] = {**issues_dict}
        return info_dict
