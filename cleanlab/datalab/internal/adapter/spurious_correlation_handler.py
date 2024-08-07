from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from cleanlab.datalab.internal.adapter.constants import DEFAULT_CLEANVISION_ISSUES
from cleanlab.datalab.internal.adapter.constants import SPURIOUS_CORRELATION_ISSUE
from cleanlab.datalab.internal.spurious_correlation import SpuriousCorrelations


class SpuriousCorrelationHandler:
    def __init__(self, threshold: Optional[float] = None) -> None:
        if threshold is None:
            self.threshold = SPURIOUS_CORRELATION_ISSUE["spurious_correlation"]["threshold"]
        else:
            self._threshold = threshold

    @property
    def threshold(self) -> Optional[float]:
        return self._threshold

    @threshold.setter
    def threshold(self, new_threshold: Optional[float]) -> None:
        self._threshold = new_threshold

    def handle_spurious_correlations(
        self, imagelab_issues: pd.DataFrame, labels: np.ndarray
    ) -> Dict[str, Any]:
        imagelab_columns = imagelab_issues.columns.tolist()
        # Check if all vision issue scores are computed
        if not all(
            default_cleanvision_issue + "_score" in imagelab_columns
            for default_cleanvision_issue in DEFAULT_CLEANVISION_ISSUES.keys()
        ):
            raise ValueError(
                "Not all vision issue scores have been computed by find_issues() method"
            )

        score_columns = [col for col in imagelab_columns if col.endswith("_score")]
        correlations_df = SpuriousCorrelations(
            data=imagelab_issues[score_columns], labels=labels
        ).calculate_correlations()
        return {
            "correlations_df": correlations_df,
            "threshold": self.threshold,
        }
