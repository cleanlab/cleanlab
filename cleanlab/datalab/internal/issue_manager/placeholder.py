from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cleanlab.datalab.internal.issue_manager import IssueManager

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

# Minimum rows with non-null values required to analyze a column.
_MIN_ROWS_FOR_ANALYSIS = 20
# Subsample size for frequency-based candidate detection.
_SUBSAMPLE_SIZE = 5000
# Candidate must appear at least this many times and this fraction of non-null rows.
_MIN_CANDIDATE_COUNT = 3
_MIN_CANDIDATE_FRACTION = 0.01
# Minimum non-candidate values needed for distribution comparisons.
_MIN_NON_CANDIDATE_VALUES = 10
# Negative sentinel in a mostly non-negative column.
_SIGN_MISMATCH_NON_NEGATIVE_FRACTION = 0.8
# Distance from bulk distribution in units of MAD (or IQR if MAD is ~0).
_MAD_MULTIPLIER = 5.0


class PlaceholderIssueManager(IssueManager):
    """Manages issues related to placeholder values in numeric feature columns.

    Parameters
    ----------
    datalab :
        The Datalab instance that this issue manager searches for issues in.
    """

    description: ClassVar[
        str
    ] = """Examples identified with the placeholder issue correspond to rows that contain
        numeric placeholder values (e.g. -99) in feature columns where such values likely
        represent missing data rather than legitimate measurements.
        """
    issue_name: ClassVar[str] = "placeholder"
    verbosity_levels = {
        0: [],
        1: [],
        2: ["placeholder_by_column"],
    }

    @staticmethod
    def _numeric_column_indices(features: pd.DataFrame) -> List[int]:
        return [
            i
            for i, col in enumerate(features.columns)
            if pd.api.types.is_numeric_dtype(features[col])
            and not pd.api.types.is_bool_dtype(features[col])
        ]

    @staticmethod
    def _column_label(
        column_index: int, column_names: Optional[List[str]]
    ) -> Union[str, int]:
        if column_names is not None and column_index < len(column_names):
            return column_names[column_index]
        return column_index

    @staticmethod
    def _values_match(candidate: float, values: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
        if np.issubdtype(values.dtype, np.floating):
            return np.isclose(values, candidate, rtol=0.0, atol=0.0, equal_nan=False)
        return values == candidate

    @staticmethod
    def _is_placeholder_like(candidate: float, others: npt.NDArray[np.floating]) -> bool:
        if others.size < _MIN_NON_CANDIDATE_VALUES:
            return False

        if candidate < 0 and np.mean(others >= 0) >= _SIGN_MISMATCH_NON_NEGATIVE_FRACTION:
            return True

        median = float(np.median(others))
        mad = float(np.median(np.abs(others - median)))
        if mad < 1e-12:
            iqr = float(np.percentile(others, 75) - np.percentile(others, 25))
            if iqr > 1e-12:
                return abs(candidate - median) > _MAD_MULTIPLIER * iqr
            return False

        return abs(candidate - median) > _MAD_MULTIPLIER * mad

    @classmethod
    def _find_confirmed_placeholders(
        cls, column_values: npt.NDArray[np.floating]
    ) -> List[float]:
        """Return placeholder values confirmed for a single numeric column."""
        non_null = column_values[~np.isnan(column_values)]
        if non_null.size < _MIN_ROWS_FOR_ANALYSIS:
            return []

        iqr = float(np.percentile(non_null, 75) - np.percentile(non_null, 25))
        if iqr < 1e-12:
            return []

        if non_null.size > _SUBSAMPLE_SIZE:
            rng = np.random.default_rng(0)
            sample = rng.choice(non_null, size=_SUBSAMPLE_SIZE, replace=False)
        else:
            sample = non_null

        # Compare frequencies relative to the sample (not the full column) so the
        # threshold has the same meaning regardless of whether subsampling kicked in.
        min_count = max(_MIN_CANDIDATE_COUNT, int(_MIN_CANDIDATE_FRACTION * sample.size))
        unique_values, counts = np.unique(sample, return_counts=True)

        confirmed: List[float] = []
        for candidate, count in zip(unique_values, counts):
            if count < min_count:
                continue
            if count / sample.size < _MIN_CANDIDATE_FRACTION:
                continue

            others = non_null[~cls._values_match(candidate, non_null)]
            if cls._is_placeholder_like(float(candidate), others.astype(float)):
                confirmed.append(float(candidate))

        return confirmed

    @classmethod
    def _calculate_placeholder_issues(
        cls,
        features: npt.NDArray[Any],
        numeric_column_indices: List[int],
        column_names: Optional[List[str]] = None,
    ) -> Tuple[
        npt.NDArray[np.bool_],
        npt.NDArray[np.float64],
        npt.NDArray[np.bool_],
        Dict[str, Any],
    ]:
        n_rows = features.shape[0]
        n_numeric_cols = len(numeric_column_indices)
        placeholder_tracker = np.zeros((n_rows, n_numeric_cols), dtype=bool)
        placeholder_by_column: Dict[str, List[float]] = {}

        if n_numeric_cols == 0:
            scores = np.ones(n_rows, dtype=float)
            is_placeholder_issue = np.zeros(n_rows, dtype=bool)
            return is_placeholder_issue, scores, placeholder_tracker, {
                "placeholder_by_column": placeholder_by_column,
            }

        for tracker_col, feature_col in enumerate(numeric_column_indices):
            column_values = features[:, feature_col].astype(float, copy=False)
            confirmed = cls._find_confirmed_placeholders(column_values)
            if not confirmed:
                continue

            column_label = str(cls._column_label(feature_col, column_names))
            placeholder_by_column[column_label] = confirmed

            column_mask = np.zeros(n_rows, dtype=bool)
            for value in confirmed:
                column_mask |= cls._values_match(value, column_values)
            placeholder_tracker[:, tracker_col] = column_mask

        placeholder_counts = placeholder_tracker.sum(axis=1)
        scores = 1.0 - (placeholder_counts / n_numeric_cols)
        is_placeholder_issue = placeholder_counts > 0

        return (
            is_placeholder_issue,
            scores.astype(np.float64),
            placeholder_tracker,
            {"placeholder_by_column": placeholder_by_column},
        )

    def find_issues(
        self,
        features: Optional[npt.NDArray | pd.DataFrame] = None,
        **kwargs,
    ) -> None:
        if features is None:
            raise ValueError("features must be provided to check for placeholder values.")

        column_names: Optional[List[str]] = None
        if isinstance(features, pd.DataFrame):
            column_names = list(features.columns)
            numeric_column_indices = self._numeric_column_indices(features)
            features_array = features.to_numpy()
        else:
            features_array = np.asarray(features)
            numeric_column_indices = list(range(features_array.shape[1]))

        (
            is_placeholder_issue,
            scores,
            placeholder_tracker,
            detection_info,
        ) = self._calculate_placeholder_issues(
            features=features_array,
            numeric_column_indices=numeric_column_indices,
            column_names=column_names,
        )

        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": is_placeholder_issue,
                self.issue_score_key: scores,
            },
        )

        self.summary = self.make_summary(score=float(scores.mean()))
        self.info = self.collect_info(
            placeholder_tracker=placeholder_tracker,
            numeric_column_indices=numeric_column_indices,
            column_names=column_names,
            detection_info=detection_info,
        )

    @staticmethod
    def _column_impact(placeholder_tracker: np.ndarray) -> Dict[str, List[float]]:
        if placeholder_tracker.size == 0:
            return {"column_impact": []}
        return {"column_impact": placeholder_tracker.mean(axis=0).tolist()}

    def collect_info(
        self,
        placeholder_tracker: np.ndarray,
        numeric_column_indices: List[int],
        column_names: Optional[List[str]],
        detection_info: Dict[str, Any],
    ) -> dict:
        column_impact = self._column_impact(placeholder_tracker=placeholder_tracker)
        average_placeholder_score = {
            "average_placeholder_score": float(self.issues[self.issue_score_key].mean())
        }
        return {
            **average_placeholder_score,
            **detection_info,
            **column_impact,
        }
