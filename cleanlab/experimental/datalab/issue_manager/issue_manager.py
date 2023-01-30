from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from cleanlab import Datalab


class IssueManager(ABC):
    """Base class for managing data issues of a particular type in a Datalab.

    For each example in a dataset, the IssueManager for a particular type of issue should compute:
    - A numeric severity score between 0 and 1,
        with values near 0 indicating severe instances of the issue.
    - A boolean `is_issue` value, which is True
        if we believe this example suffers from the issue in question.
      `is_issue` may be determined by thresholding the severity score
        (with an a priori determined reasonable threshold value),
        or via some other means (e.g. Confident Learning for flagging label issues).

    The IssueManager should also report:
    - A global value between 0 and 1 summarizing how severe this issue is in the dataset overall
        (e.g. the average severity across all examples in dataset
        or count of examples where `is_issue=True`).
    - Other interesting `info` about the issue and examples in the dataset,
      and statistics estimated from current dataset that may be reused
      to score this issue in future data.
      For example, `info` for label issues could contain the:
      confident_thresholds, confident_joint, predicted label for each example, etc.
      Another example is for (near)-duplicate detection issue, where `info` could contain:
      which set of examples in the dataset are all (nearly) identical.

    Implementing a new IssueManager:
    - Define the `issue_name` class attribute, e.g. "label", "duplicate", "outlier", etc.
    - Implement the abstract methods `find_issues` and `collect_info`.
      - `find_issues` is responsible for computing computing the `issues` and `summary` dataframes.
      - `collect_info` is responsible for computing the `info` dict. It is called by `find_issues`,
        once the manager has set the `issues` and `summary` dataframes as instance attributes.
    """

    issue_name: str
    """Returns a key that is used to store issue summary results about the assigned Lab."""

    def __init__(self, datalab: Datalab):
        self.datalab = datalab
        self.info: Dict[str, Any] = {}
        self.issues: pd.DataFrame = pd.DataFrame()
        self.summary: pd.DataFrame = pd.DataFrame()
        # TODO: Split info into two attributes: "local" info and "global" statistics (should be checked at the start of `find_issues`, but overwritten by `collect_info`).

    def __repr__(self):
        class_name = self.__class__.__name__
        return class_name

    @classmethod
    def __init_subclass__(cls):
        required_class_variables = [
            "issue_name",
        ]
        for var in required_class_variables:
            if not hasattr(cls, var):
                raise NotImplementedError(f"Class {cls.__name__} must define class variable {var}")

    @property
    def issue_score_key(self) -> str:
        """Returns a key that is used to store issue score results about the assigned Lab."""
        # TODO: The score key should just be f"{self.issue_name}_score" or f"{self.issue_name}_quality_score", f"{self.issue_name}_quality"
        return f"{self.issue_name}_score"

    @abstractmethod
    def find_issues(self, *args, **kwargs) -> None:
        """Finds occurrences of this particular issue in the dataset.

        Computes the `issues` and `summary` dataframes. Calls `collect_info` to compute the `info` dict.
        """
        raise NotImplementedError

    def collect_info(self, *args, **kwargs) -> dict:
        """Collects data for the info attribute of the Datalab.

        NOTE
        ----
        This method is called by `find_issues` after `find_issues` has set the `issues` and `summary` dataframes
        as instance attributes.
        """
        raise NotImplementedError

    # TODO: Add a `collect_global_info` method for storing useful statistics that can be used by other IssueManagers.

    def get_summary(self, score: float) -> pd.DataFrame:
        """Sets the summary attribute of this IssueManager.

        Parameters
        ----------
        score :
            The overall score for this issue.
        """
        return pd.DataFrame(
            {
                "issue_type": [self.issue_name],
                "score": [score],
            },
        )

    @property
    def verbosity_levels(self) -> Dict[int, Dict[str, List[str]]]:
        """Returns a dictionary of verbosity levels and their corresponding dictionaries of
        report items to print.

        Example
        -------

        >>> verbosity_levels = {
        ...     0: {},
        ...     1: {"info": ["some_info_key"]},
        ...     2: {
        ...         "info": ["additional_info_key"],
        ...         "issues": ["issue_column_1", "issue_column_2"],
        ...     },
        ... }

        Returns
        -------
        verbosity_levels :
            A dictionary of verbosity levels and their corresponding dictionaries of
            report items to print.
        """
        return {
            0: {},
            1: {},
            2: {},
            3: {},
        }

    def report(self, k: int = 5, verbosity: int = 0) -> str:
        import json

        top_level = max(self.verbosity_levels.keys()) + 1
        if verbosity not in list(self.verbosity_levels.keys()) + [top_level]:
            raise ValueError(
                f"Verbosity level {verbosity} not supported. "
                f"Supported levels: {self.verbosity_levels.keys()}"
                f"Use verbosity={top_level} to print all info."
            )
        if self.issues.empty:
            print(f"No issues found")

        topk_ids = self.issues.sort_values(by=self.issue_score_key, ascending=True).index[:k]

        report_str = f"{self.issue_name:-^80}\n\n"

        report_str += f"Score: {self.summary.loc[0, 'score']:.4f}\n\n"

        columns = {}
        info_to_omit = set()
        for level, verbosity_dict in self.verbosity_levels.items():
            if level <= verbosity:
                for key, values in verbosity_dict.items():
                    if key == "info":
                        info_to_omit.update(values)
                    elif key == "issue":
                        # Add the issue-specific info, with the top k ids
                        new_columns = {
                            col: np.array(self.info[col])[topk_ids]
                            for col in values
                            if self.info.get(col, None) is not None
                        }
                        columns.update(new_columns)
                        info_to_omit.update(values)

        if verbosity == max(self.verbosity_levels.keys()) + 1:
            info_to_omit = set()
            for verbosity_dict in self.verbosity_levels.values():
                info_to_omit.update(verbosity_dict.get("issue", []))

        report_str += self.issues.loc[topk_ids].copy().assign(**columns).to_string()

        # Dump the info dict, omitting the info that has already been printed
        info_to_print = {key: value for key, value in self.info.items() if key not in info_to_omit}

        def truncate(s, max_len=4) -> str:
            if hasattr(s, "shape") or hasattr(s, "ndim"):
                s = np.array(s)
                if s.ndim > 1:
                    description = f"array of shape {s.shape}\n"
                    with np.printoptions(threshold=max_len):
                        if s.ndim == 2:
                            description += f"{s}"
                        if s.ndim > 2:
                            description += f"{s}"
                    return description
                s = s.tolist()

            if isinstance(s, list):
                if all([isinstance(s_, list) for s_ in s]):
                    return truncate(np.array(s, dtype=object), max_len=max_len)
                if len(s) > max_len:
                    s = s[:max_len] + ["..."]
            return str(s)

        if info_to_print:
            # Print the info dict, truncating arrays to 4 elements,
            report_str += f"\n\nInfo: "
            for key, value in info_to_print.items():
                if isinstance(value, dict):
                    report_str += f"\n{key}:\n{json.dumps(value, indent=4)}"
                elif isinstance(value, pd.DataFrame):
                    max_rows = 5
                    df_str = value.head(max_rows).to_string()
                    if len(value) > max_rows:
                        df_str += f"\n... (total {len(value)} rows)"
                    report_str += f"\n{key}:\n{df_str}"
                else:
                    report_str += f"\n{key}: {truncate(value)}"
        return report_str
