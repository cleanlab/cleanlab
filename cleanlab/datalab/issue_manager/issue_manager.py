# Copyright (C) 2017-2023  Cleanlab Inc.
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with cleanlab.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from itertools import chain
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Set, Tuple, Type, TypeVar
import json

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from cleanlab.datalab.datalab import Datalab


T = TypeVar("T", bound="IssueManager")
TM = TypeVar("TM", bound="IssueManagerMeta")


class IssueManagerMeta(ABCMeta):
    """Metaclass for IssueManager that adds issue_score_key to the class.

    :meta private:
    """

    issue_name: ClassVar[str]
    issue_score_key: ClassVar[str]
    verbosity_levels: ClassVar[Dict[int, List[str]]] = {
        0: [],
        1: [],
        2: [],
        3: [],
    }

    def __new__(
        meta: Type[TM],
        name: str,
        bases: Tuple[Type[Any], ...],
        class_dict: Dict[str, Any],
    ) -> TM:  # Classes that inherit from ABC don't need to be modified
        if ABC in bases:
            return super().__new__(meta, name, bases, class_dict)

        # Ensure that the verbosity levels don't have keys other than those in ["issue", "info"]
        verbosity_levels = class_dict.get("verbosity_levels", meta.verbosity_levels)
        for level, level_list in verbosity_levels.items():
            if not isinstance(level_list, list):
                raise ValueError(
                    f"Verbosity levels must be lists. "
                    f"Got {level_list} in {name}.verbosity_levels"
                )
            prohibited_keys = [key for key in level_list if not isinstance(key, str)]
            if prohibited_keys:
                raise ValueError(
                    f"Verbosity levels must be lists of strings. "
                    f"Got {prohibited_keys} in {name}.verbosity_levels[{level}]"
                )

        # Concrete classes need to have an issue_name attribute
        if "issue_name" not in class_dict:
            raise TypeError("IssueManagers need an issue_name class variable")

        # Add issue_score_key to class
        class_dict["issue_score_key"] = f"{class_dict['issue_name']}_score"
        return super().__new__(meta, name, bases, class_dict)


class IssueManager(ABC, metaclass=IssueManagerMeta):
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

    description: ClassVar[str] = ""
    """Short text that summarizes the type of issues handled by this IssueManager.

    :meta hide-value:
    """
    issue_name: ClassVar[str]
    """Returns a key that is used to store issue summary results about the assigned Lab."""
    issue_score_key: ClassVar[str]
    """Returns a key that is used to store issue score results about the assigned Lab."""
    verbosity_levels: ClassVar[Dict[int, List[str]]] = {
        0: [],
        1: [],
        2: [],
        3: [],
    }
    """A dictionary of verbosity levels and their corresponding dictionaries of
    report items to print.

    :meta hide-value:

    Example
    -------

    >>> verbosity_levels = {
    ...     0: [],
    ...     1: ["some_info_key"],
    ...     2: ["additional_info_key"],
    ... }
    """

    def __init__(self, datalab: Datalab, **_):
        self.datalab = datalab
        self.info: Dict[str, Any] = {}
        self.issues: pd.DataFrame = pd.DataFrame()
        self.summary: pd.DataFrame = pd.DataFrame()

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
        This method is called by :py:meth:`find_issues` after :py:meth:`find_issues` has set the `issues` and `summary` dataframes
        as instance attributes.
        """
        raise NotImplementedError

    @classmethod
    def make_summary(cls, score: float) -> pd.DataFrame:
        """Construct a summary dataframe.

        Parameters
        ----------
        score :
            The overall score for this issue.

        Returns
        -------
        summary :
            A summary dataframe.
        """
        if not 0 <= score <= 1:
            raise ValueError(f"Score must be between 0 and 1. Got {score}.")

        return pd.DataFrame(
            {
                "issue_type": [cls.issue_name],
                "score": [score],
            },
        )

    @classmethod
    def report(
        cls,
        issues: pd.DataFrame,
        summary: pd.DataFrame,
        info: Dict[str, Any],
        num_examples: int = 5,
        verbosity: int = 0,
        include_description: bool = False,
        info_to_omit: Optional[List[str]] = None,
    ) -> str:
        """Compose a report of the issues found by this IssueManager.

        Parameters
        ----------
        issues :
            An issues dataframe.

            Example
            -------
            >>> import pandas as pd
            >>> issues = pd.DataFrame(
            ...     {
            ...         "is_X_issue": [True, False, True],
            ...         "X_score": [0.2, 0.9, 0.4],
            ...     },
            ... )

        summary :
            The summary dataframe.

            Example
            -------
            >>> summary = pd.DataFrame(
            ...     {
            ...         "issue_type": ["X"],
            ...         "score": [0.5],
            ...     },
            ... )

        info :
            The info dict.

            Example
            -------
            >>> info = {
            ...     "A": "val_A",
            ...     "B": ["val_B1", "val_B2"],
            ... }

        num_examples :
            The number of examples to print.

        verbosity :
            The verbosity level of the report.

        include_description :
            Whether to include a description of the issue in the report.

        Returns
        -------
        report_str :
            A string containing the report.
        """

        max_verbosity = max(cls.verbosity_levels.keys())
        top_level = max_verbosity + 1
        if verbosity not in list(cls.verbosity_levels.keys()) + [top_level]:
            raise ValueError(
                f"Verbosity level {verbosity} not supported. "
                f"Supported levels: {cls.verbosity_levels.keys()}"
                f"Use verbosity={top_level} to print all info."
            )
        if issues.empty:
            print(f"No issues found")

        topk_ids = issues.sort_values(by=cls.issue_score_key, ascending=True).index[:num_examples]

        score = summary["score"].loc[0]
        report_str = f"{' ' + cls.issue_name + ' issues ':-^60}\n\n"

        if include_description and cls.description:
            description = cls.description
            if verbosity == 0:
                description = description.split("\n\n", maxsplit=1)[0]
            report_str += "About this issue:\n\t" + description + "\n\n"
        report_str += (
            f"Number of examples with this issue: {issues[f'is_{cls.issue_name}_issue'].sum()}\n"
            f"Overall dataset quality in terms of this issue: {score:.4f}\n\n"
        )

        info_to_print: Set[str] = set()
        _info_to_omit = set(issues.columns).union(info_to_omit or [])
        verbosity_levels_values = chain.from_iterable(
            list(cls.verbosity_levels.values())[: verbosity + 1]
        )
        info_to_print.update(set(verbosity_levels_values) - _info_to_omit)
        if verbosity == top_level:
            info_to_print.update(set(info.keys()) - _info_to_omit)

        report_str += "Examples representing most severe instances of this issue:\n"
        report_str += issues.loc[topk_ids].to_string()

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
            info_to_print_dict = {key: info[key] for key in info_to_print}
            # Print the info dict, truncating arrays to 4 elements,
            report_str += f"\n\nAdditional Information: "
            for key, value in info_to_print_dict.items():
                if key == "statistics":
                    continue
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
