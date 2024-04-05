# Copyright (C) 2017-2024  Cleanlab Inc.
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
"""
The experimental feature that construct a datalab instance with the statistic information from a trained datalab.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from dataclasses import InitVar, dataclass, KW_ONLY

import pandas as pd
import numpy as np
from datasets import Dataset

from cleanlab.datalab.datalab import Datalab
from cleanlab.experimental.label_issues_batched import LabelInspector
from cleanlab.rank import find_top_issues

if TYPE_CHECKING:  # pragma: no cover
    from datasets.arrow_dataset import Dataset
    from scipy.sparse import csr_matrix

    DatasetLike = Union[Dataset, pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], str]
    Info = Dict[str, Dict[str, Any]]


class UnimplementedFeatureError(NotImplementedError):
    pass


@dataclass
class FindIssuesKwargs:
    """
    A class that represents the keyword arguments for finding issues in data.

    Parameters
    ----------
    labels :
        A numpy array representing the labels.
    pred_probs :
        A numpy array representing the predicted probabilities.
    _label_map :
        An optional dictionary representing the label map.
    features :
        An optional numpy array representing the features.
    knn_graph :
        An optional scipy sparse matrix representing the k-nearest neighbors graph.
    """

    _: KW_ONLY
    labels: np.ndarray
    pred_probs: np.ndarray
    _label_map: InitVar[Optional[Dict[int, str]]] = None
    features: InitVar[Optional[np.ndarray]] = None
    knn_graph: InitVar[Optional[csr_matrix]] = None

    def __post_init__(self, _label_map, features, knn_graph):
        """
        Performs post-initialization operations.

        Parameters
        ----------
        _label_map :
            An optional dictionary representing the label map.
        features :
            An optional numpy array representing the features.
            If not None, then an UnimplementedFeatureError is raised, as the DataMonitor will only support labels and pred_probs for now.
        knn_graph :
            An optional scipy sparse matrix representing the k-nearest neighbors graph.
            If not None, then an UnimplementedFeatureError is raised, as the DataMonitor will only support labels and pred_probs for now.

        Raises
        ------
        UnimplementedFeatureError :
            If any unimplemented keyword arguments are provided.
        """
        self._check_unimplemented_kwargs(features, knn_graph)
        if self.labels is not None and _label_map is not None:
            self.labels = np.vectorize(_label_map.get, otypes=[int])(self.labels)

    def _check_unimplemented_kwargs(self, features, knn_graph):
        unimplemented_kwargs = {
            "features": features,
            "knn_graph": knn_graph,
        }
        unimplemented_kwargs = {k: v for k, v in unimplemented_kwargs.items() if v is not None}
        if unimplemented_kwargs:
            raise UnimplementedFeatureError(
                f"The following arguments are not supported in this version of DataMonitor: {list(unimplemented_kwargs.keys())}"
            )


class DataMonitor:
    """
    An object that can be used to audit new data using the statistics from a fitted Datalab instance.

    Parameters
    ----------
    datalab : Datalab
        The Datalab object fitted to the original training dataset.
    """

    def __init__(self, datalab: Datalab):
        if str(datalab.task) != "classification":
            raise NotImplementedError(
                f"Currently, only classification tasks are supported for DataMonitor."
                f' The task of the provided Datalab instance is "{str(datalab.task)}", which is not supported by DataMonitor.'
            )

        self.label_map = datalab._label_map

        self.info = datalab.get_info()
        # lab.get_info() is an alias for lab.info, but some keys are handled differently via lab.get_info(key) method.
        _missing_label_info_keys = set(datalab.get_info("label").keys()) - set(self.info.keys())
        self.info["label"].update(
            {k: v for (k, v) in datalab.get_info("label").items() if k in _missing_label_info_keys}
        )

        self.monitors: Dict[str, IssueMonitor] = {
            "label": LabelIssueMonitor(self.info),
        }

        issue_names = self.monitors.keys()

        # This issue dictionary will collect the issues for the entire stream of data.
        self.issues_dict = {
            col: []
            for cols in zip(
                [f"is_{name}_issue" for name in issue_names],
                [f"{name}_score" for name in issue_names],
            )
            for col in cols
        }
        self._last_printed_lines = 0

    @property
    def issues(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.issues_dict)

    @property
    def issue_summary(self) -> pd.DataFrame:
        issue_summary_dict = {}
        issue_names = self.monitors.keys()
        issue_summary_dict["issue_type"] = issue_names
        issue_summary_dict["num_issues"] = [
            np.sum(self.issues_dict[f"is_{issue_name}_issue"]) for issue_name in issue_names
        ]
        return pd.DataFrame.from_dict(issue_summary_dict)

    def find_issues(self, *, labels: np.ndarray, pred_probs: np.ndarray) -> None:
        str_to_int_map = {v: k for (k, v) in self.label_map.items()}
        find_issues_kwargs = FindIssuesKwargs(
            labels=labels,
            pred_probs=pred_probs,
            _label_map=str_to_int_map,
        )
        issues_dict = {k: [] for k in self.issues_dict.keys()}

        # Flag to track if any monitor has found issues
        display_results = False
        for issue_name, issue_monitor in self.monitors.items():
            issue_monitor.find_issues(find_issues_kwargs)

            # Update issues_dict based on the current monitor's findings for the current batch
            issues_dict[f"is_{issue_name}_issue"] = issue_monitor.issues_dict["is_issue"]
            issues_dict[f"{issue_name}_score"] = issue_monitor.issues_dict["score"]

            if issue_monitor.batch_has_issues:
                display_results = True

            # Clear the current monitor's issues dictionary immediately after processing
            issue_monitor.clear_issues_dict()

        if display_results:
            self._display_batch_issues(issues_dict)

        # Append the issues to the existing issues dictionary
        for k, v in issues_dict.items():
            self.issues_dict[k].extend(v)

    def _display_batch_issues(self, issues_dicts: Dict[str, np.ndarray]) -> None:
        start_index = len(next(iter(self.issues_dict.values())))
        end_index = start_index + len(next(iter(issues_dicts.values())))
        index = np.arange(start_index, end_index)
        df_issues = pd.DataFrame(issues_dicts, index=index)
        # Note: Maybe filter out the rows where all the "is_X_issue" columns are False.
        is_issue_columns = [
            col for col in df_issues.columns if (col.startswith("is_") and col.endswith("_issue"))
        ]
        print(
            "Detected issues in the current batch:\n",
            (
                df_issues.query(
                    f"{' | '.join([f'{col} == True' for col in is_issue_columns])}"
                ).to_string()
            ),
            "\n",
        )


class IssueMonitor(ABC):
    def __init__(self, info: Info):
        self.info = info
        # This issue dictionary will collect the issues for a single batch of data, then be manually cleared.
        self.issues_dict = {
            "is_issue": [],
            "score": [],
        }

    def clear_issues_dict(self):
        self.issues_dict["is_issue"] = []
        self.issues_dict["score"] = []

    @abstractmethod
    def find_issues(self, fi_kwargs: FindIssuesKwargs) -> None:
        pass

    @property
    def batch_has_issues(self) -> bool:
        return any(self.issues_dict["is_issue"])


class LabelIssueMonitor(IssueMonitor):
    def __init__(self, info: Info):
        super().__init__(info)
        label_info = self.info.get("label")
        if label_info is None:
            raise ValueError("The label information is missing in the info dictionary.")

        confident_thresholds = label_info.get("confident_thresholds")
        if confident_thresholds is None:
            raise ValueError("The confident thresholds are missing in the info dictionary.")

        self.inspector = self._setup_label_inspector(confident_thresholds)
        self._total_num_issues = 0
        self._found_issues_in_batch = False

    def _setup_label_inspector(self, confident_thresholds: List[float]) -> LabelInspector:
        inspector = LabelInspector(num_class=len(confident_thresholds), store_results=False)
        # The LabelInspector cannot configure the thresholds during initialization, so we set them manually here.
        inspector.confident_thresholds = np.array(confident_thresholds)

        # The LabelInspector cannot configure the examples_processed_thresh during initialization, so we set it manually here.
        inspector.examples_processed_thresh = 1
        return inspector

    def find_issues(self, fi_kwargs: FindIssuesKwargs) -> None:
        """Identifies and records label issues in a batch of data.

        Parameters
        ----------
        fi_kwargs :
            An instance of FindIssuesKwargs containing the labels and predicted probabilities.

        Raises
        ------
        ValueError :
            If either the labels or predicted probabilities are not provided (i.e., None).
        """
        # Validate input parameters
        if fi_kwargs.labels is None or fi_kwargs.pred_probs is None:
            raise ValueError("Both labels and pred_probs must be provided to find issues.")

        # Initial setup
        num_examples = len(fi_kwargs.labels)
        self._found_issues_in_batch = (
            False  # Reset the flag indicating issues found in the current batch
        )

        # Score label quality and update total number of issues
        scores = self.inspector.score_label_quality(fi_kwargs.labels, fi_kwargs.pred_probs)
        new_total_num_issues = self.inspector.get_num_issues(silent=True)
        num_issues_in_batch = new_total_num_issues - self._total_num_issues

        # Update class state
        self._total_num_issues = new_total_num_issues

        # Determine which examples have issues
        is_issue_array = np.zeros(num_examples, dtype=bool)
        if num_issues_in_batch > 0:
            self._found_issues_in_batch = True
            issues_indices = find_top_issues(scores, top=num_issues_in_batch)
            is_issue_array[issues_indices] = True

        # Update issues dictionary
        self.issues_dict = {
            "is_issue": is_issue_array,
            "score": scores,
        }
