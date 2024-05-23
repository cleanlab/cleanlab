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
from dataclasses import InitVar, dataclass

import pandas as pd
import numpy as np

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

    labels: Optional[np.ndarray] = None
    pred_probs: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None
    _label_map: InitVar[Optional[Dict[int, str]]] = None
    knn_graph: InitVar[Optional[csr_matrix]] = None

    def __post_init__(self, _label_map, knn_graph):
        """
        Performs post-initialization operations.

        Parameters
        ----------
        _label_map :
            An optional dictionary representing the label map.
        knn_graph :
            An optional scipy sparse matrix representing the k-nearest neighbors graph.
            If not None, then an UnimplementedFeatureError is raised, as the DataMonitor will only support labels, pred_probs and features for now.

        Raises
        ------
        UnimplementedFeatureError :
            If any unimplemented keyword arguments are provided.
        """
        self._check_unimplemented_kwargs(knn_graph)
        if self.labels is not None and _label_map is not None:
            self.labels = np.vectorize(_label_map.get, otypes=[int])(self.labels)

        def _adapt_to_singletons(self):
            # TODO: Implement this method to adapt the input to singletons.
            # For instance, single data points could be passed directly as scalar values or single-element arrays,
            # and batches could be passed as lists or arrays.
            pass

    def _check_unimplemented_kwargs(self, knn_graph):
        unimplemented_kwargs = {
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
    datalab :
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
        datalab_issue_summary = datalab.get_issue_summary()
        issue_names_checked = datalab_issue_summary["issue_type"].tolist()
        for issue_name in issue_names_checked:
            # lab.get_info() is an alias for lab.info, but some keys are handled differently via lab.get_info(key) method.
            _missing_keys = set(datalab.get_info(issue_name).keys()) - set(self.info.keys())
            self.info[issue_name].update(
                {k: v for (k, v) in datalab.get_info(issue_name).items() if k in _missing_keys}
            )

        # TODO: Compare monitors and the issue types that Datalab managed to check. Print types that DataMonitor won't consider.
        # TODO: If label issues were checked by Datalab, but with features, then the monitor will skip the label issue check, explaining that it won't support that argument for now. Generalize this for all issue types.
        # TODO: Fail on issue types that DataMonitor is asked to check, but Datalab didn't check.

        # Set up monitors
        self.monitors: Dict[str, IssueMonitor] = {}

        # Helper lmabda to figure out what inputs where used for the issue checks in Datalab
        _check_issue_input = (
            lambda n, i: self.info.get(n, {}).get("find_issues_inputs", {}).get(i, False)
        )

        # Only consider label error detection if checked by Datalab, using pred_probs
        label_issue_checked = "label" in issue_names_checked
        pred_probs_checked_for_label = _check_issue_input("label", "pred_probs")

        if label_issue_checked and pred_probs_checked_for_label:
            self.monitors["label"] = LabelIssueMonitor(self.info)

        # Only consider outlier detection if checked by Datalab, using features
        outliers_checked = "outlier" in issue_names_checked
        features_checked_for_outlier = _check_issue_input("outlier", "features")

        if outliers_checked and features_checked_for_outlier:
            self.monitors["outlier"] = OutlierIssueMonitor(self.info)

        if not self.monitors:
            if issue_names_checked:
                # No monitors were created, so we can't check for any issues.
                raise ValueError("No issue types checked by Datalab are supported by DataMonitor.")
            # Datalab didn't check any issues with the expected inputs, so we can't check for any issues.
            error_msg = (
                "No issue types checked by Datalab. DataMonitor requires at least one issue type to be checked."
                " The following issue types are supported by DataMonitor: label, outlier."
            )
            raise ValueError(error_msg)

        # Set up dictionary of lists for efficiently constructing the issues DataFrame
        issue_names = self.monitors.keys()

        # This issue dictionary will collect the issues for the entire stream of data.
        self.issues_dict: Dict[str, Union[List[bool], List[float]]] = {
            col: []
            for cols in zip(
                [f"is_{name}_issue" for name in issue_names],
                [f"{name}_score" for name in issue_names],
            )
            for col in cols
        }

    @property
    def issues(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.issues_dict)

    @property
    def issue_summary(self) -> pd.DataFrame:
        issue_summary_dict: Dict[str, Union[List[str], List[int], List[float]]] = {
            "issue_type": [],
            "num_issues": [],
            "score": [],
        }
        issue_names = self.monitors.keys()
        issue_summary_dict["issue_type"] = list(issue_names)
        issue_summary_dict["num_issues"] = [
            np.sum(self.issues_dict[f"is_{issue_name}_issue"]) for issue_name in issue_names
        ]
        issue_summary_dict["score"] = [
            float(np.mean(self.issues_dict[f"{issue_name}_score"])) for issue_name in issue_names
        ]
        return pd.DataFrame.from_dict(issue_summary_dict)

    def find_issues(
        self,
        *,
        labels: Optional[np.ndarray] = None,
        pred_probs: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
    ) -> None:
        # TODO: Simplifying User Input: Ensure that users can pass input in the simplest form possible.
        # See FindIssuesKwargs._adapt_to_singletons TODO for more details.

        str_to_int_map: Dict[Any, Any] = {v: k for (k, v) in self.label_map.items()}
        find_issues_kwargs = FindIssuesKwargs(
            labels=labels,
            pred_probs=pred_probs,
            features=features,
            _label_map=str_to_int_map,
        )
        issues_dict: Dict[str, Union[List[float], List[bool], np.ndarray]] = {
            k: [] for k in self.issues_dict.keys()
        }

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
            self._display_batch_issues(issues_dict, labels=labels, pred_probs=pred_probs)

        # Append the issues to the existing issues dictionary
        for k, v in issues_dict.items():
            self.issues_dict[k].extend(v)  # type: ignore[arg-type]

    def _display_batch_issues(
        self, issues_dicts: Dict[str, Union[List[float], List[bool], np.ndarray]], **kwargs
    ) -> None:
        start_index = len(
            next(iter(self.issues_dict.values()))
        )  # TODO: Abstract this into a method for checking how many examples have been processed/checked. E.g. __len__ or a property.
        end_index = start_index + len(next(iter(issues_dicts.values())))
        index = np.arange(start_index, end_index)

        issue_columns = [
            col for col in issues_dicts.keys() if (col.startswith("is_") and col.endswith("_issue"))
        ]
        score_columns = [
            col.replace("is_", "").replace("_issue", "_score") for col in issue_columns
        ]
        issue_score_pairs = zip(issue_columns, score_columns)

        pairs_to_keep = [
            (issue_col, score_col)
            for issue_col, score_col in issue_score_pairs
            if np.any(issues_dicts[issue_col])
        ]
        filtered_issues_dicts = {}
        for issue_col, score_col in pairs_to_keep:
            filtered_issues_dicts.update(
                {issue_col: issues_dicts[issue_col], score_col: issues_dicts[score_col]}
            )

        df_issues = pd.DataFrame(filtered_issues_dicts, index=index)

        is_issue_columns = [
            col for col in df_issues.columns if (col.startswith("is_") and col.endswith("_issue"))
        ]
        if "is_label_issue" in is_issue_columns:
            df_issues["given_label"] = kwargs["labels"]
            df_issues["suggested_label"] = np.vectorize(self.label_map.get)(
                np.argmax(kwargs["pred_probs"], axis=1)
            )

        df_subset = df_issues.query(f"{' | '.join([f'{col} == True' for col in is_issue_columns])}")

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
    """Class for monitoring a batch of data for issues."""

    def __init__(self, info: Info):
        self.info = info
        # This issue dictionary will collect the issues for a single batch of data, then be manually cleared.
        self.issues_dict: Dict[str, Union[List[bool], List[float], np.ndarray]] = {
            "is_issue": [],
            "score": [],
        }

    def clear_issues_dict(self):
        """Helper method for the DataMonitor to clear the issues dictionary after processing a batch."""
        self.issues_dict["is_issue"] = []
        self.issues_dict["score"] = []

    @abstractmethod
    def find_issues(self, fi_kwargs: FindIssuesKwargs) -> None:
        pass

    @property
    def batch_has_issues(self) -> bool:
        return any(self.issues_dict["is_issue"])


class LabelIssueMonitor(IssueMonitor):
    """Class that monitors a batch of data for label issues."""

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


class OutlierIssueMonitor(IssueMonitor):
    """Class that monitors a batch of data for outlier issues."""

    def __init__(self, info: Info):
        super().__init__(info)
        outlier_info = info.get("outlier")
        if outlier_info is None:
            raise ValueError("The outlier information is missing in the info dictionary.")

        self.knn = outlier_info["knn"]
        self.ood = outlier_info["ood"]
        self.issue_threshold: float = outlier_info["issue_threshold"]

    def find_issues(self, fi_kwargs: FindIssuesKwargs) -> None:
        """
        Finds outlier issues in the provided batch of features.

        Parameters
        ----------
        fi_kwargs :
            An object containing the keyword arguments for finding issues.
            It should contain features.

        Raises
        ------
        ValueError :
            If `features` are not provided.
        """
        if fi_kwargs.features is None:
            raise ValueError("Features must be provided to find outlier issues.")

        # Reset the flag indicating issues found in the current batch
        self._found_issues_in_batch = False
        scores = self.ood.score(features=fi_kwargs.features)

        distances, _ = self.knn.kneighbors(fi_kwargs.features, return_distance=True)
        avg_distances = np.mean(distances, axis=1)

        # Determine which examples have outlier issues
        is_issue_array = np.zeros(len(fi_kwargs.features), dtype=bool)
        issue_indices = np.where(avg_distances > self.issue_threshold)[0]
        num_issues_in_batch = len(issue_indices)
        if num_issues_in_batch > 0:
            self._found_issues_in_batch = True
            is_issue_array[issue_indices] = True

        # Update issues dictionary
        self.issues_dict = {
            "is_issue": is_issue_array,
            "score": scores,
        }
