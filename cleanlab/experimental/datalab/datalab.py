# Copyright (C) 2017-2022  Cleanlab Inc.
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
Implements cleanlab's Datalab interface as a one-stop-shop for tracking
and managing all kinds of issues in datasets.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

import cleanlab
from cleanlab.experimental.datalab.data import Data
from cleanlab.experimental.datalab.data_issues import DataIssues
from cleanlab.experimental.datalab.display import _Displayer
from cleanlab.experimental.datalab.factory import _IssueManagerFactory
from cleanlab.experimental.datalab.serialize import _Serializer

if TYPE_CHECKING:  # pragma: no cover
    from datasets.arrow_dataset import Dataset

    from cleanlab.experimental.datalab.issue_manager import IssueManager

__all__ = ["Datalab"]


class Datalab:
    """
    A single object to find all kinds of issues in datasets.
    It tracks intermediate state from certain functions that can be
    re-used across other functions.  This will become the main way 90%
    of users interface with cleanlab library.

    Parameters
    ----------
    data :
        A Hugging Face Dataset object.

    label_name :
        The name of the label column in the dataset.

    Examples
    --------
    >>> import datasets
    >>> from cleanlab import Datalab
    >>> data = datasets.load_dataset("glue", "sst2", split="train")
    >>> datalab = Datalab(data, label_name="label")
    """

    def __init__(
        self,
        data: Dataset,
        label_name: Union[str, List[str]],
    ) -> None:
        self._data = Data(data, label_name)  # TODO: Set extracted class instance to self.data
        self.data = self._data._data
        self._labels, self._label_map = self._data._labels, self._data._label_map
        self._data_hash = self._data._data_hash
        self.label_name = self._data._label_name
        self.data_issues = DataIssues(self._data)
        self.cleanlab_version = cleanlab.version.__version__
        self.path = ""

    def __repr__(self) -> str:
        """What is displayed if user executes: datalab"""
        return _Displayer(self).__repr__()

    def __str__(self) -> str:
        """What is displayed if user executes: print(datalab)"""
        return _Displayer(self).__str__()

    @property
    def labels(self) -> np.ndarray:
        """Labels of the dataset, in a [0, 1, ..., K-1] format."""
        return self._labels

    @property
    def issues(self) -> pd.DataFrame:
        """Issues found in the dataset."""
        return self.data_issues.issues

    @issues.setter
    def issues(self, issues: pd.DataFrame) -> None:
        self.data_issues.issues = issues

    @property
    def issue_summary(self) -> pd.DataFrame:
        """Summary of issues found in the dataset."""
        return self.data_issues.issue_summary

    @issue_summary.setter
    def issue_summary(self, issue_summary: pd.DataFrame) -> None:
        self.data_issues.issue_summary = issue_summary

    @property
    def info(self) -> Dict[str, Dict[str, Any]]:
        """Information and statistics about the dataset issues found."""
        return self.data_issues.info

    @info.setter
    def info(self, info: Dict[str, Dict[str, Any]]) -> None:
        self.data_issues.info = info

    def _resolve_required_args(self, pred_probs, features, model):
        """Resolves the required arguments for each issue type.

        This is a helper function that filters out any issue manager that does not have the required arguments.

        This does not consider custom hyperparameters for each issue type.


        Parameters
        ----------
        pred_probs :
            Out-of-sample predicted probabilities made on the data.

        features :
            Name of column containing precomputed embeddings.

        model :
            sklearn compatible model used to compute out-of-sample predicted probabilities for the labels.

        Returns
        -------
        args_dict :
            Dictionary of required arguments for each issue type, if available.
        """
        args_dict = {
            "label": {"pred_probs": pred_probs, "model": model},
            "outlier": {"pred_probs": pred_probs, "features": features},
            "near_duplicate": {"features": features},
        }

        args_dict = {
            k: {k2: v2 for k2, v2 in v.items() if v2 is not None} for k, v in args_dict.items() if v
        }
        args_dict = {k: v for k, v in args_dict.items() if v}

        return args_dict

    def _set_issue_types(
        self,
        issue_types: Optional[Dict[str, Any]],
        required_defaults_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Set necessary configuration for each IssueManager in a dictionary.

        While each IssueManager defines default values for its arguments,
        the Datalab class needs to organize the calls to each IssueManager
        with different arguments, some of which may be

        Parameters
        ----------
        issue_types :
            Dictionary of issue types and argument configuration for their respective IssueManagers.
            If None, then the `required_defaults_dict` is used.

        required_defaults_dict :
            Dictionary of default parameter configuration for each issue type.

        Returns
        -------
        issue_types_copy :
            Dictionary of issue types and their parameter configuration.
            The input `issue_types` is copied and updated with the necessary default values.
        """
        if issue_types is not None:
            issue_types_copy = issue_types.copy()
            self._check_missing_args(required_defaults_dict, issue_types_copy)
        else:
            issue_types_copy = required_defaults_dict.copy()
        # Check that all required arguments are provided.
        self._validate_issue_types_dict(issue_types_copy, required_defaults_dict)

        # Remove None values from argument list, rely on default values in IssueManager
        for key, value in issue_types_copy.items():
            issue_types_copy[key] = {k: v for k, v in value.items() if v is not None}
        return issue_types_copy

    @staticmethod
    def _check_missing_args(required_defaults_dict, issue_types):
        for key, issue_type_value in issue_types.items():
            missing_args = set(required_defaults_dict.get(key, {})) - set(issue_type_value.keys())
            # Impute missing arguments with default values.
            missing_dict = {
                missing_arg: required_defaults_dict[key][missing_arg]
                for missing_arg in missing_args
            }
            issue_types[key].update(missing_dict)

    @staticmethod
    def _validate_issue_types_dict(
        issue_types: Dict[str, Any], required_defaults_dict: Dict[str, Any]
    ) -> None:
        missing_required_args_dict = {}
        for issue_name, required_args in required_defaults_dict.items():
            if issue_name in issue_types:
                missing_args = set(required_args.keys()) - set(issue_types[issue_name].keys())
                if missing_args:
                    missing_required_args_dict[issue_name] = missing_args
        if any(missing_required_args_dict.values()):
            error_message = ""
            for issue_name, missing_required_args in missing_required_args_dict.items():
                error_message += f"Required argument {missing_required_args} for issue type {issue_name} was not provided.\n"
            raise ValueError(error_message)

    def _get_report(self, k: int, verbosity: int, include_description: bool) -> str:
        # Sort issues based on the score
        # Show top K issues
        # Show the info (get_info) with some verbosity level
        #   E.g. for label issues, only show the confident joint computed with the health_summary
        report_str = ""
        issue_type_sorted = self.issue_summary.sort_values(by="score", ascending=True)
        report_str += self._add_issue_summary_to_report(summary=issue_type_sorted)
        issue_type_sorted_keys: List[str] = issue_type_sorted["issue_type"].tolist()
        issue_manager_reports = []
        for key in issue_type_sorted_keys:
            issue_manager_class = _IssueManagerFactory.from_str(issue_type=key)
            issue_manager_reports.append(
                issue_manager_class.report(
                    issues=self.get_issues(issue_name=key),
                    summary=self.get_summary(issue_name=key),
                    info=self.get_info(issue_name=key),
                    k=k,
                    verbosity=verbosity,
                    include_description=include_description,
                )
            )

        report_str += "\n\n\n".join(issue_manager_reports)
        return report_str

    def get_issues(self, issue_name: str) -> pd.DataFrame:
        columns = [col for col in self.issues.columns if issue_name in col]

        if not columns:
            raise ValueError(f"Issue type {issue_name} not found in the dataset.")
        return self.issues[columns]

    def get_summary(self, issue_name: str) -> pd.DataFrame:
        """Get summary of issues for a given issue type.

        Parameters
        ----------
        issue_name :
            Name of the issue type.

        Returns
        -------
        summary :
            Summary of issues for a given issue type.
        """
        if self.issue_summary.empty:
            raise ValueError(
                "No issues found in the dataset. "
                "Call `find_issues` before calling `get_summary`."
            )
        row_mask = self.issue_summary["issue_type"] == issue_name
        if not any(row_mask):
            raise ValueError(f"Issue type {issue_name} not found in the summary.")
        return self.issue_summary[row_mask].reset_index(drop=True)

    def _add_issue_summary_to_report(self, summary: pd.DataFrame) -> str:
        return (
            "Here is a summary of the different kinds of issues found in the data:\n\n"
            + summary.to_string(index=False)
            + "\n\n"
            + "(Note: A lower score indicates a more severe issue across all examples in the dataset.)\n\n\n"
        )

    def find_issues(
        self,
        *,
        pred_probs: Optional[np.ndarray] = None,
        issue_types: Optional[Dict[str, Any]] = None,
        features: Optional[npt.NDArray] = None,
        model=None,  # sklearn.Estimator compatible object  # noqa: F821
    ) -> None:
        """
        Checks for all sorts of issues in the data, including in labels and in features.

        Can utilize either provided model or pred_probs.

        Note
        ----
        The issues are saved in self.issues, but are not returned.

        Parameters
        ----------
        pred_probs :
            Out-of-sample predicted probabilities made on the data.

        issue_types :
            Collection of the types of issues to search for.

            Keyword arguments to pass to the IssueManager constructor.

            See Also
            --------
            IssueManager

            It is a dictionary of dictionaries, where the keys are the issue types
            and the values are dictionaries of keyword arguments to pass to the
            IssueManager constructor.

            Example
            -------
            For example, if you want to pass the keyword argument "clean_learning_kwargs"
            to the constructor of the LabelIssueManager, you would pass:


            .. code-block:: python

                issue_init_kwargs = {
                    "label": {
                        "clean_learning_kwargs": {
                            "prune_method": "prune_by_noise_rate",
                        },
                    },
                }

        features :
            Precomputed feature embeddings of the data.

        model :
            sklearn compatible model used to compute out-of-sample
            predicted probability for the labels.

            WARNING
            -------
            This is not yet implemented.
        """

        required_args_per_issue_type = self._resolve_required_args(pred_probs, features, model)

        issue_types_copy = self._set_issue_types(issue_types, required_args_per_issue_type)

        new_issue_managers = [
            factory(datalab=self, **issue_types_copy.get(factory.issue_name, {}))
            for factory in _IssueManagerFactory.from_list(list(issue_types_copy.keys()))
        ]

        failed_managers = []
        for issue_manager, arg_dict in zip(new_issue_managers, issue_types_copy.values()):
            try:
                issue_manager.find_issues(**arg_dict)
                self.data_issues._collect_results_from_issue_manager(issue_manager)
            except Exception as e:
                print(f"Error in {issue_manager.issue_name}: {e}")
                failed_managers.append(issue_manager)

        if failed_managers:
            print(f"Failed to find issues for {failed_managers}")

    def get_info(self, issue_name) -> Dict[str, Any]:
        """Returns dict of info about a specific issue, or None if this issue does not exist in self.info.
        Internally fetched from self.info[issue_name] and prettified.
        Keys might include: number of examples suffering from issue,
        indicates of top-K examples most severely suffering,
        other misc stuff like which sets of examples are duplicates if the issue=="duplicated".
        """  # TODO: Revise Datalab.get_info docstring
        return self.data_issues.get_info(issue_name)

    def report(self, k: int = 5, verbosity: int = 0, include_description: bool = True) -> None:
        """Prints helpful summary of all issues.

        Parameters
        ----------
        k :
            Number of examples to show for each type of issue.

        verbosity :
            Level of verbosity. 0 is the default and prints the top k examples
            for each issue. Higher levels may add more information to the report.
        """
        # Show summary of issues
        print(self._get_report(k=k, verbosity=verbosity, include_description=include_description))

    def save(self, path: str) -> None:
        """Saves this Lab to file (all files are in folder at path/).
        Uses nice format for the DF attributes (csv) and dict attributes (eg. json if possible).
        We do not guarantee saved Lab can be loaded from future versions of cleanlab.

        You have to save the Dataset yourself if you want it saved to file!
        """  # TODO: Revise Datalab.save docstring: Formats and guarantees
        _Serializer.serialize(path=path, datalab=self)

    @staticmethod
    def load(path: str, data: Optional[Dataset] = None) -> "Datalab":
        """Loads Lab from file. Folder could ideally be zipped or unzipped.

        Checks which cleanlab version Lab was previously saved from
        and raises warning if they dont match.

        If a Dataset is passed (via the `data` argument), it will be added to the Datalab
        if it matches the Dataset that was used to create the Datalab.

        Parameters
        ----------

        `path` :
            Path to the folder containing the save Datalab and
            associated files, not the Dataset.
        """  # TODO: Revise Datalab.load docstring: Zipped/unzipped, guarantees
        return _Serializer.deserialize(path=path, data=data)
