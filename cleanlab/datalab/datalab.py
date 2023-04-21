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
"""
Datalab offers a unified audit to detect all kinds of issues in data and labels.

.. note::
    .. include:: optional_dependencies.rst
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

import cleanlab
from cleanlab.datalab.data import Data
from cleanlab.datalab.data_issues import DataIssues
from cleanlab.datalab.display import _Displayer
from cleanlab.datalab.factory import _IssueManagerFactory
from cleanlab.datalab.serialize import _Serializer

if TYPE_CHECKING:  # pragma: no cover
    from datasets.arrow_dataset import Dataset
    from scipy.sparse import csr_matrix

    DatasetLike = Union[Dataset, pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], str]

__all__ = ["Datalab"]


class Datalab:
    """
    A single object to automatically detect all kinds of issues in datasets.
    This is how we recommend you interface with the cleanlab library if you want to audit the quality of your data. If you have other specific goals, then consider using the other methods across this library. Even then, Datalab may be the easiest way to run specific analyses of your data. Datalab tracks intermediate state (e.g. data statistics) from certain cleanlab functions that can be re-used across other cleanlab functions for better efficiency.

    Parameters
    ----------
    data : Union[Dataset, pd.DataFrame, dict, list, str]
        Dataset-like object that can be converted to a Hugging Face Dataset object.

        It should contain the labels for all examples, identified by a
        `label_name` column in the Dataset object.

        Supported formats:
            - datasets.Dataset
            - pandas.DataFrame
            - dict (keys are strings, values are arrays/lists of length ``N``)
            - list (list of dictionaries that each have the same keys)
            - str
                - path to a local file: Text (.txt), CSV (.csv), JSON (.json)
                - or a dataset identifier on the Hugging Face Hub


    label_name :
        The name of the label column in the dataset.

    verbosity : int, optional
        The higher the verbosity level, the more information
        Datalab prints when auditing a dataset.
        Valid values are 0 through 4. Default is 1.

    Examples
    --------
    >>> import datasets
    >>> from cleanlab import Datalab
    >>> data = datasets.load_dataset("glue", "sst2", split="train")
    >>> datalab = Datalab(data, label_name="label")
    """

    def __init__(
        self,
        data: "DatasetLike",
        label_name: str,
        verbosity: int = 1,
    ) -> None:
        self._data = Data(data, label_name)
        self.data = self._data._data
        self._labels, self._label_map = self._data._labels, self._data._label_map
        self._data_hash = self._data._data_hash
        self.label_name = self._data._label_name
        self.data_issues = DataIssues(self._data)
        self.cleanlab_version = cleanlab.version.__version__
        self.path = ""
        self.verbosity = verbosity

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
    def class_names(self) -> List[str]:
        """Names of the classes in the dataset."""
        return self._data.class_names

    @property
    def issues(self) -> pd.DataFrame:
        """Issues found in each example from the dataset."""
        return self.data_issues.issues

    @issues.setter
    def issues(self, issues: pd.DataFrame) -> None:
        self.data_issues.issues = issues

    @property
    def issue_summary(self) -> pd.DataFrame:
        """Summary of issues found in the dataset and the overall severity of each type of issue.

        Example
        -------

        If checks for "label" and "outlier" issues were run,
        then the issue summary will look something like this:

        >>> datalab.issue_summary
        issue_type  score
        outlier     0.123
        label       0.456
        """
        return self.data_issues.issue_summary

    @issue_summary.setter
    def issue_summary(self, issue_summary: pd.DataFrame) -> None:
        self.data_issues.issue_summary = issue_summary

    @property
    def info(self) -> Dict[str, Dict[str, Any]]:
        """Information and statistics about the dataset issues found.

        Example
        -------

        If checks for "label" and "outlier" issues were run,
        then the info will look something like this:

        >>> datalab.info
        {
            "label": {
                "given_labels": [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, ...],
                "predicted_label": [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, ...],
                ...,
            },
            "outlier": {
                "nearest_neighbor": [3, 7, 1, 2, 8, 4, 5, 9, 6, 0, ...],
                "distance_to_nearest_neighbor": [0.123, 0.789, 0.456, ...],
                ...,
            },
        }
        """
        return self.data_issues.info

    @info.setter
    def info(self, info: Dict[str, Dict[str, Any]]) -> None:
        self.data_issues.info = info

    def _resolve_required_args(self, pred_probs, features, knn_graph):
        """Resolves the required arguments for each issue type.

        This is a helper function that filters out any issue manager
        that does not have the required arguments.

        This does not consider custom hyperparameters for each issue type.


        Parameters
        ----------
        pred_probs :
            Out-of-sample predicted probabilities made on the data.

        features :
            Name of column containing precomputed embeddings.

        Returns
        -------
        args_dict :
            Dictionary of required arguments for each issue type, if available.
        """
        args_dict = {
            "label": {"pred_probs": pred_probs},
            "outlier": {"pred_probs": pred_probs, "features": features, "knn_graph": knn_graph},
            "near_duplicate": {"features": features, "knn_graph": knn_graph},
            "non_iid": {"features": features, "knn_graph": knn_graph},
        }

        args_dict = {
            k: {k2: v2 for k2, v2 in v.items() if v2 is not None} for k, v in args_dict.items() if v
        }

        # Prefer `knn_graph` over `features` if both are provided.
        for v in args_dict.values():
            if "knn_graph" in v and "features" in v:
                warnings.warn(
                    "Both `features` and `knn_graph` were provided. "
                    "Most issue managers will likely prefer using `knn_graph` "
                    "instead of `features` for efficiency."
                )

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
        with different arguments, some of which may be user-provided.

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

    def _get_report(self, num_examples: int, verbosity: int, include_description: bool) -> str:
        # Sort issues based on the score
        # Show top k issues
        # Show the info (get_info) with some verbosity level
        #   E.g. for label issues, only show the confident joint computed with the health_summary
        report_str = ""
        issue_type_sorted = self._sort_issue_summary_by_issue_counts(self.issue_summary)
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
                    num_examples=num_examples,
                    verbosity=verbosity,
                    include_description=include_description,
                )
            )

        report_str += "\n\n\n".join(issue_manager_reports)
        return report_str

    def _sort_issue_summary_by_issue_counts(self, issue_summary: pd.DataFrame) -> pd.DataFrame:
        """Sort issue_summary by the number of issues per issue type.

        Returns
        -------
        sorted_summary :
            Sorted issue_summary.

        Examples
        --------
        >>> issue_summary = pd.DataFrame(
        ...     {
        ...         "issue_type": ["label", "outlier", "near_duplicate"],
        ...         "score": [0.5, 0.2, 0.1],
        ...     }
        ... )
        >>> # Calling this method will sort the issue_summary by the number of issues
        >>> sorted_summary = Datalab._sort_issue_summary_by_issue_counts(issue_summary)
        >>> sorted_summary
               issue_type  score  num_issues
        0         outlier    0.2           5
        1           label    0.5           3
        2  near_duplicate    0.1           2
        """
        summary = issue_summary.copy()
        names = summary["issue_type"].tolist()
        counts = [self.get_issues(issue_name=name)[f"is_{name}_issue"].sum() for name in names]
        # Rank issue_summary by the number of issues in issue_type_counts
        summary["num_issues"] = counts
        # issue_type_sorted = self.issue_summary.sort_values(by="score", ascending=True)
        sorted_summary = summary.sort_values(by="num_issues", ascending=False)
        return sorted_summary

    def get_issues(self, issue_name: Optional[str] = None) -> pd.DataFrame:
        """
        Use this after finding issues to see which examples suffer from which types of issues.

        Parameters
        ----------
        issue_name : str or None
            The type of issue to focus on. If `None`, returns full DataFrame summarizing all of the types of issues detected in each example from the dataset.

        Raises
        ------
        ValueError
            If `issue_name` is not a type of issue previously considered in the audit.

        Returns
        -------
        specific_issues :
            A DataFrame where each row corresponds to an example from the dataset and columns specify:
            whether this example exhibits a particular type of issue and how severely (via a numeric quality score where lower values indicate more severe instances of the issue).

            Additional columns may be present in the DataFrame depending on the type of issue specified.
        """
        if issue_name is None:
            return self.issues

        specific_issues = self._get_matching_issue_columns(issue_name)
        info = self.get_info(issue_name=issue_name)
        if issue_name == "label":
            specific_issues = specific_issues.assign(
                given_label=info["given_label"], predicted_label=info["predicted_label"]
            )

        if issue_name == "outlier":
            column_dict = {
                k: info.get(k)
                for k in ["nearest_neighbor", "distance_to_nearest_neighbor"]
                if info.get(k) is not None
            }
            specific_issues = specific_issues.assign(**column_dict)

        if issue_name == "near_duplicate":
            column_dict = {
                k: info.get(k)
                for k in ["near_duplicate_sets", "distance_to_nearest_neighbor"]
                if info.get(k) is not None
            }
            specific_issues = specific_issues.assign(**column_dict)
        return specific_issues

    def _get_matching_issue_columns(self, issue_name: str) -> pd.DataFrame:
        columns = [col for col in self.issues.columns if issue_name in col]
        if not columns:
            raise ValueError(f"No columns found for issue type '{issue_name}'.")
        return self.issues[columns]

    def get_summary(self, issue_name: Optional[str] = None) -> pd.DataFrame:
        """Summarize the issues found in dataset of a particular type,
        including how severe this type of issue is overall across the dataset.

        Parameters
        ----------
        issue_name :
            Name of the issue type to summarize. If `None`, summarizes each of the different issue types previously considered in the audit.

        Returns
        -------
        summary :
            DataFrame where each row corresponds to a type of issue, and columns quantify:
            the number of examples in the dataset estimated to exhibit this type of issue,
            and the overall severity of the issue across the dataset (via a numeric quality score where lower values indicate that the issue is overall more severe).
        """
        if self.issue_summary.empty:
            raise ValueError(
                "No issues found in the dataset. "
                "Call `find_issues` before calling `get_summary`."
            )

        if issue_name is None:
            return self.issue_summary

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
        features: Optional[npt.NDArray] = None,
        knn_graph: Optional[csr_matrix] = None,
        issue_types: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Checks the dataset for all sorts of common issues in real-world data (in both labels and feature values).

        You can use Datalab to find issues in your data, utilizing *any* model you have already trained.
        This method only interacts with your model via its predictions or embeddings (and other functions thereof).
        The more of these inputs you provide, the more types of issues Datalab can detect in your dataset/labels.
        If you provide a subset of these inputs, Datalab will output what insights it can based on the limited information from your model.

        Note
        ----
        The issues are saved in the ``self.issues`` attribute, but are not returned.

        Parameters
        ----------
        pred_probs :
            Out-of-sample predicted class probabilities made by the model for every example in the dataset.
            To best detect label issues, provide this input obtained from the most accurate model you can produce.

        features :
            Feature embeddings (vector representations) of every example in the dataset.

        knn_graph :
            Sparse matrix representing similarities between examples in the dataset in a K nearest neighbor graph.
            If both `knn_graph` and `features` are provided, the `knn_graph` will take precendence.
            If `knn_graph` is not provided, it is constructed based on the provided `features`.
            If neither `knn_graph` nor `features` are provided, certain issue types like (near) duplicates will not be considered.

        issue_types :
            Collection specifying which types of issues to consider in audit and any non-default parameter settings to use.
            If unspecified, a default set of issue types and recommended parameter settings is considered.

            This is a dictionary of dictionaries, where the keys are the issue types of interest
            and the values are dictionaries of parameter values that control how each type of issue is detected.
            More specifically, the values are constructor keyword arguments passed to the corresponding ``IssueManager``,
            which is responsible for detecting the particular issue type.

            .. seealso::
                IssueManager

            Examples
            --------

            Suppose you want to only consider label issues. Just pass a dictionary with the key "label" and an empty dictionary as the value (to use default label issue parameters).

            .. code-block:: python

                issue_types = {"label": {}}


            If you are advanced user who wants greater control, you can pass keyword arguments to the issue manager that handles the label issues.
            For example, if you want to pass the keyword argument "clean_learning_kwargs"
            to the constructor of the LabelIssueManager, you would pass:


            .. code-block:: python

                issue_types = {
                    "label": {
                        "clean_learning_kwargs": {
                            "prune_method": "prune_by_noise_rate",
                        },
                    },
                }
        """

        if issue_types is not None and not issue_types:
            warnings.warn(
                "No issue types were specified. " "No issues will be found in the dataset."
            )
            return None

        issue_types_copy = self.get_available_issue_types(
            pred_probs=pred_probs,
            features=features,
            knn_graph=knn_graph,
            issue_types=issue_types,
        )

        new_issue_managers = [
            factory(datalab=self, **issue_types_copy.get(factory.issue_name, {}))
            for factory in _IssueManagerFactory.from_list(list(issue_types_copy.keys()))
        ]

        if not new_issue_managers:
            no_args_passed = all(arg is None for arg in [pred_probs, features, knn_graph])
            if no_args_passed:
                warnings.warn("No arguments were passed to find_issues.")
            warnings.warn("No issue check performed.")
            return None

        failed_managers = []
        for issue_manager, arg_dict in zip(new_issue_managers, issue_types_copy.values()):
            try:
                if self.verbosity:
                    print(f"Finding {issue_manager.issue_name} issues ...")
                issue_manager.find_issues(**arg_dict)
                self.data_issues.collect_statistics_from_issue_manager(issue_manager)
                self.data_issues._collect_results_from_issue_manager(issue_manager)
            except Exception as e:
                print(f"Error in {issue_manager.issue_name}: {e}")
                failed_managers.append(issue_manager)

        if self.verbosity:
            print(
                f"Audit complete. {self.issue_summary['num_issues'].sum()} issues found in the dataset."
            )
        if failed_managers:
            print(f"Failed to check for these issue types: {failed_managers}")

        self.data_issues.set_health_score()

    @staticmethod
    def list_default_issue_types() -> List[str]:
        """Returns a list of the issue types that are run by default by Datalab.

        See Also
        --------
        :py:class:`REGISTRY <cleanlab.datalab.factory.REGISTRY>` : The issue types available by default and their corresponding issue managers can be found here.
        """
        return ["label", "outlier", "near_duplicate"]

    def get_available_issue_types(self, **kwargs):
        """Returns a dictionary of issue types that can be used in :py:meth:`Datalab.find_issues
        <cleanlab.datalab.datalab.Datalab.find_issues>` method."""

        pred_probs = kwargs.get("pred_probs", None)
        features = kwargs.get("features", None)
        knn_graph = kwargs.get("knn_graph", None)
        issue_types = kwargs.get("issue_types", None)

        # Determine which parameters are required for each issue type
        required_args_per_issue_type = self._resolve_required_args(pred_probs, features, knn_graph)

        issue_types_copy = self._set_issue_types(issue_types, required_args_per_issue_type)

        if issue_types is None:
            # Only run default issue types if no issue types are specified
            issue_types_copy = {
                issue: issue_types_copy[issue]
                for issue in self.list_default_issue_types()
                if issue in issue_types_copy
            }

        return issue_types_copy

    def get_info(self, issue_name: Optional[str] = None) -> Dict[str, Any]:
        """Returns dict of info about a specific issue,
        or None if this issue does not exist in self.info.
        Internally fetched from self.info[issue_name] and prettified.
        Keys might include: number of examples suffering from issue,
        indicates of top-K examples most severely suffering,
        other misc stuff like which sets of examples are duplicates if the issue=="duplicated".
        """  # TODO: Revise Datalab.get_info docstring
        return self.data_issues.get_info(issue_name)

    def report(
        self,
        *,
        num_examples: int = 5,
        verbosity: Optional[int] = None,
        include_description: bool = True,
    ) -> None:
        """Prints informative summary of all issues.

        Parameters
        ----------
        num_examples :
            Number of examples to show for each type of issue.
            The report shows the top `num_examples` instances in the dataset that suffer the most from each type of issue.

        verbosity :
            Higher verbosity levels add more information to the report.

        include_description :
            Whether or not to include a description of each issue type in the report.
            Consider setting this to ``False`` once you're familiar with how each issue type is defined.
        """
        if verbosity is None:
            verbosity = self.verbosity
        # Show summary of issues
        print(
            self._get_report(
                num_examples=num_examples,
                verbosity=verbosity,
                include_description=include_description,
            )
        )

    def save(self, path: str, force: bool = False) -> None:
        """Saves this Datalab object to file (all files are in folder at `path/`).
        We do not guarantee saved Datalab can be loaded from future versions of cleanlab.

        Parameters
        ----------
        path :
            Folder in which all information about this Datalab should be saved.

        force :
            If ``True``, overwrites any existing files in the folder at `path`. Use this with caution!

        Note
        ----
        You have to save the Dataset yourself separately if you want it saved to file.
        """
        _Serializer.serialize(path=path, datalab=self, force=force)
        save_message = f"Saved Datalab to folder: {path}"
        print(save_message)

    @staticmethod
    def load(path: str, data: Optional[Dataset] = None) -> "Datalab":
        """Loads Datalab object from a previously saved folder.

        Parameters
        ----------
        `path` :
            Path to the folder previously specified in ``Datalab.save()``.

        `data` :
            The dataset used to originally construct the Datalab.
            Remember the dataset is not saved as part of the Datalab,
            you must save/load the data separately.

        Returns
        -------
        `datalab` :
            A Datalab object that is identical to the one originally saved.
        """
        datalab = _Serializer.deserialize(path=path, data=data)
        load_message = f"Datalab loaded from folder: {path}"
        print(load_message)
        return datalab
