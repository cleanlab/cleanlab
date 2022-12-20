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
Implements cleanlab's DataLab interface as a one-stop-shop for tracking
and managing all kinds of issues in datasets.
"""
from __future__ import annotations

import os
import pickle
import warnings
from typing import Any, Dict, Mapping, Optional, Union

import datasets
import numpy as np
import pandas as pd
from datasets import load_from_disk
from datasets.arrow_dataset import Dataset

import cleanlab
from cleanlab.experimental.datalab.factory import _IssueManagerFactory
from cleanlab.experimental.datalab.issue_manager import IssueManager
from cleanlab.internal.validation import assert_valid_inputs, labels_to_array

__all__ = ["Datalab"]

# Constants:
OBJECT_FILENAME = "datalab.pkl"
ISSUES_FILENAME = "issues.csv"
ISSUE_SUMMARY_FILENAME = "summary.csv"
INFO_FILENAME = "info.pkl"
DATA_DIRNAME = "data"


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
    """

    def __init__(
        self,
        data: Dataset,
        label_name: Union[str, list[str]],
    ) -> None:
        self._validate_data(data)
        self._validate_data_and_labels(data, label_name)

        if isinstance(label_name, list):
            raise NotImplementedError("TODO: multi-label support.")

        self.data = data
        self._data_hash = hash(data)
        self.label_name = label_name
        self.data.set_format(
            type="numpy"
        )  # TODO: figure out if we are setting all features to numpy, maybe exclude label_name?
        self.issues: pd.DataFrame = pd.DataFrame(
            index=range(len(self.data))
        )  # TODO: Keep track of all issue types,
        self.issue_summary: pd.DataFrame = pd.DataFrame(
            columns=["issue_type", "score"]
        )  # TODO: For each issue type, add a score
        self._labels, self._label_map = self._extract_labels(self.label_name)
        class_names = self.data.unique(self.label_name)  # TODO
        self.info = {
            "num_examples": len(self.data),
            "class_names": class_names,
            "num_classes": len(class_names),
            "multi_label": False,  # TODO: Add multi-label support.
            "health_score": None,
        }
        self.cleanlab_version = cleanlab.__version__
        self.path = ""

    def find_issues(
        self,
        *,
        pred_probs=None,
        issue_types: Optional[Dict["str", Any]] = None,
        feature_values=None,  # embeddings of data
        model=None,  # sklearn.Estimator compatible object
        issue_init_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Any:
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

        feature_values :
            Precomputed embeddings of the features in the dataset.

            WARNING
            -------
            This is not yet implemented.

        model :
            sklearn compatible model used to compute out-of-sample
            predicted probability for the labels.

            WARNING
            -------
            This is not yet implemented.

        issue_init_kwargs :
            # Add path to IssueManager class docstring.
            Keyword arguments to pass to the IssueManager constructor.

            See Also
            --------
            IssueManager


            It is a dictionary of dictionaries, where the keys are the issue types
            and the values are dictionaries of keyword arguments to pass to the
            IssueManager constructor.

            For example, if you want to pass the keyword argument "clean_learning_kwargs"
            to the constructor of the LabelIssueManager, you would pass:

            .. code-block:: python

                issue_init_kwargs = {
                    "label": {
                        "clean_learning_kwargs": {
                            "prune_method": "prune_by_noise_rate",
                        }
                    }
                }

        """

        issue_kwargs = {
            "pred_probs": pred_probs,
            "feature_values": feature_values,
            "model": model,
        }

        if issue_types is None:
            issue_types = {
                "label": True,
                # "health": True,
            }
        issue_keys = list(issue_types.keys())
        issue_managers = [
            factory(datalab=self)
            for (factory, issue_key) in zip(_IssueManagerFactory.from_list(issue_keys), issue_keys)
        ]

        if issue_types is not None and not issue_types["label"]:
            raise NotImplementedError("TODO: issue_types is not yet supported.")

        for issue_manager in issue_managers:
            # TODO: find_issues should return None, set self.issues in a set_issues(issue_manager) method  # noqa: E501
            issue_manager.find_issues(**issue_kwargs)
            self.collect_results_from_issue_manager(issue_manager)

    def collect_results_from_issue_manager(self, issue_manager: IssueManager) -> None:
        """
        Collects results from an IssueManager and update the corresponding
        attributes of the Datalab object.

        This includes:
        - self.issues
        - self.issue_summary
        - self.info

        Parameters
        ----------
        issue_manager :
            IssueManager object to collect results from.
        """
        self.issues = self.issues.join(issue_manager.issues, how="inner")
        self.issue_summary = pd.concat(
            [self.issue_summary, issue_manager.summary],
            axis=0,
            ignore_index=True,
        )
        self.info[issue_manager.issue_key] = issue_manager.info

    def _extract_labels(self, label_name: Union[str, list[str]]) -> tuple[np.ndarray, Mapping]:
        """
        Extracts labels from the data and stores it in self._labels.

        Parameters
        ----------
        ... : ...
            ...
        """

        if isinstance(label_name, list):

            raise NotImplementedError("TODO")

            # _labels = np.vstack([my_data[label] for label in labels]).T

        # Raw values from the dataset
        _labels = self.data[label_name]
        _labels = labels_to_array(_labels)  # type: ignore[assignment]
        if _labels.ndim != 1:
            raise ValueError("labels must be 1D numpy array.")

        unique_labels = np.unique(_labels)
        label_map = {label: i for i, label in enumerate(unique_labels)}
        # labels 0, 1, ..., K-1
        formatted_labels = np.array([label_map[label] for label in _labels])
        inverse_map = {i: label for label, i in label_map.items()}

        return formatted_labels, inverse_map

    @staticmethod
    def _validate_data(data) -> None:
        if isinstance(data, datasets.DatasetDict):
            raise ValueError(
                "Please pass a single dataset, not a DatasetDict. "
                "Try initializing with data['train'] instead."
            )

        assert isinstance(data, Dataset)

    @staticmethod
    def _validate_data_and_labels(data, labels) -> None:
        if isinstance(labels, np.ndarray):
            assert labels.shape[0] == data.shape[0]

        if isinstance(labels, str):
            pass

    def get_info(self, issue_name) -> Any:
        """Returns dict of info about a specific issue, or None if this issue does not exist in self.info.
        Internally fetched from self.info[issue_name] and prettified.
        Keys might include: number of examples suffering from issue,
        indicates of top-K examples most severely suffering,
        other misc stuff like which sets of examples are duplicates if the issue=="duplicated".
        """
        if issue_name in self.info:
            return self.info[issue_name]
        else:
            raise ValueError(
                f"issue_name {issue_name} not found in self.info. These have not been computed yet."
            )
            # could alternatively consider:
            # raise ValueError("issue_name must be a valid key in Datalab.info dict.")

    def report(self) -> None:
        """Prints helpful summary of all issues."""
        print("Issues will be summarized here.")  # TODO
        # Show summary of issues
        # Sort issues base on the score
        # Show top K issues
        # Show the info (get_info) with some verbosity level
        #   E.g. for label issues, only show the confident joint computed with the health_summary

    def __repr__(self) -> str:
        """What is displayed in console if user executes: >>> datalab"""
        checks_run = self.issues is None
        display_str = f"checks_run={checks_run},"
        num_examples = self.get_info("num_examples")
        if num_examples is not None:
            display_str += f"num_examples={num_examples},"
        num_classes = self.get_info("num_classes")
        if num_classes is not None:
            display_str += f"num_classes={num_classes},"
        if display_str[-1] == ",":  # delete trailing comma
            display_str = display_str[:-1]

        # Useful info could be: num_examples, task, issues_identified
        # (numeric or None if issue-finding not run yet).
        return f"Datalab({display_str})"

    def __str__(self) -> str:
        """What is displayed if user executes: print(datalab)"""
        return "Datalab"  # TODO

    def __getstate__(self) -> dict:
        """Used by pickle to serialize the object.

        We don't want to pickle the issues, since it's just a dataframe and can be exported to
        a human readable format. We can replace it with the file path to the exported file.

        """
        state = self.__dict__.copy()
        save_path = self.path

        # Update the issues to be the path to the exported file.
        state["issues"] = os.path.join(save_path, ISSUES_FILENAME)
        self.issues.to_csv(state["issues"], index=False)

        # Update the issue summary to be the path to the exported file.
        state["issue_summary"] = os.path.join(save_path, ISSUE_SUMMARY_FILENAME)
        self.issue_summary.to_csv(state["issue_summary"], index=False)

        # Save the dataset to disk
        if self.data is not None:
            state["data"] = os.path.join(save_path, DATA_DIRNAME)
            self.data.save_to_disk(state["data"])
        # if self.info is not None:
        #     state["info"] = os.path.join(save_path, INFO_FILENAME)
        #     # Pickle the info dict.
        #     with open(state["info"], "wb") as f:
        #         pickle.dump(self.info, f)

        return state

    def __setstate__(self, state: dict) -> None:
        """Used by pickle to deserialize the object.

        We need to load the issues from the file path.
        """

        save_path = state.get("path", "")
        if save_path:
            issues_path = state["issues"]
            if isinstance(issues_path, str) and os.path.exists(issues_path):
                state["issues"] = pd.read_csv(issues_path)

            issue_summary_path = state["issue_summary"]
            if isinstance(issue_summary_path, str) and os.path.exists(issue_summary_path):
                state["issue_summary"] = pd.read_csv(issue_summary_path)

            data_path = state["data"]
            if isinstance(data_path, str) and os.path.exists(data_path):
                state["data"] = load_from_disk(data_path)

            # info_path = state["info"]
            # if isinstance(info_path, str) and os.path.exists(info_path):
            #     with open(info_path, "r") as f:
            #         state["info"] = pickle.load(f)
        self.__dict__.update(state)

    def save(self, path: str) -> None:
        """Saves this Lab to file (all files are in folder at path/).
        Uses nice format for the DF attributes (csv) and dict attributes (eg. json if possible).
        We do not guarantee saved Lab can be loaded from future versions of cleanlab.

        You have to save the Dataset yourself if you want it saved to file!
        """
        if os.path.exists(path):
            print(f"WARNING: Existing files will be overwritten by newly saved files at: {path}")
        else:
            os.mkdir(path)

        self.path = path

        # delete big attributes of this object that should be save
        # d in separate formats:
        # info, issues, results

        # stored_info = None
        # if self.info is not None:
        #     stored_info = self.info
        #     self.info = None
        #     info_file = os.path.join(self.path, INFO_FILENAME)
        #     with open(info_file, "wb") as f:
        #         # json.dump(stored_info, f)
        #         pickle.dump(stored_info, f)

        # # Save the issues to a csv file.
        # if isinstance(self.issues, pd.DataFrame):
        #     self.issues.to_csv(os.path.join(self.path, ISSUES_FILENAME), index=False)

        # stored_results = None
        # if self.issue_summary is not None:
        #     stored_results = self.issue_summary
        #     self.issue_summary = None
        #     results_file = os.path.join(self.path, RESULTS_FILENAME)
        #     # stored_results.to_csv(results_file)
        #     with open(results_file, "wb") as f:
        #         pickle.dump(stored_results, f)

        # save trimmed version of this object
        object_file = os.path.join(self.path, OBJECT_FILENAME)
        with open(object_file, "wb") as f:
            pickle.dump(self, f)

        # revert object back to original state
        # self.info = stored_info
        # self.issue_summary = stored_results
        print(f"Saved Datalab to folder: {path}")
        print(
            "The Dataset must be saved/loaded separately "
            "to access it after reloading this Datalab."
        )

    @classmethod
    def load(cls, path: str, data: Optional[Dataset] = None) -> "Datalab":
        """Loads Lab from file. Folder could ideally be zipped or unzipped.
        Checks which cleanlab version Lab was previously saved from
            and raises warning if they dont match.
        `path` is the path to the saved Datalab, not Dataset.

        Dataset should be the same one used before saving.
        If data is None, the self.data attribute of this object
            will be empty and some functionality may not work.
        """
        if not os.path.exists(path):
            raise ValueError(f"No folder found at specified path: {path}")

        object_file = os.path.join(path, OBJECT_FILENAME)
        with open(object_file, "rb") as f:
            datalab = pickle.load(f)

        # info_file = os.path.join(path, INFO_FILENAME)
        # with open(info_file, "rb") as f:
        #     datalab.info = pickle.load(f)

        # issues_file = os.path.join(path, ISSUES_FILENAME)
        # if os.path.exists(issues_file):
        #     datalab.issues = pd.read_csv(issues_file)

        # results_file = os.path.join(path, RESULTS_FILENAME)
        # if os.path.exists(results_file):
        #     with open(results_file, "rb") as f:
        #         datalab.results = pickle.load(f)

        if data is not None:
            if hash(data) != datalab._data_hash:
                raise ValueError(
                    "Data has been modified since Lab was saved. "
                    "Cannot load Lab with modified data."
                )

            labels = datalab._labels
            if len(data) != len(labels):
                raise ValueError(
                    f"Length of data ({len(data)}) does not match length of labels ({len(labels)})"
                )

            datalab.data = data

        current_version = cleanlab.__version__
        datalab_version = datalab.cleanlab_version
        if current_version != datalab_version:
            warnings.warn(
                f"Saved Datalab was created using different version of cleanlab "
                f"({datalab_version}) than current version ({current_version}). "
                f"Things may be broken!"
            )
        return datalab

    def _health_summary(self, pred_probs, **kwargs) -> dict:
        """Returns a short summary of the health of this Lab."""
        from cleanlab.dataset import health_summary

        # Validate input
        self._validate_pred_probs(pred_probs)

        class_names = list(self._label_map.values())
        summary = health_summary(self._labels, pred_probs, class_names=class_names, **kwargs)
        return summary

    def _validate_pred_probs(self, pred_probs) -> None:
        assert_valid_inputs(X=None, y=self._labels, pred_probs=pred_probs)
