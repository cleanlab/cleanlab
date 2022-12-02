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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union, Mapping
import os
import pickle
import json
import pandas as pd
import numpy as np
import warnings

import datasets
from datasets.arrow_dataset import Dataset

import cleanlab
from cleanlab.classification import CleanLearning
from cleanlab.internal.validation import labels_to_array, assert_valid_inputs

__all__ = ["DataLab"]

# Constants:
OBJECT_FILENAME = "datalab.pkl"
ISSUES_FILENAME = "issues.csv"
RESULTS_FILENAME = "results.pkl"
INFO_FILENAME = "info.pkl"


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
        self.label_name = label_name
        self.data.set_format(
            type="numpy"
        )  # TODO: figure out if we are setting all features to numpy, maybe exclude label_name?
        self.issues: Optional[pd.DataFrame] = None  # TODO: Keep track of all issue types,
        self.results = None  # TODO: For each issue type, add a score
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
            sklearn compatible model used to compute out-of-sample predicted probability for the labels.

            WARNING
            -------
            This is not yet implemented.
        """

        issue_kwargs = {
            "pred_probs": pred_probs,
            "feature_values": feature_values,
            "model": model,
        }

        if issue_types is None:
            issue_types = {
                "label": True,
                "health": True,
            }
        issue_managers = [
            factory(datalab=self)
            for factory in _IssueManagerFactory.from_list(list(issue_types.keys()))
        ]

        if issue_types is not None and not issue_types["label"]:
            raise NotImplementedError("TODO: issue_types is not yet supported.")

        for issue_manager in issue_managers:
            issue_manager.find_issues(**issue_kwargs)

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
        formatted_labels = np.array([label_map[l] for l in _labels])
        inverse_map = {i: label for label, i in label_map.items()}

        return formatted_labels, inverse_map

    @staticmethod
    def _validate_data(data) -> None:
        assert not isinstance(
            data, datasets.DatasetDict
        ), "Please pass a single dataset, not a DatasetDict. Try initializing with data['train'] instead."

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
        Keys might include: number of examples suffering from issue, indicates of top-K examples most severely suffering,
        other misc stuff like which sets of examples are duplicates if the issue=="duplicated".
        """
        if issue_name in self.info:
            return self.info[issue_name]
        else:
            raise ValueError(
                f"issue_name {issue_name} not found in self.info. These have not been computed yet."
            )
            # could alternatively consider: raise ValueError("issue_name must be a valid key in Datalab.info dict.")

    def report(self) -> None:
        """Prints helpful summary of all issues."""
        print("Issues will be summarized here.")  # TODO

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

        # Useful info could be: num_examples, task, issues_identified (numeric or None if issue-finding not run yet).
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
        if self.issues is not None:
            state["issues"] = os.path.join(save_path, ISSUES_FILENAME)
            self.issues.to_csv(state["issues"])

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
        # TODO: Try using a custom context manager to save the object and the heavy attributes safely.
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
        # if self.results is not None:
        #     stored_results = self.results
        #     self.results = None
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
        # self.results = stored_results
        print(f"Saved Datalab to folder: {path}")
        print(
            f"The Dataset must be saved/loaded separately to access it after reloading this Datalab."
        )

    @classmethod
    def load(cls, path: str, data: Dataset = None) -> "Datalab":
        """Loads Lab from file. Folder could ideally be zipped or unzipped.
        Checks which cleanlab version Lab was previously saved from and raises warning if they dont match.
        path is path to saved Datalab, not Dataset.

        Dataset should be the same one used before saving.
        If data is None, the self.data attribute of this object will be empty and some functionality may not work.
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
            datalab.data = data
            # TODO: check this matches any of the other attributes, ie. is the Dataset that was used before

        current_version = cleanlab.__version__
        if current_version != datalab.cleanlab_version:
            warnings.warn(
                f"Saved Datalab was created using different version of cleanlab ({datalab.cleanlab_version}) than current version ({current_version}). Things may be broken!"
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


class IssueManager(ABC):
    """Base class for managing issues in a Datalab."""

    def __init__(self, datalab: Datalab):
        self.datalab = datalab

    def __repr__(self):
        class_name = self.__class__.__name__
        return class_name

    @abstractmethod
    def find_issues(self, /, *args, **kwargs) -> Union[dict, pd.DataFrame]:
        """Finds issues in this Lab."""
        raise NotImplementedError

    @abstractmethod
    def update_info(self, /, *args, **kwargs) -> None:
        """Updates the info attribute of this Lab."""
        raise NotImplementedError


class HealthIssueManager(IssueManager):

    info_keys = ["summary"]

    def find_issues(self, pred_probs: np.ndarray, **kwargs) -> dict:
        health_summary = self._health_summary(pred_probs=pred_probs, **kwargs)
        self.update_info(summary=health_summary)
        return health_summary

    def update_info(self, summary: Dict[str, Any], **kwargs) -> None:
        """Updates the info attribute of this Lab."""
        self.datalab.results = summary["overall_label_health_score"]
        for key in self.info_keys:
            if key == "summary":
                self.datalab.info[key] = summary
        self.datalab.info["summary"] = summary

    def _health_summary(self, pred_probs, summary_kwargs=None, **kwargs) -> dict:
        """Returns a short summary of the health of this Lab."""
        from cleanlab.dataset import health_summary

        # Validate input
        self._validate_pred_probs(pred_probs)

        if summary_kwargs is None:
            summary_kwargs = {}

        kwargs_copy = kwargs.copy()
        for k in kwargs_copy:
            if k not in [
                "asymmetric",
                "class_names",
                "num_examples",
                "joint",
                "confident_joint",
                "multi_label",
                "verbose",
            ]:
                del kwargs[k]

        summary_kwargs.update(kwargs)

        class_names = list(self.datalab._label_map.values())
        summary = health_summary(
            self.datalab._labels, pred_probs, class_names=class_names, **summary_kwargs
        )
        return summary

    def _validate_pred_probs(self, pred_probs) -> None:
        assert_valid_inputs(X=None, y=self.datalab._labels, pred_probs=pred_probs)


class LabelIssueManager(IssueManager):

    # TODO: Add `results_key = "label"` to this class
    # TODO: Add `info_keys = ["label"]` to this class
    def __init__(self, datalab: Datalab):
        super().__init__(datalab)
        self.cl = CleanLearning()

    def find_issues(self, pred_probs: np.ndarray, model=None, **kwargs) -> pd.DataFrame:
        if pred_probs is None and model is not None:
            raise NotImplementedError("TODO: We assume pred_probs is provided.")

        issues = self.cl.find_label_issues(labels=self.datalab._labels, pred_probs=pred_probs)
        self.update_info(issues=issues)
        self.datalab.issues = issues
        return issues

    def update_info(self, issues, **kwargs) -> None:
        self.datalab.info["num_label_issues"] = len(issues)


# Construct concrete issue manager with a from_str method
class _IssueManagerFactory:
    """Factory class for constructing concrete issue managers."""

    types = {
        "health": HealthIssueManager,
        "label": LabelIssueManager,
    }

    @classmethod
    def from_str(cls, issue_type: str) -> Type[IssueManager]:
        """Constructs a concrete issue manager from a string."""
        if isinstance(issue_type, list):
            raise ValueError(
                "issue_type must be a string, not a list. Try using from_list instead."
            )
        if issue_type not in cls.types:
            raise ValueError(f"Invalid issue type: {issue_type}")
        return cls.types[issue_type]

    @classmethod
    def from_list(cls, issue_types: List[str]) -> List[Type[IssueManager]]:
        """Constructs a list of concrete issue managers from a list of strings."""
        return [cls.from_str(issue_type) for issue_type in issue_types]
