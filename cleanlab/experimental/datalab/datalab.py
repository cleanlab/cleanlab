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

from typing import Any, Optional, Union, Mapping
import os
import pickle
import json
import pandas as pd
import numpy as np

import datasets
from datasets import Dataset

from cleanlab.classification import CleanLearning
from cleanlab.internal.validation import labels_to_array


class Datalab:
    """
    A single object to find all kinds of issues in datasets.
    It tracks intermediate state from certain functions that can be
    re-used across other functions.  This will become the main way 90%
    of users interface with cleanlab library.

    Parameters
    ----------
    ... : ...
        ...
    """

    # Constants:
    OBJECT_FILENAME = "datalab.p"
    ISSUES_FILENAME = "issues.csv"
    RESULTS_FILENAME = "results.csv"
    INFO_FILENAME = "info.json"

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
        self.issues: Optional[pd.DataFrame] = None
        self.results = None
        self._labels, self._label_map = self._extract_labels(self.label_name)
        class_names = self.data.unique(self.label_name)  # TODO
        self.info = {
            "num_examples": len(self.data),
            "class_names": class_names,
            "num_classes": len(class_names),
        }
        self._silo = self.info.copy()

    def find_issues(
        self,
        *,
        pred_probs=None,
        issue_types: dict = None,
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
        cl = CleanLearning()

        if pred_probs is None and model is not None:
            raise NotImplementedError("TODO: We assume pred_probs is provided.")

        if issue_types is not None:
            raise NotImplementedError("TODO: issue_types is not yet supported.")

        if pred_probs is not None:
            self.issues = cl.find_label_issues(labels=self._labels, pred_probs=pred_probs)

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
            return None
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

        # delete big attributes of this object that should be saved in separate formats:
        # info, issues, results
        stored_info = None
        if self.info is not None:
            stored_info = self.info
            self.info = None
            info_file = os.path.join(path, INFO_FILENAME)
            with open(info_file, "w") as f:
                json.dump(stored_info, f)

        stored_issues = None
        if self.issues is not None:
            stored_issues = self.issues
            self.issues = None
            issues_file = os.path.join(path, ISSUES_FILENAME)
            stored_issues.to_csv(issues_file)

        stored_results = None
        if self.results is not None:
            stored_results = self.results
            self.results = None
            results_file = os.path.join(path, RESULTS_FILENAME)
            stored_results.to_csv(results_file)

        # save trimmed version of this object
        object_file = os.path.join(path, OBJECT_FILENAME)
        with open(object_file, "wb") as f:
            pickle.dumps(self, f)

        # revert object back to original state
        self.info = stored_info
        self.issues = stored_issues
        self.results = stored_results
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

        info_file = os.path.join(path, INFO_FILENAME)
        with open(info_file, "r") as f:
            datalab.info = json.load(f)

        issues_file = os.path.join(path, ISSUES_FILENAME)
        if os.path.exists(issues_file):
            datalab.issues = pd.read_csv(issues_file)

        results_file = os.path.join(path, RESULTS_FILENAME)
        if os.path.exists(results_file):
            datalab.results = pd.read_csv(results_file)

        if data is not None:
            datalab.data = data
            # TODO: check this matches any of the other attributes, ie. is the Dataset that was used before

        current_version = cleanlab.__version__
        if current_version != datalab.cleanlab_version:
            warnings.warn(
                f"Saved Datalab was created using different version of cleanlab ({datalab.cleanlab_version}) than current version ({current_version}). Things may be broken!"
            )
        return datalab
