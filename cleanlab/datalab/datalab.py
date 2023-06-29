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

import numpy as np
import pandas as pd

import cleanlab
from cleanlab.datalab.data import Data
from cleanlab.datalab.data_issues import DataIssues
from cleanlab.datalab.display import _Displayer
from cleanlab.datalab.issue_finder import IssueFinder
from cleanlab.datalab.serialize import _Serializer
from cleanlab.datalab.report import Reporter

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from datasets.arrow_dataset import Dataset
    from scipy.sparse import csr_matrix

    DatasetLike = Union[Dataset, pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], str]

__all__ = ["Datalab"]


class Datalab:
    """
    A single object to automatically detect all kinds of issues in datasets.
    This is how we recommend you interface with the cleanlab library if you want to audit the quality of your data and detect issues within it.
    If you have other specific goals (or are doing a less standard ML task not supported by Datalab), then consider using the other methods across the library.
    Datalab tracks intermediate state (e.g. data statistics) from certain cleanlab functions that can be re-used across other cleanlab functions for better efficiency.

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

    label_name : str
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
        label_name: Optional[str] = None,
        verbosity: int = 1,
    ) -> None:
        self._data = Data(data, label_name)
        self.data = self._data._data
        self._labels = self._data.labels
        self._label_map = self._labels.label_map
        self.label_name = self._labels.label_name
        self._data_hash = self._data._data_hash
        self.data_issues = DataIssues(self._data)
        self.cleanlab_version = cleanlab.version.__version__
        self.verbosity = verbosity

    def __repr__(self) -> str:
        return _Displayer(data_issues=self.data_issues).__repr__()

    def __str__(self) -> str:
        return _Displayer(data_issues=self.data_issues).__str__()

    @property
    def labels(self) -> np.ndarray:
        """Labels of the dataset, in a [0, 1, ..., K-1] format."""
        return self._labels.labels

    @property
    def has_labels(self) -> bool:
        """Whether the dataset has labels."""
        return self._labels.is_available

    @property
    def class_names(self) -> List[str]:
        """Names of the classes in the dataset.

        If the dataset has no labels, returns an empty list.
        """
        return self._labels.class_names

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
        This method acts as a wrapper around the :py:meth:`IssueFinder.find_issues <cleanlab.datalab.issue_finder.IssueFinder.find_issues>` method,
        where the core logic for issue detection is implemented.

        Note
        ----
        The issues are saved in the ``self.issues`` attribute, but are not returned.

        Parameters
        ----------
        pred_probs :
            Out-of-sample predicted class probabilities made by the model for every example in the dataset.
            To best detect label issues, provide this input obtained from the most accurate model you can produce.

            If provided, this must be a 2D array with shape (num_examples, K) where K is the number of classes in the dataset.

        features : Optional[np.ndarray]
            Feature embeddings (vector representations) of every example in the dataset.

            If provided, this must be a 2D array with shape (num_examples, num_features).

        knn_graph :
            Sparse matrix representing distances between examples in the dataset in a k nearest neighbor graph.

            If provided, this must be a square CSR matrix with shape (num_examples, num_examples) and (k*num_examples) non-zero entries (k is the number of nearest neighbors considered for each example)
            evenly distributed across the rows.
            The non-zero entries must be the distances between the corresponding examples. Self-distances must be omitted
            (i.e. the diagonal must be all zeros and the k nearest neighbors of each example must not include itself).

            For any duplicated examples i,j whose distance is 0, there should be an *explicit* zero stored in the matrix, i.e. ``knn_graph[i,j] = 0``.

            If both `knn_graph` and `features` are provided, the `knn_graph` will take precendence.
            If `knn_graph` is not provided, it is constructed based on the provided `features`.
            If neither `knn_graph` nor `features` are provided, certain issue types like (near) duplicates will not be considered.

        issue_types :
            Collection specifying which types of issues to consider in audit and any non-default parameter settings to use.
            If unspecified, a default set of issue types and recommended parameter settings is considered.

            This is a dictionary of dictionaries, where the keys are the issue types of interest
            and the values are dictionaries of parameter values that control how each type of issue is detected (only for advanced users).
            More specifically, the values are constructor keyword arguments passed to the corresponding ``IssueManager``,
            which is responsible for detecting the particular issue type.

            .. seealso::
                :py:class:`IssueManager <cleanlab.datalab.issue_manager.issue_manager.IssueManager>`

        Examples
        --------

        Here are some ways to provide inputs to :py:meth:`find_issues`:

        - Passing ``pred_probs``:
            .. code-block:: python

                >>> from sklearn.linear_model import LogisticRegression
                >>> import numpy as np
                >>> from cleanlab import Datalab
                >>> X = np.array([[0, 1], [1, 1], [2, 2], [2, 0]])
                >>> y = np.array([0, 1, 1, 0])
                >>> clf = LogisticRegression(random_state=0).fit(X, y)
                >>> pred_probs = clf.predict_proba(X)
                >>> lab = Datalab(data={"X": X, "y": y}, label_name="y")
                >>> lab.find_issues(pred_probs=pred_probs)


        - Passing ``features``:
            .. code-block:: python

                >>> from sklearn.linear_model import LogisticRegression
                >>> from sklearn.neighbors import NearestNeighbors
                >>> import numpy as np
                >>> from cleanlab import Datalab
                >>> X = np.array([[0, 1], [1, 1], [2, 2], [2, 0]])
                >>> y = np.array([0, 1, 1, 0])
                >>> lab = Datalab(data={"X": X, "y": y}, label_name="y")
                >>> lab.find_issues(features=X)

        .. note::

            You can pass both ``pred_probs`` and ``features`` to :py:meth:`find_issues` for a more comprehensive audit.

        - Passing a ``knn_graph``:
            .. code-block:: python

                >>> from sklearn.neighbors import NearestNeighbors
                >>> import numpy as np
                >>> from cleanlab import Datalab
                >>> X = np.array([[0, 1], [1, 1], [2, 2], [2, 0]])
                >>> y = np.array([0, 1, 1, 0])
                >>> nbrs = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X)
                >>> knn_graph = nbrs.kneighbors_graph(mode="distance")
                >>> knn_graph # Pass this to Datalab
                <4x4 sparse matrix of type '<class 'numpy.float64'>'
                        with 8 stored elements in Compressed Sparse Row format>
                >>> knn_graph.toarray()  # DO NOT PASS knn_graph.toarray() to Datalab, only pass the sparse matrix itself
                array([[0.        , 1.        , 2.23606798, 0.        ],
                        [1.        , 0.        , 1.41421356, 0.        ],
                        [0.        , 1.41421356, 0.        , 2.        ],
                        [0.        , 1.41421356, 2.        , 0.        ]])
                >>> lab = Datalab(data={"X": X, "y": y}, label_name="y")
                >>> lab.find_issues(knn_graph=knn_graph)

        - Configuring issue types:
            Suppose you want to only consider label issues. Just pass a dictionary with the key "label" and an empty dictionary as the value (to use default label issue parameters).

            .. code-block:: python

                >>> issue_types = {"label": {}}
                >>> # lab.find_issues(pred_probs=pred_probs, issue_types=issue_types)

            If you are advanced user who wants greater control, you can pass keyword arguments to the issue manager that handles the label issues.
            For example, if you want to pass the keyword argument "clean_learning_kwargs"
            to the constructor of the :py:class:`LabelIssueManager <cleanlab.datalab.issue_manager.label.LabelIssueManager>`, you would pass:


            .. code-block:: python

                >>> issue_types = {
                ...     "label": {
                ...         "clean_learning_kwargs": {
                ...             "prune_method": "prune_by_noise_rate",
                ...         },
                ...     },
                ... }
                >>> # lab.find_issues(pred_probs=pred_probs, issue_types=issue_types)

        """
        issue_finder = IssueFinder(datalab=self, verbosity=self.verbosity)
        issue_finder.find_issues(
            pred_probs=pred_probs,
            features=features,
            knn_graph=knn_graph,
            issue_types=issue_types,
        )

    def report(
        self,
        *,
        num_examples: int = 5,
        verbosity: Optional[int] = None,
        include_description: bool = True,
        show_summary_score: bool = False,
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

        See Also
        --------
        For advanced usage, see documentation for the
        :py:class:`Reporter <cleanlab.datalab.report.Reporter>` class.
        """
        if verbosity is None:
            verbosity = self.verbosity
        reporter = Reporter(
            data_issues=self.data_issues,
            verbosity=verbosity,
            include_description=include_description,
            show_summary_score=show_summary_score,
        )
        reporter.report(num_examples=num_examples)

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

        This is a wrapper around the ``DataIssues.issue_summary`` attribute.

        Examples
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

        This is a wrapper around the ``DataIssues.info`` attribute.

        Examples
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

    def get_issues(self, issue_name: Optional[str] = None) -> pd.DataFrame:
        """
        Use this after finding issues to see which examples suffer from which types of issues.

        NOTE
        ----
        This is a wrapper around the :py:meth:`DataIssues.get_issues <cleanlab.datalab.data_issues.DataIssues.get_issues>` method.

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
            whether this example exhibits a particular type of issue, and how severely (via a numeric quality score where lower values indicate more severe instances of the issue).
            The quality scores lie between 0-1 and are directly comparable between examples (for the same issue type), but not across different issue types.

            Additional columns may be present in the DataFrame depending on the type of issue specified.
        """
        return self.data_issues.get_issues(issue_name=issue_name)

    def get_issue_summary(self, issue_name: Optional[str] = None) -> pd.DataFrame:
        """Summarize the issues found in dataset of a particular type,
        including how severe this type of issue is overall across the dataset.

        NOTE
        ----
        This is a wrapper around the
        :py:meth:`DataIssues.get_issue_summary <cleanlab.datalab.data_issues.DataIssues.get_issue_summary>` method.

        Parameters
        ----------
        issue_name :
            Name of the issue type to summarize. If `None`, summarizes each of the different issue types previously considered in the audit.

        Returns
        -------
        issue_summary :
            DataFrame where each row corresponds to a type of issue, and columns quantify:
            the number of examples in the dataset estimated to exhibit this type of issue,
            and the overall severity of the issue across the dataset (via a numeric quality score where lower values indicate that the issue is overall more severe).
            The quality scores lie between 0-1 and are directly comparable between multiple datasets (for the same issue type), but not across different issue types.
        """
        return self.data_issues.get_issue_summary(issue_name=issue_name)

    def get_info(self, issue_name: Optional[str] = None) -> Dict[str, Any]:
        """Get the info for the issue_name key.

        This function is used to get the info for a specific issue_name. If the info is not computed yet, it will raise an error.

        NOTE
        ----
        This is a wrapper around the
        :py:meth:`DataIssues.get_info <cleanlab.datalab.data_issues.DataIssues.get_info>` method.

        Parameters
        ----------
        issue_name :
            The issue name for which the info is required.

        Returns
        -------
        :py:meth:`info <cleanlab.datalab.data_issues.DataIssues.get_info>` :
            The info for the issue_name.
        """
        return self.data_issues.get_info(issue_name)

    @staticmethod
    def list_possible_issue_types() -> List[str]:
        """Returns a list of all registered issue types.

        Any issue type that is not in this list cannot be used in the :py:meth:`find_issues` method.

        Note
        ----
        This method is a wrapper around :py:meth:`IssueFinder.list_possible_issue_types <cleanlab.datalab.issue_finder.IssueFinder.list_possible_issue_types>`.

        See Also
        --------
        :py:class:`REGISTRY <cleanlab.datalab.factory.REGISTRY>` : All available issue types and their corresponding issue managers can be found here.
        """
        return IssueFinder.list_possible_issue_types()

    @staticmethod
    def list_default_issue_types() -> List[str]:
        """Returns a list of the issue types that are run by default
        when :py:meth:`find_issues` is called without specifying `issue_types`.

        Note
        ----
        This method is a wrapper around :py:meth:`IssueFinder.list_default_issue_types <cleanlab.datalab.issue_finder.IssueFinder.list_default_issue_types>`.

        See Also
        --------
        :py:class:`REGISTRY <cleanlab.datalab.factory.REGISTRY>` : All available issue types and their corresponding issue managers can be found here.
        """
        return IssueFinder.list_default_issue_types()

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
