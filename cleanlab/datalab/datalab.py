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

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

import cleanlab
from cleanlab.datalab.internal.adapter.imagelab import create_imagelab
from cleanlab.datalab.internal.data import Data
from cleanlab.datalab.internal.display import _Displayer
from cleanlab.datalab.internal.helper_factory import (
    _DataIssuesBuilder,
    issue_finder_factory,
    report_factory,
)
from cleanlab.datalab.internal.issue_manager_factory import (
    list_default_issue_types as _list_default_issue_types,
    list_possible_issue_types as _list_possible_issue_types,
)
from cleanlab.datalab.internal.serialize import _Serializer
from cleanlab.datalab.internal.task import Task

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

    task : str
        The type of machine learning task that the dataset is used for.

        Supported tasks:
          - "classification" (default): Multiclass classification
          - "regression" : Regression
          - "multilabel" : Multilabel classification

    label_name : str, optional
        The name of the label column in the dataset.

    image_key : str, optional
        Optional key that can be specified for image datasets to point to the field containing the actual images themselves.
        If specified, additional image-specific issue types can be detected in the dataset.
        See the CleanVision package `documentation <https://cleanvision.readthedocs.io/en/latest/>`_ for descriptions of these image-specific issue types.

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
        task: str = "classification",
        label_name: Optional[str] = None,
        image_key: Optional[str] = None,
        verbosity: int = 1,
    ) -> None:
        # Assume continuous values of labels for regression task
        # Map labels to integers for classification task
        self.task = Task.from_str(task)
        self._data = Data(data, self.task, label_name)
        self.data = self._data._data
        self._labels = self._data.labels
        self._label_map = self._labels.label_map
        self.label_name = self._labels.label_name
        self._data_hash = self._data._data_hash
        self.cleanlab_version = cleanlab.version.__version__
        self.verbosity = verbosity
        self._imagelab = create_imagelab(dataset=self.data, image_key=image_key)

        # Create the builder for DataIssues
        builder = _DataIssuesBuilder(self._data)
        builder.set_imagelab(self._imagelab).set_task(self.task)
        self.data_issues = builder.build()

    # todo: check displayer methods
    def __repr__(self) -> str:
        return _Displayer(data_issues=self.data_issues, task=self.task).__repr__()

    def __str__(self) -> str:
        return _Displayer(data_issues=self.data_issues, task=self.task).__str__()

    @property
    def labels(self) -> Union[np.ndarray, List[List[int]]]:
        """Labels of the dataset, in a [0, 1, ..., K-1] format."""
        return self._labels.labels

    @property
    def has_labels(self) -> bool:
        """Whether the dataset has labels, and that they are in a [0, 1, ..., K-1] format."""
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

        NOTE
        ----
        The issues are saved in the ``self.issues`` attribute of the ``Datalab`` object, but are not returned.

        Parameters
        ----------
        pred_probs :
            Out-of-sample predicted class probabilities made by the model for every example in the dataset.
            To best detect label issues, provide this input obtained from the most accurate model you can produce.

            For classification data, this must be a 2D array with shape ``(num_examples, K)`` where ``K`` is the number of classes in the dataset.
            Make sure that the columns of your `pred_probs` are properly ordered with respect to the ordering of classes, which for Datalab is: lexicographically sorted by class name.

            For regression data, this must be a 1D array with shape ``(num_examples,)`` containing the predicted value for each example.

            For multilabel classification data, this must be a 2D array with shape ``(num_examples, K)`` where ``K`` is the number of classes in the dataset.
                Make sure that the columns of your `pred_probs` are properly ordered with respect to the ordering of classes, which for Datalab is: lexicographically sorted by class name.


        features : Optional[np.ndarray]
            Feature embeddings (vector representations) of every example in the dataset.

            If provided, this must be a 2D array with shape (num_examples, num_features).

        knn_graph :
            Sparse matrix of precomputed distances between examples in the dataset in a k nearest neighbor graph.

            If provided, this must be a square CSR matrix with shape ``(num_examples, num_examples)`` and ``(k*num_examples)`` non-zero entries (``k`` is the number of nearest neighbors considered for each example),
            evenly distributed across the rows.
            Each non-zero entry in this matrix is a distance between a pair of examples in the dataset. Self-distances must be omitted
            (i.e. diagonal must be all zeros, k nearest neighbors for each example do not include the example itself).

            This CSR format uses three 1D arrays (`data`, `indices`, `indptr`) to store a 2D matrix ``M``:

            - `data`: 1D array containing all the non-zero elements of matrix ``M``, listed in a row-wise fashion (but sorted within each row).
            - `indices`: 1D array storing the column indices in matrix ``M`` of these non-zero elements. Each entry in `indices` corresponds to an entry in `data`, indicating the column of ``M`` containing this entry.
            - `indptr`: 1D array indicating the start and end indices in `data` for each row of matrix ``M``. The non-zero elements of the i-th row of ``M`` are stored from ``data[indptr[i]]`` to ``data[indptr[i+1]]``.

            Within each row of matrix ``M`` (defined by the ranges in `indptr`), the corresponding non-zero entries (distances) of `knn_graph` must be sorted in ascending order (specifically in the segments of the `data` array that correspond to each row of ``M``). The `indices` array must also reflect this ordering, maintaining the correct column positions for these sorted distances.

            This type of matrix is returned by the method: `sklearn.neighbors.NearestNeighbors.kneighbors_graph <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors_graph>`_.

            Below is an example to illustrate:

            .. code-block:: python

                knn_graph.todense()
                # matrix([[0. , 0.3, 0.2],
                #         [0.3, 0. , 0.4],
                #         [0.2, 0.4, 0. ]])

                knn_graph.data
                # array([0.2, 0.3, 0.3, 0.4, 0.2, 0.4])
                # Here, 0.2 and 0.3 are the sorted distances in the first row, 0.3 and 0.4 in the second row, and so on.

                knn_graph.indices
                # array([2, 1, 0, 2, 0, 1])
                # Corresponding neighbor indices for the distances from the `data` array.

                knn_graph.indptr
                # array([0, 2, 4, 6])
                # The non-zero entries in the first row are stored from `knn_graph.data[0]` to `knn_graph.data[2]`, the second row from `knn_graph.data[2]` to `knn_graph.data[4]`, and so on.

            For any duplicated examples i,j whose distance is 0, there should be an *explicit* zero stored in the matrix, i.e. ``knn_graph[i,j] = 0``.

            If both `knn_graph` and `features` are provided, the `knn_graph` will take precendence.
            If `knn_graph` is not provided, it is constructed based on the provided `features`.
            If neither `knn_graph` nor `features` are provided, certain issue types like (near) duplicates will not be considered.

            .. seealso::
                See the
                `scipy.sparse.csr_matrix documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_
                for more details on the CSR matrix format.

        issue_types :
            Collection specifying which types of issues to consider in audit and any non-default parameter settings to use.
            If unspecified, a default set of issue types and recommended parameter settings is considered.

            This is a dictionary of dictionaries, where the keys are the issue types of interest
            and the values are dictionaries of parameter values that control how each type of issue is detected (only for advanced users).
            More specifically, the values are constructor keyword arguments passed to the corresponding ``IssueManager``,
            which is responsible for detecting the particular issue type.

            .. seealso::
                :py:class:`IssueManager <cleanlab.datalab.internal.issue_manager.issue_manager.IssueManager>`

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
            to the constructor of the :py:class:`LabelIssueManager <cleanlab.datalab.internal.issue_manager.label.LabelIssueManager>`, you would pass:


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

        if issue_types is not None and not issue_types:
            warnings.warn(
                "No issue types were specified so no issues will be found in the dataset. Set `issue_types` as None to consider a default set of issues."
            )
            return None
        issue_finder = issue_finder_factory(self._imagelab)(
            datalab=self, task=self.task, verbosity=self.verbosity
        )
        issue_finder.find_issues(
            pred_probs=pred_probs,
            features=features,
            knn_graph=knn_graph,
            issue_types=issue_types,
        )

        if self.verbosity:
            print(
                f"\nAudit complete. {self.data_issues.issue_summary['num_issues'].sum()} issues found in the dataset."
            )

    def report(
        self,
        *,
        num_examples: int = 5,
        verbosity: Optional[int] = None,
        include_description: bool = True,
        show_summary_score: bool = False,
        show_all_issues: bool = False,
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

        show_summary_score :
            Whether or not to include the overall severity score of each issue type in the report.
            These scores are not comparable across different issue types,
            see the ``issue_summary`` documentation to learn more.

        show_all_issues :
            Whether or not the report should show all issue types that were checked for, or only the types of issues detected in the dataset.
            With this set to ``True``, the report may include more types of issues that were not detected in the dataset.

        See Also
        --------
        For advanced usage, see documentation for the
        :py:class:`Reporter <cleanlab.datalab.internal.report.Reporter>` class.
        """
        if verbosity is None:
            verbosity = self.verbosity
        if self.data_issues.issue_summary.empty:
            print("Please specify some `issue_types` in datalab.find_issues() to see a report.\n")
            return

        reporter = report_factory(self._imagelab)(
            data_issues=self.data_issues,
            task=self.task,
            verbosity=verbosity,
            include_description=include_description,
            show_summary_score=show_summary_score,
            show_all_issues=show_all_issues,
            imagelab=self._imagelab,
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

        Each type of issue has a summary score, which is usually defined as an average of
        per-example issue-severity scores (over all examples in the dataset).
        So these summary scores are not directly tied to the number of examples estimated to exhibit
        a particular type of issue. Issue-severity (ie. quality of each example) is measured differently for each issue type,
        and these per-example scores are only comparable across different examples for the same issue-type, but are not comparable across different issue types.
        For instance, label quality might be scored via estimated likelihood of the given label,
        whereas outlier quality might be scored via distance to K-nearest-neighbors in feature space (fundamentally incomparable quantities).
        For some issue types, the summary score is not an average of per-example scores, but rather a global statistic of the dataset
        (eg. for `non_iid` issue type, the p-value for hypothesis test that data are IID).

        In summary, you can compare these summary scores across datasets for the same issue type, but never compare them across different issue types.

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

        See the documentation of the ``issue_summary`` attribute to learn more.

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

        Parameters
        ----------
        issue_name :
            The issue name for which the info is required.

        Returns
        -------
        :py:meth:`info <cleanlab.datalab.internal.data_issues.DataIssues.get_info>` :
            The info for the issue_name.
        """
        return self.data_issues.get_info(issue_name)

    def list_possible_issue_types(self) -> List[str]:
        """Returns a list of all registered issue types.

        Any issue type that is not in this list cannot be used in the :py:meth:`find_issues` method.

        See Also
        --------
        :py:class:`REGISTRY <cleanlab.datalab.internal.issue_manager_factory.REGISTRY>` : All available issue types and their corresponding issue managers can be found here.
        """
        return _list_possible_issue_types(task=self.task)

    def list_default_issue_types(self) -> List[str]:
        """Returns a list of the issue types that are run by default
        when :py:meth:`find_issues` is called without specifying `issue_types`.

        See Also
        --------
        :py:class:`REGISTRY <cleanlab.datalab.internal.issue_manager_factory.REGISTRY>` : All available issue types and their corresponding issue managers can be found here.
        """
        return _list_default_issue_types(task=self.task)

    def save(self, path: str, force: bool = False) -> None:
        """Saves this Datalab object to file (all files are in folder at `path/`).
        We do not guarantee saved Datalab can be loaded from future versions of cleanlab.

        Parameters
        ----------
        path :
            Folder in which all information about this Datalab should be saved.

        force :
            If ``True``, overwrites any existing files in the folder at `path`. Use this with caution!

        NOTE
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
