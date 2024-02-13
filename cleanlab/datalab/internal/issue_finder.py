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
Module for the :class:`IssueFinder` class, which is responsible for configuring,
creating and running issue managers.

It determines which types of issues to look for, instatiates the IssueManagers
via a factory, run the issue managers
(:py:meth:`IssueManager.find_issues <cleanlab.datalab.internal.issue_manager.issue_manager.IssueManager.find_issues>`),
and collects the results to :py:class:`DataIssues <cleanlab.datalab.internal.data_issues.DataIssues>`.

.. note::

    This module is not intended to be used directly. Instead, use the public-facing
    :py:meth:`Datalab.find_issues <cleanlab.datalab.datalab.Datalab.find_issues>` method.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from scipy.sparse import csr_matrix

from cleanlab.datalab.internal.issue_manager_factory import (
    _IssueManagerFactory,
    list_default_issue_types,
)
from cleanlab.datalab.internal.model_outputs import (
    MultiClassPredProbs,
    RegressionPredictions,
    MultiLabelPredProbs,
)
from cleanlab.datalab.internal.task import Task

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from typing import Callable

    from cleanlab.datalab.datalab import Datalab


_CLASSIFICATION_ARGS_DICT = {
    "label": ["pred_probs", "features"],
    "outlier": ["pred_probs", "features", "knn_graph"],
    "near_duplicate": ["features", "knn_graph"],
    "non_iid": ["pred_probs", "features", "knn_graph"],
    "underperforming_group": ["pred_probs", "features", "knn_graph", "cluster_ids"],
    "data_valuation": ["knn_graph"],
    "class_imbalance": [],
    "null": ["features"],
}
_REGRESSION_ARGS_DICT = {
    "label": ["features", "predictions"],
    "outlier": ["features", "knn_graph"],
    "near_duplicate": ["features", "knn_graph"],
    "non_iid": ["features", "knn_graph"],
    "null": ["features"],
}

_MULTILABEL_ARGS_DICT = {
    "label": ["pred_probs"],
    "outlier": ["features", "knn_graph"],
    "near_duplicate": ["features", "knn_graph"],
    "non_iid": ["features", "knn_graph"],
    "null": ["features"],
}


def _resolve_required_args_for_classification(**kwargs):
    """Resolves the required arguments for each issue type intended for classification tasks."""
    initial_args_dict = _CLASSIFICATION_ARGS_DICT.copy()
    args_dict = {
        issue_type: {arg: kwargs.get(arg, None) for arg in initial_args_dict[issue_type]}
        for issue_type in initial_args_dict
    }

    # Some issue types (like class-imbalance) have no required args.
    # This conditional lambda is used to include them in args dict.
    keep_empty_argument = lambda k: not len(_CLASSIFICATION_ARGS_DICT[k])

    # Remove None values from argument list, rely on default values in IssueManager
    args_dict = {
        k: {k2: v2 for k2, v2 in v.items() if v2 is not None}
        for k, v in args_dict.items()
        if (v or keep_empty_argument(k))
    }

    # Prefer `knn_graph` over `features` if both are provided.
    for v in args_dict.values():
        if "cluster_ids" in v and ("knn_graph" in v or "features" in v):
            warnings.warn(
                "`cluster_ids` have been provided with `knn_graph` or `features`."
                "Issue managers that require cluster labels will prefer"
                "`cluster_ids` over computation of cluster labels using"
                "`knn_graph` or `features`. "
            )
        if "knn_graph" in v and "features" in v:
            warnings.warn(
                "Both `features` and `knn_graph` were provided. "
                "Most issue managers will likely prefer using `knn_graph` "
                "instead of `features` for efficiency."
            )

    # Only keep issue types that have at least one argument
    # or those that require no arguments.
    args_dict = {k: v for k, v in args_dict.items() if (v or keep_empty_argument(k))}

    return args_dict


def _resolve_required_args_for_regression(**kwargs):
    """Resolves the required arguments for each issue type intended for regression tasks."""
    initial_args_dict = _REGRESSION_ARGS_DICT.copy()
    args_dict = {
        issue_type: {arg: kwargs.get(arg, None) for arg in initial_args_dict[issue_type]}
        for issue_type in initial_args_dict
    }
    # Some issue types have no required args.
    # This conditional lambda is used to include them in args dict.
    keep_empty_argument = lambda k: not len(_REGRESSION_ARGS_DICT[k])

    # Remove None values from argument list, rely on default values in IssueManager
    args_dict = {
        k: {k2: v2 for k2, v2 in v.items() if v2 is not None}
        for k, v in args_dict.items()
        if v or keep_empty_argument(k)
    }

    # Only keep issue types that have at least one argument
    # or those that require no arguments.
    args_dict = {k: v for k, v in args_dict.items() if (v or keep_empty_argument(k))}

    return args_dict


def _resolve_required_args_for_multilabel(**kwargs):
    """Resolves the required arguments for each issue type intended for multilabel tasks."""
    initial_args_dict = _MULTILABEL_ARGS_DICT.copy()
    args_dict = {
        issue_type: {arg: kwargs.get(arg, None) for arg in initial_args_dict[issue_type]}
        for issue_type in initial_args_dict
    }
    # Some issue types have no required args.
    # This conditional lambda is used to include them in args dict.
    keep_empty_argument = lambda k: not len(_MULTILABEL_ARGS_DICT[k])

    # Remove None values from argument list, rely on default values in IssueManager
    args_dict = {
        k: {k2: v2 for k2, v2 in v.items() if v2 is not None}
        for k, v in args_dict.items()
        if v or keep_empty_argument(k)  # Allow label issues to require no arguments
    }

    # Only keep issue types that have at least one argument
    # or those that require no arguments.
    args_dict = {k: v for k, v in args_dict.items() if (v or keep_empty_argument(k))}

    return args_dict


def _select_strategy_for_resolving_required_args(task: Task) -> Callable:
    """Helper function that selects the strategy for resolving required arguments for each issue type.

    Each strategy resolves the required arguments for each issue type.

    This is a helper function that filters out any issue manager
    that does not have the required arguments.

    This does not consider custom hyperparameters for each issue type.

    Parameters
    ----------
    task : str
        The type of machine learning task that the dataset is used for.

    Returns
    -------
    args_dict :
        Dictionary of required arguments for each issue type, if available.
    """
    strategies = {
        Task.CLASSIFICATION: _resolve_required_args_for_classification,
        Task.REGRESSION: _resolve_required_args_for_regression,
        Task.MULTILABEL: _resolve_required_args_for_multilabel,
    }
    selected_strategy = strategies.get(task, None)
    if selected_strategy is None:
        raise ValueError(f"No strategy for resolving required arguments for task '{task}'")
    return selected_strategy


class IssueFinder:
    """
    The IssueFinder class is responsible for managing the process of identifying
    issues in the dataset by handling the creation and execution of relevant
    IssueManagers. It serves as a coordinator or helper class for the Datalab class
    to encapsulate the specific behavior of the issue finding process.

    At a high level, the IssueFinder is responsible for:

    - Determining which types of issues to look for.
    - Instantiating the appropriate IssueManagers using a factory.
    - Running the IssueManagers' `find_issues` methods.
    - Collecting the results into a DataIssues instance.

    Parameters
    ----------
    datalab : Datalab
        The Datalab instance associated with this IssueFinder.

    task : str
        The type of machine learning task that the dataset is used for.

    verbosity : int
        Controls the verbosity of the output during the issue finding process.

    Note
    ----
    This class is not intended to be used directly. Instead, use the
    `Datalab.find_issues` method which internally utilizes an IssueFinder instance.
    """

    def __init__(self, datalab: "Datalab", task: Task, verbosity=1):
        self.datalab = datalab
        self.task = task
        self.verbosity = verbosity

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
        This method is not intended to be used directly. Instead, use the
        :py:meth:`Datalab.find_issues <cleanlab.datalab.datalab.Datalab.find_issues>` method.

        Note
        ----
        The issues are saved in the ``self.datalab.data_issues.issues`` attribute, but are not returned.

        Parameters
        ----------
        pred_probs :
            Out-of-sample predicted class probabilities made by the model for every example in the dataset.
            To best detect label issues, provide this input obtained from the most accurate model you can produce.

            If provided for classification, this must be a 2D array with shape ``(num_examples, K)`` where K is the number of classes in the dataset.
            If provided for regression, this must be a 1D array with shape ``(num_examples,)``.

        features : Optional[np.ndarray]
            Feature embeddings (vector representations) of every example in the dataset.

            If provided, this must be a 2D array with shape (num_examples, num_features).

        knn_graph :
            Sparse matrix representing distances between examples in the dataset in a k nearest neighbor graph.

            For details, refer to the documentation of the same argument in :py:class:`Datalab.find_issues <cleanlab.datalab.datalab.Datalab.find_issues>`

        issue_types :
            Collection specifying which types of issues to consider in audit and any non-default parameter settings to use.
            If unspecified, a default set of issue types and recommended parameter settings is considered.

            This is a dictionary of dictionaries, where the keys are the issue types of interest
            and the values are dictionaries of parameter values that control how each type of issue is detected (only for advanced users).
            More specifically, the values are constructor keyword arguments passed to the corresponding ``IssueManager``,
            which is responsible for detecting the particular issue type.

            .. seealso::
                :py:class:`IssueManager <cleanlab.datalab.internal.issue_manager.issue_manager.IssueManager>`
        """

        issue_types_copy = self.get_available_issue_types(
            pred_probs=pred_probs,
            features=features,
            knn_graph=knn_graph,
            issue_types=issue_types,
        )

        if not issue_types_copy:
            return None

        new_issue_managers = [
            factory(datalab=self.datalab, **issue_types_copy.get(factory.issue_name, {}))
            for factory in _IssueManagerFactory.from_list(
                list(issue_types_copy.keys()), task=self.task
            )
        ]

        failed_managers = []
        data_issues = self.datalab.data_issues
        for issue_manager, arg_dict in zip(new_issue_managers, issue_types_copy.values()):
            try:
                if self.verbosity:
                    print(f"Finding {issue_manager.issue_name} issues ...")
                issue_manager.find_issues(**arg_dict)
                data_issues.collect_statistics(issue_manager)
                data_issues.collect_issues_from_issue_manager(issue_manager)
            except Exception as e:
                print(f"Error in {issue_manager.issue_name}: {e}")
                failed_managers.append(issue_manager)
        if failed_managers:
            print(f"Failed to check for these issue types: {failed_managers}")
        data_issues.set_health_score()

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
            # keep only default issue types
            issue_types_copy = {
                issue: issue_types_copy[issue]
                for issue in list_default_issue_types(self.task)
                if issue in issue_types_copy
            }

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

    def get_available_issue_types(self, **kwargs):
        """Returns a dictionary of issue types that can be used in :py:meth:`Datalab.find_issues
        <cleanlab.datalab.datalab.Datalab.find_issues>` method."""

        pred_probs = kwargs.get("pred_probs", None)
        features = kwargs.get("features", None)
        knn_graph = kwargs.get("knn_graph", None)
        issue_types = kwargs.get("issue_types", None)

        model_output = None
        if pred_probs is not None:
            model_output_dict = {
                Task.REGRESSION: RegressionPredictions,
                Task.CLASSIFICATION: MultiClassPredProbs,
                Task.MULTILABEL: MultiLabelPredProbs,
            }

            model_output_class = model_output_dict.get(self.task)
            if model_output_class is None:
                raise ValueError(f"Unknown task type '{self.task}'")

            model_output = model_output_class(pred_probs)

        if model_output is not None:
            # A basic trick to assign the model output to the correct argument
            # E.g. Datalab accepts only `pred_probs`, but those are assigned to the `predictions` argument for regression-related issue_managers
            kwargs.update({model_output.argument: model_output.collect()})

        # Determine which parameters are required for each issue type
        strategy_for_resolving_required_args = _select_strategy_for_resolving_required_args(
            self.task
        )
        required_args_per_issue_type = strategy_for_resolving_required_args(**kwargs)

        issue_types_copy = self._set_issue_types(issue_types, required_args_per_issue_type)
        if issue_types is None:
            # Only run default issue types if no issue types are specified
            issue_types_copy = {
                issue: issue_types_copy[issue]
                for issue in list_default_issue_types(self.task)
                if issue in issue_types_copy
            }
        drop_label_check = (
            "label" in issue_types_copy
            and not self.datalab.has_labels
            and self.task != Task.REGRESSION
        )

        if drop_label_check:
            warnings.warn("No labels were provided. " "The 'label' issue type will not be run.")
            issue_types_copy.pop("label")

        outlier_check_needs_features = (
            self.task == "classification"
            and "outlier" in issue_types_copy
            and not self.datalab.has_labels
        )
        if outlier_check_needs_features:
            no_features = features is None
            no_knn_graph = knn_graph is None
            pred_probs_given = issue_types_copy["outlier"].get("pred_probs", None) is not None

            only_pred_probs_given = pred_probs_given and no_features and no_knn_graph
            if only_pred_probs_given:
                warnings.warn(
                    "No labels were provided. " "The 'outlier' issue type will not be run."
                )
                issue_types_copy.pop("outlier")

        drop_class_imbalance_check = (
            "class_imbalance" in issue_types_copy
            and not self.datalab.has_labels
            and self.task == Task.CLASSIFICATION
        )
        if drop_class_imbalance_check:
            issue_types_copy.pop("class_imbalance")

        return issue_types_copy
