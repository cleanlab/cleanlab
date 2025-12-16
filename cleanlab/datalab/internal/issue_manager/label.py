from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

from cleanlab.classification import CleanLearning
from cleanlab.count import get_confident_thresholds
from cleanlab.datalab.internal.issue_manager import IssueManager
from cleanlab.internal.validation import assert_valid_inputs

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    import pandas as pd

    from cleanlab.datalab.datalab import Datalab


class LabelIssueManager(IssueManager):
    """Manages label issues in a Datalab.

    Parameters
    ----------
    datalab :
        A Datalab instance.

    k :
        The number of nearest neighbors to consider when computing pred_probs from features.
        Only applicable if features are provided and pred_probs are not.

    clean_learning_kwargs :
        Keyword arguments to pass to the :py:meth:`CleanLearning <cleanlab.classification.CleanLearning>` constructor.

    health_summary_parameters :
        Keyword arguments to pass to the :py:meth:`health_summary <cleanlab.dataset.health_summary>` function.
    """

    description: ClassVar[
        str
    ] = """Examples whose given label is estimated to be potentially incorrect
    (e.g. due to annotation error) are flagged as having label issues.
    """

    issue_name: ClassVar[str] = "label"
    verbosity_levels = {
        0: [],
        1: [],
        2: [],
        3: ["classes_by_label_quality", "overlapping_classes"],
    }

    def __init__(
        self,
        datalab: Datalab,
        k: int = 10,
        clean_learning_kwargs: Optional[Dict[str, Any]] = None,
        health_summary_parameters: Optional[Dict[str, Any]] = None,
        **_,
    ):
        super().__init__(datalab)
        self.cl = CleanLearning(**(clean_learning_kwargs or {}))
        self.k = k
        self.health_summary_parameters: Dict[str, Any] = (
            health_summary_parameters.copy() if health_summary_parameters else {}
        )
        self._find_issues_inputs: Dict[str, bool] = {"features": False, "pred_probs": False}
        self._reset()

    @staticmethod
    def _process_find_label_issues_kwargs(**kwargs) -> Dict[str, Any]:
        """Searches for keyword arguments that are meant for the
        CleanLearning.find_label_issues method call

        Examples
        --------
        >>> from cleanlab.datalab.internal.issue_manager.label import LabelIssueManager
        >>> LabelIssueManager._process_find_label_issues_kwargs(thresholds=[0.1, 0.9])
        {'thresholds': [0.1, 0.9]}
        """
        accepted_kwargs = [
            "thresholds",
            "noise_matrix",
            "inverse_noise_matrix",
            "save_space",
            "clf_kwargs",
            "validation_func",
        ]
        return {k: v for k, v in kwargs.items() if k in accepted_kwargs and v is not None}

    def _reset(self) -> None:
        """Reset the attributes of this manager based on the available datalab info
        and the keyword arguments stored as instance attributes.

        This allows the builder to use pre-computed info from the datalab to speed up
        some computations in the :py:meth:`find_issues` method.
        """
        if not self.health_summary_parameters:
            statistics_dict = self.datalab.get_info("statistics")
            self.health_summary_parameters = {
                "labels": self.datalab.labels,
                "class_names": list(self.datalab._label_map.values()),
                "num_examples": statistics_dict.get("num_examples"),
                "joint": statistics_dict.get("joint", None),
                "confident_joint": statistics_dict.get("confident_joint", None),
                "multi_label": statistics_dict.get("multi_label", None),
                "asymmetric": statistics_dict.get("asymmetric", None),
                "verbose": False,
            }
        self.health_summary_parameters = {
            k: v for k, v in self.health_summary_parameters.items() if v is not None
        }

    def find_issues(
        self,
        pred_probs: Optional[npt.NDArray] = None,
        features: Optional[npt.NDArray] = None,
        **kwargs,
    ) -> None:
        """Find label issues in the datalab.

        Parameters
        ----------
        pred_probs :
            The predicted probabilities for each example.

        features :
            The features for each example.
        """
        if pred_probs is not None:
            self._find_issues_inputs.update({"pred_probs": True})
        if pred_probs is None:
            self._find_issues_inputs.update({"features": True})
            if features is None:
                raise ValueError(
                    "Either pred_probs or features must be provided to find label issues."
                )
            # produce out-of-sample pred_probs from features
            labels = self.datalab.labels
            if not isinstance(labels, np.ndarray):
                error_msg = (
                    f"Expected labels to be a numpy array of shape (n_samples,) to use in LabelIssueManager, "
                    f"but got {type(labels)} instead."
                )
                raise TypeError(error_msg)

            knn = KNeighborsClassifier(n_neighbors=self.k + 1)
            knn.fit(features, labels)
            pred_probs = knn.predict_proba(features)

            encoder = OneHotEncoder()
            label_transform = labels.reshape(-1, 1)
            one_hot_label = encoder.fit_transform(label_transform)

            # adjust pred_probs so it is out-of-sample
            pred_probs = np.asarray(
                (pred_probs - 1 / (self.k + 1) * one_hot_label) * (self.k + 1) / self.k
            )

        self.health_summary_parameters.update({"pred_probs": pred_probs})
        # Find examples with label issues
        labels = self.datalab.labels
        self.issues = self.cl.find_label_issues(
            labels=labels,
            pred_probs=pred_probs,
            **self._process_find_label_issues_kwargs(**kwargs),
        )
        self.issues.rename(columns={"label_quality": self.issue_score_key}, inplace=True)

        summary_dict = self.get_health_summary(pred_probs=pred_probs)

        # Get a summarized dataframe of the label issues
        self.summary = self.make_summary(score=summary_dict["overall_label_health_score"])

        confident_thresholds = get_confident_thresholds(labels=labels, pred_probs=pred_probs)
        # Collect info about the label issues
        self.info = self.collect_info(
            issues=self.issues,
            summary_dict=summary_dict,
            confident_thresholds=confident_thresholds,
        )

        # Drop columns from issues that are in the info
        self.issues = self.issues.drop(columns=["given_label", "predicted_label"])

    def get_health_summary(self, pred_probs) -> dict:
        """Returns a short summary of the health of this Lab."""
        from cleanlab.dataset import health_summary

        # Validate input
        self._validate_pred_probs(pred_probs)

        summary_kwargs = self._get_summary_parameters(pred_probs)
        summary = health_summary(**summary_kwargs)
        return summary

    def _get_summary_parameters(self, pred_probs) -> Dict["str", Any]:
        """Collects a set of input parameters for the health summary function based on
        any info available in the datalab.

        Parameters
        ----------
        pred_probs :
            The predicted probabilities for each example.

        kwargs :
            Keyword arguments to pass to the health summary function.

        Returns
        -------
        summary_parameters :
            A dictionary of parameters to pass to the health summary function.
        """
        if "confident_joint" in self.health_summary_parameters:
            summary_parameters = {
                "confident_joint": self.health_summary_parameters["confident_joint"]
            }
        elif all([x in self.health_summary_parameters for x in ["joint", "num_examples"]]):
            summary_parameters = {
                k: self.health_summary_parameters[k] for k in ["joint", "num_examples"]
            }
        else:
            summary_parameters = {
                "pred_probs": pred_probs,
                "labels": self.datalab.labels,
            }

        summary_parameters["class_names"] = self.health_summary_parameters["class_names"]

        for k in ["asymmetric", "verbose"]:
            # Start with the health_summary_parameters, then override with kwargs
            if k in self.health_summary_parameters:
                summary_parameters[k] = self.health_summary_parameters[k]

        return (
            summary_parameters  # will be called in `dataset.health_summary(**summary_parameters)`
        )

    def collect_info(
        self, issues: pd.DataFrame, summary_dict: dict, confident_thresholds: np.ndarray
    ) -> dict:
        issues_info = {
            "num_label_issues": sum(issues[f"is_{self.issue_name}_issue"]),
            "average_label_quality": issues[self.issue_score_key].mean(),
            "given_label": issues["given_label"].tolist(),
            "predicted_label": issues["predicted_label"].tolist(),
        }

        health_summary_info = {
            "confident_joint": summary_dict["joint"],
            "classes_by_label_quality": summary_dict["classes_by_label_quality"],
            "overlapping_classes": summary_dict["overlapping_classes"],
        }

        cl_info = {}
        for k in self.cl.__dict__:
            if k not in ["py", "noise_matrix", "inverse_noise_matrix", "confident_joint"]:
                continue
            cl_info[k] = self.cl.__dict__[k]

        info_dict = {
            **issues_info,
            **health_summary_info,
            **cl_info,
            "confident_thresholds": confident_thresholds.tolist(),
            "find_issues_inputs": self._find_issues_inputs,
        }

        return info_dict

    def _validate_pred_probs(self, pred_probs) -> None:
        assert_valid_inputs(X=None, y=self.datalab.labels, pred_probs=pred_probs)
