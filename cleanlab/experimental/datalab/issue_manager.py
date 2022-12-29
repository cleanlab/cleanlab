from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from cleanlab.classification import CleanLearning
from cleanlab.outlier import OutOfDistribution

import pandas as pd
import numpy as np

from cleanlab.internal.validation import assert_valid_inputs

if TYPE_CHECKING:  # pragma: no cover
    from cleanlab.experimental.datalab.datalab import Datalab  # pragma: no cover


class IssueManager(ABC):
    """Base class for managing data issues of a particular type in a Datalab.

    For each example in a dataset, the IssueManager for a particular type of issue should compute:
    - A numeric severity score between 0 and 1,
        with values near 0 indicating severe instances of the issue.
    - A boolean `is_issue` value, which is True
        if we believe this example suffers from the issue in question.
      `is_issue` may be determined by thresholding the severity score
        (with an a priori determined reasonable threshold value),
        or via some other means (e.g. Confident Learning for flagging label issues).

    The IssueManager should also report:
    - A global value between 0 and 1 summarizing how severe this issue is in the dataset overall
        (e.g. the average severity across all examples in dataset
        or count of examples where `is_issue=True`).
    - Other interesting `info` about the issue and examples in the dataset,
      and statistics estimated from current dataset that may be reused
      to score this issue in future data.
      For example, `info` for label issues could contain the:
      confident_thresholds, confident_joint, predicted label for each example, etc.
      Another example is for (near)-duplicate detection issue, where `info` could contain:
      which set of examples in the dataset are all (nearly) identical.

    Implementing a new IssueManager:
    - Define the `issue_key` class attribute, e.g. "label", "duplicate", "ood", etc.
    - Implement the abstract methods `find_issues` and `collect_info`.
      - `find_issues` is responsible for computing computing the `issues` and `summary` dataframes.
      - `collect_info` is responsible for computing the `info` dict. It is called by `find_issues`,
        once the manager has set the `issues` and `summary` dataframes as instance attributes.
    """

    def __init__(self, datalab: Datalab):
        self.datalab = datalab
        self.info: Dict[str, Any] = {}
        self.issues: pd.DataFrame = pd.DataFrame()
        self.summary: pd.DataFrame = pd.DataFrame()

    def __repr__(self):
        class_name = self.__class__.__name__
        return class_name

    @classmethod
    @property
    @abstractmethod
    def issue_key(cls) -> str:
        """Returns a key that is used to store issue summary results about the assigned Lab."""
        raise NotImplementedError

    @classmethod
    @property
    def issue_score_key(cls) -> str:
        """Returns a key that is used to store issue score results about the assigned Lab."""
        # TODO: The score key should just be f"{cls.issue_key}_score" or f"{cls.issue_key}_quality_score", f"{cls.issue_key}_quality"
        return f"{cls.issue_key}_score"

    @abstractmethod
    def find_issues(self, /, *args, **kwargs) -> None:
        """Finds occurrences of this particular issue in the dataset.

        Computes the `issues` and `summary` dataframes. Calls `collect_info` to compute the `info` dict.
        """
        raise NotImplementedError

    @abstractmethod
    def collect_info(self, /, *args, **kwargs) -> dict:
        """Collects data for the info attribute of the Datalab.

        NOTE
        ----
        This method is called by `find_issues` after `find_issues` has set the `issues` and `summary` dataframes
        as instance attributes.
        """
        raise NotImplementedError

    def get_summary(self):
        assert not self.issues.empty
        assert len(self.info) > 0


class HealthIssueManager(IssueManager):
    @classmethod
    @property
    def issue_key(cls) -> str:
        return "health"

    def find_issues(self, pred_probs: np.ndarray, **kwargs) -> dict:
        health_summary = self._health_summary(pred_probs=pred_probs, **kwargs)
        self.collect_info(summary=health_summary)
        return health_summary

    def collect_info(self, summary: Dict[str, Any], **kwargs) -> None:
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
    """Manages label issues in a Datalab.

    Parameters
    ----------
    datalab :
        A Datalab instance.

    clean_learning_kwargs :
        Keyword arguments to pass to the CleanLearning constructor.
    """

    @classmethod
    @property
    def issue_key(cls) -> str:
        return "label"

    @classmethod
    @property
    def issue_score_key(cls) -> str:
        return "label_quality"

    @classmethod
    @property
    def info_keys(cls) -> List[str]:
        return ["given_label", "predicted_label"]

    def __init__(
        self,
        datalab: Datalab,
        clean_learning_kwargs: Optional[Dict[str, Any]] = None,
        health_summary_parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(datalab)
        self.cl = CleanLearning(**(clean_learning_kwargs or {}))
        self.health_summary_parameters: Dict[str, Any] = health_summary_parameters or {}
        self.reset()

    def reset(self) -> None:
        """Reset the attributes of this manager based on the available datalab info
        and the keyword arguments stored as instance attributes.

        This allows the builder to use pre-computed info from the datalab to speed up
        some computations in the `find_issues` method.
        """
        if not self.health_summary_parameters:
            self.health_summary_parameters = {
                "labels": self.datalab._labels,
                "asymmetric": self.datalab.info["data"].get("asymmetric", None),
                "class_names": list(self.datalab._label_map.values()),
                "num_examples": self.datalab.info["data"].get("num_examples"),
                "joint": self.datalab.info["data"].get("joint", None),
                "confident_joint": self.datalab.info["data"].get("confident_joint", None),
                "multi_label": self.datalab.info["data"].get("multi_label", None),
            }
        self.health_summary_parameters = {
            k: v for k, v in self.health_summary_parameters.items() if v is not None
        }

    def find_issues(self, pred_probs: np.ndarray, model=None, **kwargs) -> None:
        if pred_probs is None and model is not None:
            raise NotImplementedError("TODO: We assume pred_probs is provided.")

        self.health_summary_parameters.update({"pred_probs": pred_probs})
        # Find examples with label issues
        self.issues = self.cl.find_label_issues(labels=self.datalab._labels, pred_probs=pred_probs)

        summary_dict = self.get_health_summary(pred_probs=pred_probs, **kwargs)

        # Get a summarized dataframe of the label issues
        self.summary = pd.DataFrame(
            {
                "issue_type": [self.issue_key],
                "score": [summary_dict["overall_label_health_score"]],
            },
        )

        # Collect info about the label issues
        self.info = self.collect_info(issues=self.issues, summary_dict=summary_dict)

        # Drop drop column from issues that are in the info
        self.issues = self.issues.drop(columns=self.info_keys)

    def get_health_summary(self, pred_probs, **kwargs) -> dict:
        """Returns a short summary of the health of this Lab."""
        from cleanlab.dataset import health_summary

        # Validate input
        self._validate_pred_probs(pred_probs)

        summary_kwargs = self._get_summary_parameters(pred_probs, **kwargs)
        summary = health_summary(**summary_kwargs)
        return summary

    def _get_summary_parameters(self, pred_probs, **kwargs) -> Dict["str", Any]:
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
        elif (
            "joint" in self.health_summary_parameters
            and "num_examples" in self.health_summary_parameters
        ):
            summary_parameters = {
                "joint": self.health_summary_parameters["joint"],
                "num_examples": self.health_summary_parameters["num_examples"],
            }
        else:
            summary_parameters = {
                "pred_probs": pred_probs,
                "labels": self.datalab._labels,
            }

        summary_parameters["class_names"] = self.health_summary_parameters["class_names"]

        for k in ["asymmetric", "verbose"]:
            # Start with the health_summary_parameters, then override with kwargs
            if k in self.health_summary_parameters:
                summary_parameters[k] = self.health_summary_parameters[k]
            if k in kwargs:
                summary_parameters[k] = kwargs[k]
        return summary_parameters

    def collect_info(self, issues: pd.DataFrame, summary_dict: dict) -> dict:
        issues_info = {
            "num_label_issues": sum(issues[f"is_{self.issue_key}_issue"]),
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
        }

        return info_dict

    def _validate_pred_probs(self, pred_probs) -> None:
        assert_valid_inputs(X=None, y=self.datalab._labels, pred_probs=pred_probs)

    @property
    def verbosity_levels(self) -> Dict[int, Any]:
        return {
            0: {},
            1: {"summary": ["confident_joint"]},
            2: {"issue": ["given_label", "predicted_label"]},
        }


class OutOfDistributionIssueManager(IssueManager):
    """Manages issues related to out-of-distribution examples."""

    @classmethod
    @property
    def issue_key(cls) -> str:
        return "ood"

    @classmethod
    @property
    def issue_score_key(cls) -> str:
        return "ood_score"

    @classmethod
    @property
    def info_keys(cls) -> List[str]:
        return []

    def __init__(
        self,
        datalab: Datalab,
        ood_kwargs: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ):
        super().__init__(datalab)
        self.ood: OutOfDistribution = OutOfDistribution(**(ood_kwargs or {}))
        self.threshold = threshold

    def find_issues(
        self,
        features: Optional[List[str]] = None,
        pred_probs: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:

        if features is not None:
            scores = self._score_with_features(features, **kwargs)
        elif pred_probs is not None:
            scores = self._score_with_pred_probs(pred_probs, **kwargs)
        else:
            raise ValueError(f"Either features or pred_probs must be provided.")

        if self.threshold is None:
            # 10th percentile of scores
            self.threshold = np.percentile(scores, 10)

        self.issues = pd.DataFrame(
            {
                "is_ood_issue": scores < self.threshold,
                self.issue_score_key: scores,
            },
        )

        self.summary = pd.DataFrame(
            {
                "issue_type": [self.issue_key],
                "score": [scores.mean()],
            },
        )

        self.info = self.collect_info()

    def _score_with_pred_probs(self, pred_probs: np.ndarray, **kwargs) -> np.ndarray:
        scores = self.ood.fit_score(pred_probs=pred_probs, labels=self.datalab._labels, **kwargs)
        return scores

    def _score_with_features(self, features: List[str], **kwargs) -> np.ndarray:
        embeddings = self._extract_embeddings(columns=features, **kwargs)

        scores = self.ood.fit_score(features=embeddings)
        return scores

    def collect_info(self) -> dict:

        issues_dict = {
            "num_ood_issues": sum(self.issues["is_ood_issue"]),
            "average_ood_score": self.issues[self.issue_score_key].mean(),
        }
        pred_probs_issues_dict = {}  # TODO: Implement collect_info for pred_probs related issues
        feature_issues_dict = {}

        # Compute
        if self.ood.params["knn"] is not None:
            knn = self.ood.params["knn"]
            dists, nn_ids = [array[:, 0] for array in knn.kneighbors()]
            weighted_knn_graph = knn.kneighbors_graph(mode="distance").toarray()

            # TODO: Reverse the order of the calls to knn.kneighbors() and knn.kneighbors_graph()
            #   to avoid computing the (distance, id) pairs twice.
            feature_issues_dict.update(
                {
                    "nearest_neighbour": nn_ids[:, 0].tolist(),
                    "distance_to_nearest_neighbour": dists[:, 0].tolist(),
                    # TODO Check scipy-dependency
                    "weighted_knn_graph": weighted_knn_graph.tolist(),
                }
            )

        if self.ood.params["confident_thresholds"] is not None:
            pass  #
        ood_params_dict = self.ood.params
        knn_dict = {
            **pred_probs_issues_dict,
            **feature_issues_dict,
        }
        info_dict = {
            **issues_dict,
            **ood_params_dict,
            **knn_dict,
        }
        return info_dict

    # TODO: Update annotation for columns and related args in other methods
    def _extract_embeddings(self, columns: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Extracts embeddings for the given columns."""

        if isinstance(columns, list):
            raise NotImplementedError("TODO: Support list of columns.")

        format_kwargs = kwargs.get("format_kwargs", {})

        return self.datalab.data.with_format("numpy", **format_kwargs)[columns]
